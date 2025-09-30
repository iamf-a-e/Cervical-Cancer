import os
import re
import time
import json
import base64
import logging
import urllib.parse
import threading
from datetime import datetime, timedelta

import requests
import redis
from flask import Flask, request, jsonify, render_template
from sqlalchemy import create_engine, Column, Integer, String, DateTime, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from urlextract import URLExtract

import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted

from huggingface_hub import InferenceClient

# -----------------------------------------------------------------------------
# Logging setup
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)

# -----------------------------------------------------------------------------
# ENV VARS
# -----------------------------------------------------------------------------
wa_token = os.environ.get("WA_TOKEN")        # WhatsApp API Key
phone_id = os.environ.get("PHONE_ID")        # WhatsApp phone ID
gen_api = os.environ.get("GEN_API")          # Gemini API Key
owner_phone = os.environ.get("OWNER_PHONE")  # Owner’s phone number

# Hugging Face
HF_TOKEN = os.environ.get("HF_TOKEN")
HF_MODEL = os.environ.get("HF_MODEL", "KhanyiTapiwa00/medsiglip-diagnosis")
HF_ENDPOINT_URL = os.environ.get("HF_ENDPOINT_URL")  # optional dedicated endpoint
HF_TIMEOUT = int(os.environ.get("HF_TIMEOUT", "60"))

hf_client = InferenceClient(model=HF_MODEL, token=HF_TOKEN)

# Redis
redis_url = os.environ.get("REDIS_URL")
redis_token = os.environ.get("UPSTASH_REDIS_TOKEN")

# -----------------------------------------------------------------------------
# Redis Setup
# -----------------------------------------------------------------------------
def setup_redis_connection():
    if redis_url and redis_token:
        try:
            if 'upstash.io' in redis_url:
                parsed = urllib.parse.urlparse(redis_url)
                host = parsed.hostname
                port = 6379
                redis_client = redis.Redis(
                    host=host,
                    port=port,
                    password=redis_token,
                    ssl=True,
                    ssl_cert_reqs=None,
                    decode_responses=True,
                    socket_connect_timeout=10,
                    socket_timeout=10,
                    retry_on_timeout=True
                )
            else:
                redis_client = redis.from_url(
                    redis_url,
                    password=redis_token,
                    ssl=True,
                    decode_responses=True
                )
            redis_client.ping()
            logging.info("✅ Redis connected")
            return redis_client
        except Exception as e:
            logging.error(f"❌ Redis connection failed: {e}")
    logging.warning("⚠️ Redis disabled")
    return None

redis_client = setup_redis_connection()
user_states = {}

# -----------------------------------------------------------------------------
# Hugging Face Helper
# -----------------------------------------------------------------------------
def _hf_image_classify_bytes(image_bytes: bytes):
    """Call Hugging Face API (dedicated endpoint or serverless)."""
    if HF_ENDPOINT_URL:
        headers = {
            "Authorization": f"Bearer {HF_TOKEN}",
            "Content-Type": "application/octet-stream"
        }
        backoff = [0.5, 1.5, 3]
        for wait in backoff + [0]:
            resp = requests.post(HF_ENDPOINT_URL, headers=headers, data=image_bytes, timeout=HF_TIMEOUT)
            if resp.status_code == 503:
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()
    else:
        return hf_client.image_classification(image=image_bytes, timeout=HF_TIMEOUT)

# -----------------------------------------------------------------------------
# Cervical Cancer Analysis with Hugging Face
# -----------------------------------------------------------------------------
def stage_cervical_cancer(image_path):
    try:
        with open(image_path, "rb") as f:
            image_bytes = f.read()
        preds = _hf_image_classify_bytes(image_bytes)
        logging.info(f"HF raw preds: {preds}")

        if isinstance(preds, list) and preds:
            top = max(preds, key=lambda p: p.get("score", 0.0))
            return {
                "stage": top.get("label", "Unknown"),
                "confidence": float(top.get("score", 0.0)),
                "success": True,
                "response_type": "classification"
            }
        return {"stage": "Error", "confidence": 0, "success": False, "error": str(preds)}
    except Exception as e:
        logging.error(f"HF error: {e}", exc_info=True)
        return {"stage": "Error", "confidence": 0, "success": False, "error": str(e)}

# -----------------------------------------------------------------------------
# WhatsApp Helpers
# -----------------------------------------------------------------------------
def send(answer, sender, phone_id):
    url = f"https://graph.facebook.com/v19.0/{phone_id}/messages"
    headers = {"Authorization": f"Bearer {wa_token}", "Content-Type": "application/json"}
    data = {
        "messaging_product": "whatsapp",
        "to": sender,
        "type": "text",
        "text": {"body": answer}
    }
    try:
        r = requests.post(url, headers=headers, json=data)
        r.raise_for_status()
    except Exception as e:
        logging.error(f"❌ send error: {e}")

def download_whatsapp_media(media_id, file_path):
    try:
        headers = {"Authorization": f"Bearer {wa_token}"}
        meta_url = f"https://graph.facebook.com/v19.0/{media_id}"
        meta_resp = requests.get(meta_url, headers=headers, timeout=20)
        meta_resp.raise_for_status()
        media_url = meta_resp.json().get("url")
        if not media_url: return False
        media_resp = requests.get(media_url, headers=headers, timeout=60)
        media_resp.raise_for_status()
        with open(file_path, "wb") as f:
            f.write(media_resp.content)
        return True
    except Exception as e:
        logging.error(f"❌ media download error: {e}")
        return False

# -----------------------------------------------------------------------------
# Conversation State
# -----------------------------------------------------------------------------
def handle_cervical_image(sender, media_id, phone_id):
    image_path = f"/tmp/{sender}_{int(time.time())}.jpg"
    if not download_whatsapp_media(media_id, image_path):
        send("❌ Could not download image.", sender, phone_id)
        return

    send("📨 Analyzing your image...", sender, phone_id)
    result = stage_cervical_cancer(image_path)
    worker_id = user_states.get(sender, {}).get("worker_id", "Unknown")
    patient_id = user_states.get(sender, {}).get("patient_id", "Unknown")

    if result["success"]:
        send(f"""🔬 MedSigLip Results:

📋 Worker ID: {worker_id}
👤 Patient ID: {patient_id}
🏥 Stage: {result['stage']}
✅ Confidence: {result['confidence']:.1%}

💡 Note: This does not replace a doctor's diagnosis.""", sender, phone_id)
    else:
        send(f"❌ Error: {result['error']}", sender, phone_id)

# -----------------------------------------------------------------------------
# Flask App
# -----------------------------------------------------------------------------
app = Flask(__name__)
genai.configure(api_key=gen_api)

@app.route('/', methods=['GET'])
def home():
    return render_template('connected.html')

@app.route('/webhook', methods=['GET'])
def webhook_verify():
    if request.args.get('hub.verify_token') == 'my_verify_token':
        return request.args.get('hub.challenge')
    return 'Error'

@app.route('/webhook', methods=['POST'])
def webhook_handle():
    try:
        data = request.get_json()
        if data.get("object") == "whatsapp_business_account":
            for entry in data.get("entry", []):
                for change in entry.get("changes", []):
                    if change.get("field") == "messages":
                        for msg in change["value"].get("messages", []):
                            sender = msg["from"]
                            if msg["type"] == "image":
                                handle_cervical_image(sender, msg["image"]["id"], phone_id)
                            elif msg["type"] == "text":
                                send("👋 Hello! I'm Rudo. Please upload an image for analysis.", sender, phone_id)
        return jsonify({"status": "ok"}), 200
    except Exception as e:
        logging.error(f"webhook error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "ok",
        "time": datetime.now().isoformat(),
        "hf_model": HF_MODEL,
        "hf_token_set": bool(HF_TOKEN),
        "redis_connected": redis_client is not None
    })

@app.route('/test-hf', methods=['GET'])
def test_hf():
    try:
        pixel = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGMAAQAABQABDQottAAAAABJRU5ErkJggg=="
        )
        preds = _hf_image_classify_bytes(pixel)
        return jsonify({"preds": preds})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -----------------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
