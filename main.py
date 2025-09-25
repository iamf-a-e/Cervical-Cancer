import google.generativeai as genai
from flask import Flask, request, jsonify, render_template
import requests
import os
import sched
import time
import logging
from datetime import datetime, timedelta
from urlextract import URLExtract
from sqlalchemy import create_engine, Column, Integer, String, DateTime, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from google.api_core.exceptions import ResourceExhausted
import redis
import json
import re
import base64
from google.auth import default
from google.auth.transport.requests import Request
import urllib.parse
import threading
import traceback

logging.basicConfig(level=logging.INFO)

# --------------------------------------------------------------------------------
#  Service Account Authentication (GCP)
# --------------------------------------------------------------------------------
service_account_b64 = os.environ.get("GCP_SERVICE_ACCOUNT_BASE64")
if service_account_b64:
    sa_path = "/tmp/service-account.json"
    try:
        with open(sa_path, "wb") as f:
            f.write(base64.b64decode(service_account_b64))
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = sa_path
        logging.info(f" Service account JSON written to {sa_path}")
    except Exception as e:
        logging.error(f" Failed to decode service account JSON: {e}")
else:
    logging.warning(" GCP_SERVICE_ACCOUNT_BASE64 not set.")

# --------------------------------------------------------------------------------
#  Redis Connection (Upstash or normal Redis)
# --------------------------------------------------------------------------------
redis_url = os.environ.get("REDIS_URL")
redis_token = os.environ.get("UPSTASH_REDIS_TOKEN")

def setup_redis_connection():
    if redis_url and redis_token:
        try:
            if 'upstash.io' in redis_url:
                parsed = urllib.parse.urlparse(redis_url)
                host = parsed.hostname
                redis_client = redis.Redis(
                    host=host,
                    port=6379,
                    password=redis_token,
                    ssl=True,
                    ssl_cert_reqs=None,
                    decode_responses=True
                )
            else:
                redis_client = redis.from_url(
                    redis_url,
                    password=redis_token,
                    ssl=True,
                    decode_responses=True
                )
            redis_client.ping()
            logging.info(" Redis connected")
            return redis_client
        except Exception as e:
            logging.error(f" Redis connection failed: {e}")
    return None

redis_client = setup_redis_connection()
user_states = {}

# --------------------------------------------------------------------------------
#  Environment Variables
# --------------------------------------------------------------------------------
wa_token = os.environ.get("WA_TOKEN")  # WhatsApp API Key
phone_id = os.environ.get("PHONE_ID")
gen_api = os.environ.get("GEN_API")
owner_phone = os.environ.get("OWNER_PHONE")
model_name = "gemini-2.0-flash"
bot_name = "Rudo"

VERTEX_AI_ENDPOINT_ID = os.environ.get("VERTEX_AI_ENDPOINT_ID")
VERTEX_AI_REGION = "us-west4"
VERTEX_AI_PROJECT = os.environ.get("VERTEX_AI_PROJECT")

# --------------------------------------------------------------------------------
#  Vertex AI Client
# --------------------------------------------------------------------------------
class VertexAIClient:
    def __init__(self, project_id, endpoint_id, location="us-west4"):
        self.project_id = project_id
        self.endpoint_id = endpoint_id
        self.location = location
        self.base_url = (
            f"https://{location}-aiplatform.googleapis.com/v1/projects/"
            f"{project_id}/locations/{location}/endpoints/{endpoint_id}:predict"
        )
        self.credentials, _ = default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
        if not self.credentials.valid:
            self.credentials.refresh(Request())

    def get_auth_header(self):
        if not self.credentials.valid:
            self.credentials.refresh(Request())
        return {"Authorization": f"Bearer {self.credentials.token}"}

    def predict(self, payload):
        headers = self.get_auth_header()
        headers["Content-Type"] = "application/json"
        response = requests.post(self.base_url, json=payload, headers=headers, timeout=60)
        response.raise_for_status()
        return response.json()

vertex_ai_client = None
if VERTEX_AI_PROJECT and VERTEX_AI_ENDPOINT_ID:
    try:
        vertex_ai_client = VertexAIClient(VERTEX_AI_PROJECT, VERTEX_AI_ENDPOINT_ID, VERTEX_AI_REGION)
        logging.info(" Vertex AI client initialized")
    except Exception as e:
        logging.error(f"❌ Vertex AI init failed: {e}")

# --------------------------------------------------------------------------------
#  Cervical Cancer Staging Function (Fixed)
# --------------------------------------------------------------------------------
def stage_cervical_cancer(image_path):
    if not vertex_ai_client:
        return {"stage": "Error", "confidence": 0, "success": False, "error": "Vertex AI not configured"}
    try:
        with open(image_path, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode("utf-8")
        payload = {"instances": [{"content": image_b64}]}
        result = vertex_ai_client.predict(payload)
        if "predictions" in result and result["predictions"]:
            prediction = result["predictions"][0]
            if "displayNames" in prediction and "confidences" in prediction:
                labels = prediction["displayNames"]
                scores = prediction["confidences"]
                max_idx = scores.index(max(scores))
                return {
                    "stage": labels[max_idx],
                    "confidence": float(scores[max_idx]),
                    "success": True,
                    "response_type": "classification"
                }
        return {"stage": "Error", "confidence": 0, "success": False, "error": "Unexpected response"}
    except Exception as e:
        logging.error(f"❌ Error staging: {e}\n{traceback.format_exc()}")
        return {"stage": "Error", "confidence": 0, "success": False, "error": str(e)}

# --------------------------------------------------------------------------------
#  WhatsApp Helper Functions
# --------------------------------------------------------------------------------
def send(answer, sender, phone_id):
    url = f"https://graph.facebook.com/v19.0/{phone_id}/messages"
    headers = {'Authorization': f'Bearer {wa_token}', 'Content-Type': 'application/json'}
    data = {"messaging_product": "whatsapp", "to": sender, "type": "text", "text": {"body": answer}}
    try:
        requests.post(url, headers=headers, json=data)
    except Exception as e:
        logging.error(f" WhatsApp send error: {e}")

def download_whatsapp_media(media_id, file_path):
    try:
        headers = {'Authorization': f'Bearer {wa_token}'}
        meta_url = f"https://graph.facebook.com/v19.0/{media_id}"
        meta_resp = requests.get(meta_url, headers=headers)
        media_url = meta_resp.json().get("url")
        media_resp = requests.get(media_url, headers=headers)
        with open(file_path, 'wb') as f:
            f.write(media_resp.content)
        return True
    except Exception as e:
        logging.error(f" Download failed: {e}")
        return False

# --------------------------------------------------------------------------------
#  State Handlers (Worker ID → Patient ID → Image → Diagnosis → Follow-up)
# --------------------------------------------------------------------------------
def handle_worker_id(sender, prompt):
    user_states[sender]["worker_id"] = prompt
    user_states[sender]["step"] = "patient_id"
    send("Thank you! What is the Patient ID?", sender, phone_id)

def handle_patient_id(sender, prompt):
    user_states[sender]["patient_id"] = prompt
    user_states[sender]["step"] = "awaiting_image"
    send("Thank you! Now you can upload the image for cervical cancer analysis.", sender, phone_id)

def handle_cervical_image(sender, media_id):
    image_path = f"/tmp/{sender}_{int(time.time())}.jpg"
    if download_whatsapp_media(media_id, image_path):
        send("📨 Analyzing your image...", sender, phone_id)
        result = stage_cervical_cancer(image_path)
        worker_id = user_states[sender].get("worker_id", "Unknown")
        patient_id = user_states[sender].get("patient_id", "Unknown")
        if result["success"]:
            response = f""" MedSigLip Analysis:
📋 Worker ID: {worker_id}
 Patient ID: {patient_id}
 Stage: {result['stage']}
 Confidence: {result['confidence']:.1%}
 Note: This does not replace a doctor’s diagnosis."""
        else:
            response = f"""❌ Analysis failed:
Error: {result['error']}"""
        send(response, sender, phone_id)
    user_states[sender]["step"] = "follow_up"
    send("Would you like to submit another image? (Yes/No)", sender, phone_id)

def handle_follow_up(sender, prompt):
    if prompt.lower() in ["yes", "yebo", "ehe"]:
        user_states[sender]["step"] = "awaiting_image"
        send("Please upload another image.", sender, phone_id)
    else:
        user_states[sender]["step"] = "main_menu"
        send("Thank you for using Dawa Health with MedSigLip technology.", sender, phone_id)

# --------------------------------------------------------------------------------
# Conversation Router
# --------------------------------------------------------------------------------
def message_handler(message, phone_id):
    sender = message["from"]
    if sender not in user_states:
        user_states[sender] = {"step": "worker_id"}
        send("Hello! Let's start with registration. What is your Worker ID?", sender, phone_id)
        return
    step = user_states[sender]["step"]
    if message["type"] == "text":
        prompt = message["text"]["body"]
        if step == "worker_id":
            handle_worker_id(sender, prompt)
        elif step == "patient_id":
            handle_patient_id(sender, prompt)
        elif step == "follow_up":
            handle_follow_up(sender, prompt)
    elif message["type"] == "image" and step == "awaiting_image":
        handle_cervical_image(sender, message["image"]["id"])

# --------------------------------------------------------------------------------
# Flask App
# --------------------------------------------------------------------------------
app = Flask(__name__)
genai.configure(api_key=gen_api)

@app.route('/', methods=['GET'])
def home():
    return "Cervical Cancer WhatsApp Bot is running!"

@app.route('/webhook', methods=['GET'])
def webhook():
    if request.args.get('hub.verify_token') == 'my_verify_token':
        return request.args.get('hub.challenge')
    return 'Error, wrong validation token'

@app.route('/webhook', methods=['POST'])
def webhook_handle():
    data = request.get_json()
    if data.get("object") == "whatsapp_business_account":
        for entry in data.get("entry", []):
            for change in entry.get("changes", []):
                if change.get("field") == "messages":
                    value = change.get("value", {})
                    if "messages" in value:
                        for message in value["messages"]:
                            message_handler(message, phone_id)
    return jsonify({"status": "ok"}), 200

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "vertex_ai": vertex_ai_client is not None,
        "whatsapp": wa_token is not None and phone_id is not None,
        "user_states": len(user_states)
    })

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
