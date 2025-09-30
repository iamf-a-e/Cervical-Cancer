import os, re, json, time, base64, logging, threading
from datetime import datetime, timedelta
from flask import Flask, request, jsonify, render_template
import requests, redis, urllib.parse, sched, fitz
from sqlalchemy import create_engine, Column, Integer, String, DateTime, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from google.api_core.exceptions import ResourceExhausted
from huggingface_hub import InferenceClient
import google.generativeai as genai

logging.basicConfig(level=logging.INFO)

# --------------------------------------------------------------------------------
# ✅ Hugging Face setup
# --------------------------------------------------------------------------------
HF_TOKEN = os.environ.get("HF_TOKEN")  # Put in Vercel env
HF_MODEL = os.environ.get("HF_MODEL", "KhanyiTapiwa00/medsiglip-diagnosis")
hf_client = InferenceClient(model=HF_MODEL, token=HF_TOKEN)

# --------------------------------------------------------------------------------
# ✅ Redis connection
# --------------------------------------------------------------------------------
redis_url = os.environ.get("REDIS_URL")
redis_token = os.environ.get("UPSTASH_REDIS_TOKEN")

def setup_redis_connection():
    if redis_url and redis_token:
        try:
            redis_client = redis.from_url(
                redis_url,
                password=redis_token,
                ssl=True,
                decode_responses=True
            )
            redis_client.ping()
            logging.info("✅ Connected to Redis")
            return redis_client
        except Exception as e:
            logging.error(f"❌ Redis connection failed: {e}")
    return None

redis_client = setup_redis_connection()
user_states = {}

# --------------------------------------------------------------------------------
# ✅ WhatsApp + Gemini setup
# --------------------------------------------------------------------------------
wa_token = os.environ.get("WA_TOKEN")
phone_id = os.environ.get("PHONE_ID")
gen_api = os.environ.get("GEN_API")

genai.configure(api_key=gen_api)
model_name = "gemini-2.0-flash"
generation_config = {"temperature": 1, "top_p": 0.95, "top_k": 0, "max_output_tokens": 8192}
safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]
gemini_model = genai.GenerativeModel(
    model_name=model_name, generation_config=generation_config, safety_settings=safety_settings
)

# --------------------------------------------------------------------------------
# ✅ Hugging Face inference for cervical cancer staging
# --------------------------------------------------------------------------------
def stage_cervical_cancer(image_path):
    """Run cervical cancer classification with Hugging Face model"""
    try:
        with open(image_path, "rb") as f:
            image_bytes = f.read()

        result = hf_client.image_classification(image=image_bytes)
        logging.info(f"Raw HF response: {result}")

        if result and isinstance(result, list):
            best = max(result, key=lambda x: x["score"])
            return {
                "stage": best["label"],
                "confidence": float(best["score"]),
                "success": True,
                "response_type": "classification",
            }

        return {"stage": "Error", "confidence": 0, "success": False, "error": "No predictions"}
    except Exception as e:
        logging.error(f"❌ HF staging error: {e}")
        return {"stage": "Error", "confidence": 0, "success": False, "error": str(e)}

# --------------------------------------------------------------------------------
# ✅ WhatsApp send function
# --------------------------------------------------------------------------------
def send(answer, sender, phone_id):
    url = f"https://graph.facebook.com/v19.0/{phone_id}/messages"
    headers = {"Authorization": f"Bearer {wa_token}", "Content-Type": "application/json"}
    data = {"messaging_product": "whatsapp", "to": sender, "type": "text", "text": {"body": answer}}

    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        logging.debug(f"📤 Message sent to {sender}")
    except Exception as e:
        logging.error(f"❌ Error sending message: {e}")
        response = None
    return response

# --------------------------------------------------------------------------------
# ✅ Flask app
# --------------------------------------------------------------------------------
app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return render_template("connected.html")

@app.route("/webhook", methods=["GET"])
def webhook():
    if request.args.get("hub.verify_token") == "my_verify_token":
        return request.args.get("hub.challenge")
    return "Error, wrong validation token"

@app.route("/webhook", methods=["POST"])
def webhook_handle():
    try:
        data = request.get_json()
        logging.info(f"📨 Webhook data: {json.dumps(data, indent=2)}")

        if data.get("object") == "whatsapp_business_account":
            for entry in data.get("entry", []):
                for change in entry.get("changes", []):
                    if change.get("field") == "messages":
                        value = change.get("value", {})
                        if "messages" in value:
                            for message in value["messages"]:
                                sender = message["from"]
                                if message["type"] == "image":
                                    media_id = message["image"]["id"]
                                    image_path = f"/tmp/{sender}_{int(time.time())}.jpg"
                                    # (download_whatsapp_media here… then:)
                                    result = stage_cervical_cancer(image_path)
                                    reply = f"Stage: {result['stage']} (Confidence: {result['confidence']:.1%})"
                                    send(reply, sender, phone_id)
                                else:
                                    prompt = message["text"]["body"]
                                    convo = gemini_model.start_chat(history=[])
                                    convo.send_message(prompt)
                                    reply = convo.last.text
                                    send(reply, sender, phone_id)

        return jsonify({"status": "success"}), 200
    except Exception as e:
        logging.error(f"❌ Webhook error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
