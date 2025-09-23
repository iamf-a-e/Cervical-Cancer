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
from training import products, instructions, cervical_cancer_data
import redis
import json
import re
import base64
from google.oauth2 import service_account

logging.basicConfig(level=logging.INFO)

# Initialize Redis connection for Upstash
redis_url = os.environ.get("UPSTASH_REDIS_URL")

if redis_url:
    try:
        redis_client = redis.from_url(redis_url, decode_responses=True, ssl=True)
        redis_client.ping()
        logging.info("Successfully connected to Upstash Redis")
    except Exception as e:
        logging.error(f"Failed to connect to Upstash Redis: {e}")
        redis_client = None
else:
    redis_client = None
    logging.warning("UPSTASH_REDIS_URL not set, Redis functionality disabled")

# Global user states dictionary (fallback in-memory store)
user_states = {}

wa_token = os.environ.get("WA_TOKEN")  # Whatsapp API Key
phone_id = os.environ.get("PHONE_ID")
gen_api = os.environ.get("GEN_API")  # Gemini API Key
owner_phone = os.environ.get("OWNER_PHONE")  # Owner's phone number with countrycode
model_name = "gemini-2.0-flash"
name = "Fae"
bot_name = "Rudo"
AGENT = "+263719835124"

# Vertex AI REST API configuration
VERTEX_AI_ENDPOINT_ID = "9216603443274186752"
VERTEX_AI_REGION = "us-west4"
VERTEX_AI_PROJECT = os.environ.get("VERTEX_AI_PROJECT")
VERTEX_AI_CREDENTIALS_PATH = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")


class VertexAIClient:
    def __init__(self, project_id, endpoint_id, region="us-west4"):
        self.project_id = project_id
        self.endpoint_id = endpoint_id
        self.region = region
        self.base_url = f"https://{region}-aiplatform.googleapis.com/v1"
        self.credentials = None
        self._setup_credentials()

    def _setup_credentials(self):
        try:
            if VERTEX_AI_CREDENTIALS_PATH and os.path.exists(VERTEX_AI_CREDENTIALS_PATH):
                self.credentials = service_account.Credentials.from_service_account_file(
                    VERTEX_AI_CREDENTIALS_PATH,
                    scopes=['https://www.googleapis.com/auth/cloud-platform']
                )
                logging.info("Vertex AI credentials loaded successfully")
            else:
                self.credentials = service_account.Credentials.from_service_account_file(
                    os.environ.get('GOOGLE_APPLICATION_CREDENTIALS', ''),
                    scopes=['https://www.googleapis.com/auth/cloud-platform']
                )
                logging.info("Using default Google Cloud credentials")
        except Exception as e:
            logging.error(f"Error setting up credentials: {e}")
            self.credentials = None

    def get_access_token(self):
        if self.credentials:
            try:
                from google.auth.transport.requests import Request
                if not self.credentials.valid:
                    self.credentials.refresh(Request())
                return self.credentials.token
            except Exception as e:
                logging.error(f"Error getting access token: {e}")
        return None

    def predict(self, instances):
        url = f"{self.base_url}/projects/{self.project_id}/locations/{self.region}/endpoints/{self.endpoint_id}:predict"
        token = self.get_access_token()
        if not token:
            return {"error": "Could not authenticate with Vertex AI"}
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        try:
            response = requests.post(url, headers=headers, json={"instances": instances})
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logging.error(f"Vertex AI API request failed: {e}")
            return {"error": str(e)}


# Initialize Vertex AI client
vertex_ai_client = None
if VERTEX_AI_PROJECT and VERTEX_AI_ENDPOINT_ID:
    try:
        vertex_ai_client = VertexAIClient(VERTEX_AI_PROJECT, VERTEX_AI_ENDPOINT_ID, VERTEX_AI_REGION)
        logging.info("Vertex AI client initialized successfully")
    except Exception as e:
        logging.error(f"Failed to initialize Vertex AI client: {e}")
        vertex_ai_client = None
else:
    logging.warning("Vertex AI project or endpoint ID not set, cervical cancer staging disabled")

app = Flask(__name__)
genai.configure(api_key=gen_api)

# --- Redis helpers ---
def save_user_state(sender, state):
    if redis_client:
        try:
            ttl = int(timedelta(days=30).total_seconds())
            result = redis_client.setex(f"user_state:{sender}", ttl, json.dumps(state))
            logging.info(f"Saved state for {sender}: {result}")
        except Exception as e:
            logging.error(f"Error saving user state for {sender} to Redis: {e}")

def get_user_state(sender):
    if redis_client:
        try:
            state_data = redis_client.get(f"user_state:{sender}")
            return json.loads(state_data) if state_data else None
        except Exception as e:
            logging.error(f"Error getting user state for {sender} from Redis: {e}")
    return None

def save_user_conversation(sender, role, message):
    if redis_client:
        try:
            history = redis_client.get(f"conversation:{sender}")
            history = json.loads(history) if history else []
            history.append({
                "role": role,
                "message": message,
                "timestamp": datetime.now().isoformat()
            })
            if len(history) > 100:
                history = history[-100:]
            ttl = int(timedelta(days=30).total_seconds())
            redis_client.setex(f"conversation:{sender}", ttl, json.dumps(history))
        except Exception as e:
            logging.error(f"Error saving conversation: {e}")

# --- WhatsApp send + media handling ---
def send(answer, sender, phone_id):
    url = f"https://graph.facebook.com/v19.0/{phone_id}/messages"
    headers = {
        'Authorization': f'Bearer {wa_token}',
        'Content-Type': 'application/json'
    }
    data = {
        "messaging_product": "whatsapp",
        "to": sender,
        "type": "text",
        "text": {"body": answer}
    }
    response = requests.post(url, headers=headers, json=data)
    save_user_conversation(sender, "bot", answer)
    return response

def get_media_url(media_id):
    try:
        url = f"https://graph.facebook.com/v19.0/{media_id}"
        headers = {"Authorization": f"Bearer {wa_token}"}
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        media_data = response.json()
        return media_data.get("url")
    except Exception as e:
        logging.error(f"Error fetching media URL: {e}")
        return None

def download_image(media_id, file_path):
    try:
        media_url = get_media_url(media_id)
        if not media_url:
            return False
        headers = {"Authorization": f"Bearer {wa_token}"}
        response = requests.get(media_url, headers=headers)
        response.raise_for_status()
        with open(file_path, "wb") as f:
            f.write(response.content)
        return True
    except Exception as e:
        logging.error(f"Error downloading image: {e}")
        return False

# --- Cervical staging ---
def stage_cervical_cancer(image_path):
    if not vertex_ai_client:
        return {"stage": "Error", "confidence": 0, "success": False, "error": "Vertex AI client not configured"}
    try:
        with open(image_path, "rb") as f:
            image_data = f.read()
        instance = {"image_bytes": {"b64": base64.b64encode(image_data).decode()}}
        prediction_result = vertex_ai_client.predict([instance])
        if "error" in prediction_result:
            return {"stage": "Error", "confidence": 0, "success": False, "error": prediction_result["error"]}
        if "predictions" in prediction_result and len(prediction_result["predictions"]) > 0:
            results = prediction_result["predictions"][0]
            stage = results.get('stage', results.get('class', 'Unknown'))
            confidence = results.get('confidence', results.get('score', 0))
            return {"stage": stage, "confidence": float(confidence), "success": True}
        else:
            return {"stage": "Error", "confidence": 0, "success": False, "error": "No predictions returned"}
    except Exception as e:
        logging.error(f"Error in cervical cancer staging: {e}")
        return {"stage": "Error", "confidence": 0, "success": False, "error": str(e)}

# --- Conversation handler (only showing updated cervical image + main handler) ---
def handle_cervical_image(sender, media_id, phone_id):
    state = user_states[sender]
    lang = state["language"]
    image_path = f"/tmp/{sender}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    send("I've received your image. Please wait while I analyze it.", sender, phone_id)
    if download_image(media_id, image_path):
        result = stage_cervical_cancer(image_path)
        if result["success"]:
            response = f"Diagnosis results:\n- Worker ID: {state.get('worker_id')}\n- Patient ID: {state.get('patient_id')}\n- Stage: {result['stage']}\n- Confidence: {result['confidence']:.2%}\n\nNote: This does not replace a doctor's diagnosis."
        else:
            response = "I'm sorry, I couldn't analyze your image. Please try sending another image or consult a doctor directly."
        os.remove(image_path)
        send(response, sender, phone_id)
    else:
        send("I'm sorry, I couldn't download your image. Please try again.", sender, phone_id)
    state["step"] = "follow_up"
    send("Would you like to submit another image? (Reply 'Yes' or 'No')", sender, phone_id)
    save_user_state(sender, state)

def handle_conversation_state(sender, prompt, phone_id, media_id=None, media_type=None):
    state = user_states[sender]
    if media_type == "image" and state["step"] == "awaiting_image":
        handle_cervical_image(sender, media_id, phone_id)
        return
    if state["step"] == "language_detection":
        handle_language_detection(sender, prompt, phone_id)
    elif state["step"] == "worker_id":
        handle_worker_id(sender, prompt, phone_id)
    elif state["step"] == "patient_id":
        handle_patient_id(sender, prompt, phone_id)
    elif state["step"] == "follow_up":
        handle_follow_up(sender, prompt, phone_id)
    else:
        state["step"] = "language_detection"
        handle_language_detection(sender, prompt, phone_id)
    save_user_state(sender, state)

def message_handler(data, phone_id):
    global user_states
    sender = data["from"]
    state = user_states.get(sender)
    if not state:
        redis_state = get_user_state(sender)
        if redis_state:
            state = redis_state
            user_states[sender] = redis_state
    if not state:
        state = {"step": "language_detection", "language": "english", "needs_language_confirmation": False,
                 "registered": False, "worker_id": None, "patient_id": None, "conversation_history": []}
        user_states[sender] = state
        save_user_state(sender, state)
    prompt = ""
    media_id = None
    media_type = None
    if data["type"] == "text":
        prompt = data["text"]["body"]
    elif data["type"] == "image":
        media_type = "image"
        media_id = data["image"]["id"]
        prompt = "[Image received]"
    save_user_conversation(sender, "user", prompt if prompt else "[Media message]")
    handle_conversation_state(sender, prompt, phone_id, media_id, media_type)

# --- Flask routes ---
@app.route("/", methods=["GET", "POST"])
def index():
    return render_template("connected.html")

@app.route("/webhook", methods=["GET", "POST"])
def webhook():
    if request.method == "GET":
        mode = request.args.get("hub.mode")
        token = request.args.get("hub.verify_token")
        challenge = request.args.get("hub.challenge")
        if mode == "subscribe" and token == "BOT":
            return challenge, 200
        else:
            return "Failed", 403
    elif request.method == "POST":
        try:
            data = request.get_json()
            entry = data["entry"][0]
            changes = entry["changes"][0]
            value = changes["value"]
            if "messages" in value:
                message_data = value["messages"][0]
                phone_id = value["metadata"]["phone_number_id"]
                message_handler(message_data, phone_id)
        except Exception as e:
            logging.error(f"Error in webhook: {e}")
        return jsonify({"status": "ok"}), 200

if __name__ == "__main__":
    app.run(debug=True, port=8000)
