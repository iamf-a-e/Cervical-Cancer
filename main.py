import google.generativeai as genai
from flask import Flask, request, jsonify, render_template
import requests
import os
import time
import logging
from datetime import datetime, timedelta
from urlextract import URLExtract
import redis
import json
import re
import base64
import urllib.parse
import threading

logging.basicConfig(level=logging.INFO)

# --------------------------------------------------------------------------------
# ✅ Redis Connection
# --------------------------------------------------------------------------------
redis_url = os.environ.get("REDIS_URL")
redis_token = os.environ.get("UPSTASH_REDIS_TOKEN")

def setup_redis_connection():
    """Setup Redis connection"""
    if redis_url:
        try:
            if redis_token:
                redis_client = redis.from_url(
                    redis_url,
                    password=redis_token,
                    ssl=True,
                    decode_responses=True,
                    socket_connect_timeout=10,
                    socket_timeout=10,
                    retry_on_timeout=True
                )
            else:
                redis_client = redis.from_url(
                    redis_url,
                    decode_responses=True,
                    socket_connect_timeout=10,
                    socket_timeout=10,
                    retry_on_timeout=True
                )
            
            redis_client.ping()
            logging.info("✅ Successfully connected to Redis")
            return redis_client
        except Exception as e:
            logging.error(f"❌ Failed to connect to Redis: {e}")
    
    logging.warning("⚠️ Redis functionality disabled")
    return None

redis_client = setup_redis_connection()

# Global user states dictionary (fallback)
user_states = {}

# Environment variables
wa_token = os.environ.get("WA_TOKEN")
phone_id = os.environ.get("PHONE_ID")
gen_api = os.environ.get("GEN_API")
owner_phone = os.environ.get("OWNER_PHONE")
model_name = "gemini-2.0-flash"
bot_name = "Rudo"

# Hugging Face Configuration
HF_MODEL_NAME = os.environ.get("HF_MODEL_NAME", "google/vit-base-patch16-224")
HF_API_TOKEN = os.environ.get("HF_API_TOKEN")

# --------------------------------------------------------------------------------
# ✅ Hugging Face Client
# --------------------------------------------------------------------------------

class HuggingFaceClient:
    def __init__(self, model_name=HF_MODEL_NAME, api_token=HF_API_TOKEN):
        self.model_name = model_name
        self.api_token = api_token
        logging.info(f"🔗 Using Hugging Face Inference API with model: {model_name}")

    def predict(self, image_path):
        """Predict using Hugging Face Inference API"""
        try:
            if not self.api_token:
                return {
                    "success": False,
                    "error": "Hugging Face API token required"
                }
            
            # Read image file
            with open(image_path, "rb") as f:
                image_data = f.read()
            
            # Hugging Face Inference API endpoint
            api_url = f"https://api-inference.huggingface.co/models/{self.model_name}"
            headers = {
                "Authorization": f"Bearer {self.api_token}"
            }
            
            # Send image directly
            response = requests.post(api_url, headers=headers, data=image_data, timeout=30)
            
            if response.status_code == 503:
                return {
                    "success": False,
                    "error": "Model is loading, please try again in a few seconds"
                }
            elif response.status_code == 429:
                return {
                    "success": False,
                    "error": "Rate limited, please try again later"
                }
            elif response.status_code != 200:
                return {
                    "success": False,
                    "error": f"API error: {response.status_code} - {response.text}"
                }
            
            result = response.json()
            logging.info(f"🤖 Hugging Face raw response: {result}")
            
            # Format the response
            if isinstance(result, list):
                predictions = result
            elif isinstance(result, dict) and "predictions" in result:
                predictions = result["predictions"]
            else:
                predictions = [result]
            
            # Sort by confidence score
            sorted_predictions = sorted(
                predictions, 
                key=lambda x: x.get('score', x.get('confidence', 0)), 
                reverse=True
            )
            
            formatted_results = []
            for pred in sorted_predictions[:3]:
                confidence = pred.get('score', pred.get('confidence', 0))
                label = pred.get('label', 'Unknown')
                
                formatted_results.append({
                    "label": label,
                    "confidence": confidence
                })
            
            return {
                "success": True,
                "predictions": formatted_results,
                "top_prediction": formatted_results[0] if formatted_results else None,
                "response_type": "classification"
            }
            
        except Exception as e:
            logging.error(f"❌ Hugging Face prediction error: {e}")
            return {
                "success": False,
                "error": str(e)
            }

# Initialize Hugging Face client
hf_client = None
if HF_API_TOKEN:
    try:
        hf_client = HuggingFaceClient()
        logging.info("✅ Hugging Face client initialized successfully")
    except Exception as e:
        logging.error(f"❌ Failed to initialize Hugging Face client: {e}")
        hf_client = None
else:
    logging.warning("⚠️ HF_API_TOKEN not set, image analysis disabled")

# --------------------------------------------------------------------------------
# ✅ Flask App Setup
# --------------------------------------------------------------------------------

app = Flask(__name__)

# Configure Gemini
if gen_api:
    genai.configure(api_key=gen_api)
    generation_config = {
        "temperature": 1,
        "top_p": 0.95,
        "top_k": 0,
        "max_output_tokens": 8192,
    }
    safety_settings = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    ]
    model = genai.GenerativeModel(
        model_name=model_name,
        generation_config=generation_config,
        safety_settings=safety_settings
    )
    logging.info("✅ Gemini configured successfully")
else:
    logging.warning("⚠️ GEN_API not set, Gemini functionality disabled")
    model = None

# --------------------------------------------------------------------------------
# ✅ User State Management
# --------------------------------------------------------------------------------

def save_user_state(sender, state):
    """Save user state to Redis"""
    if redis_client:
        try:
            redis_client.setex(f"user_state:{sender}", timedelta(days=1), json.dumps(state))
        except Exception as e:
            logging.error(f"❌ Error saving user state: {e}")
            user_states[sender] = state
    else:
        user_states[sender] = state

def get_user_state(sender):
    """Get user state from Redis"""
    if redis_client:
        try:
            state_data = redis_client.get(f"user_state:{sender}")
            if state_data:
                return json.loads(state_data)
        except Exception as e:
            logging.error(f"❌ Error getting user state: {e}")
    
    return user_states.get(sender)

def init_user_state(sender):
    """Initialize new user state"""
    state = {
        "step": "language_detection",
        "language": "english",
        "registered": False,
        "worker_id": None,
        "patient_id": None,
        "conversation_history": [],
        "created_at": datetime.now().isoformat()
    }
    save_user_state(sender, state)
    return state

# --------------------------------------------------------------------------------
# ✅ Core Functions
# --------------------------------------------------------------------------------

def send_message(sender, message, phone_id):
    """Send message via WhatsApp API"""
    if not wa_token or not phone_id:
        logging.error("❌ WhatsApp credentials not configured")
        return False

    url = f"https://graph.facebook.com/v19.0/{phone_id}/messages"
    headers = {
        'Authorization': f'Bearer {wa_token}',
        'Content-Type': 'application/json'
    }
    
    data = {
        "messaging_product": "whatsapp",
        "to": sender,
        "type": "text",
        "text": {
            "body": message
        }
    }

    try:
        response = requests.post(url, headers=headers, json=data, timeout=10)
        if response.status_code == 200:
            logging.info(f"📤 Message sent to {sender}")
            return True
        else:
            logging.error(f"❌ Failed to send message: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        logging.error(f"❌ Error sending message: {e}")
        return False

def download_whatsapp_media(media_id, file_path):
    """Download WhatsApp media"""
    if not wa_token:
        logging.error("❌ WA_TOKEN not configured")
        return False

    try:
        headers = {'Authorization': f'Bearer {wa_token}'}
        
        # Get media URL
        meta_url = f"https://graph.facebook.com/v19.0/{media_id}"
        meta_resp = requests.get(meta_url, headers=headers, timeout=20)
        meta_resp.raise_for_status()
        media_data = meta_resp.json()
        
        media_url = media_data.get("url")
        if not media_url:
            logging.error("❌ No media URL found")
            return False

        # Download media
        media_resp = requests.get(media_url, headers=headers, timeout=30)
        media_resp.raise_for_status()

        with open(file_path, 'wb') as f:
            f.write(media_resp.content)

        logging.info(f"✅ Media downloaded: {file_path}")
        return True
        
    except Exception as e:
        logging.error(f"❌ Error downloading media: {e}")
        return False

def detect_language(message):
    """Detect language from message"""
    message_lower = message.lower()
    
    language_keywords = {
        "shona": ["mhoro", "mhoroi", "makadini", "hesi"],
        "ndebele": ["sawubona", "unjani", "salibonani", "yebo"],
        "english": ["hi", "hello", "hey", "good morning"]
    }
    
    for lang, keywords in language_keywords.items():
        if any(keyword in message_lower for keyword in keywords):
            return lang
    return "english"

def analyze_image(image_path):
    """Analyze image using Hugging Face"""
    if not hf_client:
        return {
            "success": False,
            "error": "Image analysis not available"
        }
    
    return hf_client.predict(image_path)

# --------------------------------------------------------------------------------
# ✅ Conversation Handlers
# --------------------------------------------------------------------------------

def handle_language_detection(sender, prompt, phone_id):
    """Handle language detection"""
    language = detect_language(prompt)
    state = get_user_state(sender)
    state["language"] = language
    state["step"] = "worker_id"
    
    if language == "shona":
        send_message(sender, "Mhoro! Ndini Rudo, mubatsiri weDawa Health. Ndapota ipai Worker ID yenyu.", phone_id)
    elif language == "ndebele":
        send_message(sender, "Sawubona! Ngingu Rudo, umsizi weDawa Health. Sicela unikeze i-Worker ID yakho.", phone_id)
    else:
        send_message(sender, "Hello! I'm Rudo, Dawa Health's assistant. Please provide your Worker ID.", phone_id)
    
    save_user_state(sender, state)

def handle_worker_id(sender, prompt, phone_id):
    """Handle worker ID input"""
    state = get_user_state(sender)
    state["worker_id"] = prompt.strip()
    state["step"] = "patient_id"
    
    lang = state["language"]
    if lang == "shona":
        send_message(sender, "Ndatenda! Ndapota ipai Patient ID.", phone_id)
    elif lang == "ndebele":
        send_message(sender, "Ngiyabonga! Sicela unikeze i-Patient ID.", phone_id)
    else:
        send_message(sender, "Thank you! Please provide the Patient ID.", phone_id)
    
    save_user_state(sender, state)

def handle_patient_id(sender, prompt, phone_id):
    """Handle patient ID input"""
    state = get_user_state(sender)
    state["patient_id"] = prompt.strip()
    state["step"] = "awaiting_image"
    state["registered"] = True
    
    lang = state["language"]
    if lang == "shona":
        send_message(sender, "Ndatenda! Zvino tumirai mufananidzo wekuongororwa.", phone_id)
    elif lang == "ndebele":
        send_message(sender, "Ngiyabonga! Manje ngicela uthumele isithombe sokuhlola.", phone_id)
    else:
        send_message(sender, "Thank you! Now please send the medical image for analysis.", phone_id)
    
    save_user_state(sender, state)

def handle_image_analysis(sender, media_id, phone_id):
    """Handle image analysis"""
    state = get_user_state(sender)
    lang = state["language"]
    
    # Send processing message
    if lang == "shona":
        send_message(sender, "📨 Ndiri kuongorora mufananidzo wenyu...", phone_id)
    else:
        send_message(sender, "📨 Analyzing your image...", phone_id)
    
    # Download and analyze image
    image_path = f"/tmp/{sender}_{int(time.time())}.jpg"
    
    if download_whatsapp_media(media_id, image_path):
        try:
            result = analyze_image(image_path)
            
            worker_id = state.get("worker_id", "Unknown")
            patient_id = state.get("patient_id", "Unknown")
            
            if result["success"]:
                top_pred = result["top_prediction"]
                stage = top_pred["label"]
                confidence = top_pred["confidence"]
                
                if lang == "shona":
                    response = f"""🔬 Ongororo Yakaitwa:

📋 Worker ID: {worker_id}
👤 Patient ID: {patient_id}
🏥 Mhedzisiro: {stage}
✅ Chivimbo: {confidence:.1%}

💡 Ziva: Izvi hazvitsivi kuongororwa kwechiremba."""
                else:
                    response = f"""🔬 Analysis Results:

📋 Worker ID: {worker_id}
👤 Patient ID: {patient_id}
🏥 Findings: {stage}
✅ Confidence: {confidence:.1%}

💡 Note: This does not replace a doctor's diagnosis."""
            else:
                error_msg = result.get("error", "Analysis failed")
                if lang == "shona":
                    response = f"❌ Hatina kukwanisa kuongorora mufananidzo: {error_msg}"
                else:
                    response = f"❌ Could not analyze image: {error_msg}"
            
            send_message(sender, response, phone_id)
            
            # Clean up
            try:
                os.remove(image_path)
            except:
                pass
                
        except Exception as e:
            logging.error(f"❌ Image analysis error: {e}")
            if lang == "shona":
                send_message(sender, "❌ Pakaine dambudziko pakuongorora mufananidzo. Edza zvakare.", phone_id)
            else:
                send_message(sender, "❌ Error analyzing image. Please try again.", phone_id)
    else:
        if lang == "shona":
            send_message(sender, "❌ Hatina kukwanisa kudhawunirodha mufananidzo. Edza zvakare.", phone_id)
        else:
            send_message(sender, "❌ Could not download image. Please try again.", phone_id)
    
    # Move to follow-up
    state["step"] = "follow_up"
    if lang == "shona":
        send_message(sender, "Unoda kuendesa imwe mufananidzo here? (Ehe/Aihwa)", phone_id)
    else:
        send_message(sender, "Would you like to submit another image? (Yes/No)", phone_id)
    
    save_user_state(sender, state)

def handle_follow_up(sender, prompt, phone_id):
    """Handle follow-up response"""
    state = get_user_state(sender)
    lang = state["language"]
    prompt_lower = prompt.lower()
    
    if any(word in prompt_lower for word in ["yes", "ehe", "yebo", "hongu"]):
        state["step"] = "awaiting_image"
        if lang == "shona":
            send_message(sender, "Tumirai imwe mufananidzo.", phone_id)
        else:
            send_message(sender, "Please send another image.", phone_id)
    else:
        state["step"] = "main_menu"
        if lang == "shona":
            send_message(sender, "Ndatenda! Kana uine mimwe mibvunzo, ndapota buda.", phone_id)
        else:
            send_message(sender, "Thank you! If you have more questions, please ask.", phone_id)
    
    save_user_state(sender, state)

def handle_main_menu(sender, prompt, phone_id):
    """Handle general conversation"""
    state = get_user_state(sender)
    lang = state["language"]
    
    if not model:
        if lang == "shona":
            send_message(sender, "Ndine urombo, handigoni kupindura mibvunzo yazvino. Ndiri kugadzirirwa chete kuongorora mifananidzo.", phone_id)
        else:
            send_message(sender, "I'm sorry, I can't answer questions right now. I'm only configured for image analysis.", phone_id)
        return
    
    try:
        # Use Gemini for general conversation
        convo = model.start_chat(history=[])
        response = convo.send_message(prompt)
        
        if response and response.text:
            send_message(sender, response.text, phone_id)
        else:
            if lang == "shona":
                send_message(sender, "Ndine urombo, handina kupindura. Edza zvakare.", phone_id)
            else:
                send_message(sender, "Sorry, I didn't get a response. Please try again.", phone_id)
                
    except Exception as e:
        logging.error(f"❌ Gemini error: {e}")
        if lang == "shona":
            send_message(sender, "Ndine urombo, pane dambudziko. Edza zvakare gare gare.", phone_id)
        else:
            send_message(sender, "Sorry, there was an error. Please try again later.", phone_id)

def process_message(sender, message_data, phone_id):
    """Process incoming message"""
    logging.info(f"📩 Processing message from {sender}")
    
    # Get or initialize user state
    state = get_user_state(sender)
    if not state:
        state = init_user_state(sender)
        logging.info(f"🆕 New user initialized: {sender}")
    
    # Determine message type
    if message_data.get("type") == "text":
        prompt = message_data["text"]["body"]
        logging.info(f"💬 Text message: {prompt}")
        
        # Check for reset keywords
        if prompt.lower() in ["hi", "hello", "hey", "mhoro", "sawubona"]:
            state = init_user_state(sender)
            send_message(sender, "👋 Hello! I'm Rudo from Dawa Health. Let's get started!", phone_id)
            return
        
        # Route based on current step
        if state["step"] == "language_detection":
            handle_language_detection(sender, prompt, phone_id)
        elif state["step"] == "worker_id":
            handle_worker_id(sender, prompt, phone_id)
        elif state["step"] == "patient_id":
            handle_patient_id(sender, prompt, phone_id)
        elif state["step"] == "follow_up":
            handle_follow_up(sender, prompt, phone_id)
        elif state["step"] == "main_menu":
            handle_main_menu(sender, prompt, phone_id)
        else:
            # Default to language detection
            state["step"] = "language_detection"
            save_user_state(sender, state)
            handle_language_detection(sender, prompt, phone_id)
            
    elif message_data.get("type") == "image":
        logging.info("🖼️ Image message received")
        if state["step"] == "awaiting_image":
            media_id = message_data["image"]["id"]
            handle_image_analysis(sender, media_id, phone_id)
        else:
            lang = state.get("language", "english")
            if lang == "shona":
                send_message(sender, "Ndapota tanga wanyora Worker ID nePatient ID usati watumira mufananidzo.", phone_id)
            else:
                send_message(sender, "Please provide Worker ID and Patient ID first before sending images.", phone_id)
    else:
        logging.warning(f"⚠️ Unsupported message type: {message_data.get('type')}")
        send_message(sender, "I only support text and image messages at the moment.", phone_id)

# --------------------------------------------------------------------------------
# ✅ Flask Routes
# --------------------------------------------------------------------------------

@app.route('/webhook', methods=['GET'])
def verify_webhook():
    """Verify webhook for WhatsApp"""
    mode = request.args.get('hub.mode')
    token = request.args.get('hub.verify_token')
    challenge = request.args.get('hub.challenge')
    
    if mode == 'subscribe' and token == 'hello':
        logging.info("✅ Webhook verified successfully")
        return challenge
    else:
        logging.error("❌ Webhook verification failed")
        return 'Verification failed', 403

@app.route('/webhook', methods=['POST'])
def handle_webhook():
    """Handle incoming WhatsApp messages"""
    try:
        data = request.get_json()
        logging.info(f"📨 Webhook received: {json.dumps(data, indent=2)[:500]}...")
        
        if data.get("object") == "whatsapp_business_account":
            for entry in data.get("entry", []):
                for change in entry.get("changes", []):
                    if change.get("field") == "messages":
                        value = change.get("value", {})
                        if "messages" in value:
                            for message in value["messages"]:
                                # Process in background thread
                                threading.Thread(
                                    target=process_message,
                                    args=(message["from"], message, phone_id)
                                ).start()
        
        return jsonify({"status": "success"}), 200
        
    except Exception as e:
        logging.error(f"❌ Webhook error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/')
def home():
    return jsonify({
        "status": "running",
        "service": "Dawa Health WhatsApp Bot",
        "timestamp": datetime.now().isoformat()
    })

@app.route('/health')
def health_check():
    return jsonify({
        "status": "healthy",
        "redis": "connected" if redis_client else "disconnected",
        "huggingface": "configured" if hf_client else "not configured",
        "gemini": "configured" if model else "not configured",
        "whatsapp": "configured" if wa_token and phone_id else "not configured"
    })

@app.route('/debug/user/<sender>')
def debug_user(sender):
    state = get_user_state(sender)
    return jsonify({"user_state": state})

# --------------------------------------------------------------------------------
# ✅ Startup
# --------------------------------------------------------------------------------

logging.info("🚀 Dawa Health WhatsApp Bot starting...")
logging.info(f"📞 Phone ID: {phone_id}")
logging.info(f"🤖 Hugging Face: {'configured' if hf_client else 'not configured'}")
logging.info(f"🧠 Gemini: {'configured' if model else 'not configured'}")

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
