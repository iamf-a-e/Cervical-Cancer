import google.generativeai as genai
from flask import Flask, request, jsonify, render_template
import requests
import os
import fitz
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
from google.auth import default
import google.auth
from google.auth.transport.requests import Request
import urllib.parse
import threading

logging.basicConfig(level=logging.INFO)

# --------------------------------------------------------------------------------
# ✅ Improved Redis Connection for Upstash
# --------------------------------------------------------------------------------
redis_url = os.environ.get("REDIS_URL")
redis_token = os.environ.get("UPSTASH_REDIS_TOKEN")

def setup_redis_connection():
    """Setup Redis connection with Upstash compatibility"""
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
                    decode_responses=True,
                    socket_connect_timeout=10,
                    socket_timeout=10,
                    retry_on_timeout=True
                )
            
            redis_client.ping()
            logging.info("✅ Successfully connected to Upstash Redis")
            return redis_client
            
        except Exception as e:
            logging.error(f"❌ Failed to connect to Upstash Redis: {e}")
    logging.warning("⚠️ Redis functionality disabled")
    return None

redis_client = setup_redis_connection()

# Global user states dictionary (fallback)
user_states = {}

wa_token = os.environ.get("WA_TOKEN")
phone_id = os.environ.get("PHONE_ID")
gen_api = os.environ.get("GEN_API")
owner_phone = os.environ.get("OWNER_PHONE")
model_name = "gemini-2.0-flash"
name = "Fae"
bot_name = "Rudo"
AGENT = "+263719835124"

# Hugging Face Configuration
HF_MODEL_NAME = os.environ.get("HF_MODEL_NAME", "google/vit-base-patch16-224")
HF_API_TOKEN = os.environ.get("HF_API_TOKEN")
# Use Hugging Face Inference API (simpler than local models)
HF_USE_INFERENCE_API = os.environ.get("HF_USE_INFERENCE_API", "true").lower() == "true"

# --------------------------------------------------------------------------------
# ✅ Simplified Hugging Face Client (No PIL/torch dependencies)
# --------------------------------------------------------------------------------

class HuggingFaceClient:
    def __init__(self, model_name=HF_MODEL_NAME, use_inference_api=HF_USE_INFERENCE_API, api_token=HF_API_TOKEN):
        self.model_name = model_name
        self.use_inference_api = use_inference_api
        self.api_token = api_token
        
        if self.use_inference_api:
            logging.info(f"🔗 Using Hugging Face Inference API with model: {model_name}")
        else:
            logging.warning("⚠️ Local model loading disabled, using Inference API only")

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
            
            # Send image directly (no base64 encoding needed for Inference API)
            response = requests.post(api_url, headers=headers, data=image_data, timeout=60)
            
            if response.status_code == 503:
                # Model is loading
                return {
                    "success": False,
                    "error": "Model is loading, please try again in a few seconds"
                }
            elif response.status_code == 429:
                # Rate limited
                return {
                    "success": False,
                    "error": "Rate limited, please try again later"
                }
            
            response.raise_for_status()
            result = response.json()
            
            # Format the response
            if isinstance(result, list):
                predictions = result
            elif isinstance(result, dict) and "predictions" in result:
                predictions = result["predictions"]
            else:
                predictions = [result]
            
            # Sort by confidence score (most models return score)
            sorted_predictions = sorted(
                predictions, 
                key=lambda x: x.get('score', x.get('confidence', 0)), 
                reverse=True
            )
            
            formatted_results = []
            for pred in sorted_predictions[:3]:  # Top 3 predictions
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
            
        except requests.exceptions.RequestException as e:
            logging.error(f"❌ Hugging Face API request failed: {e}")
            return {
                "success": False,
                "error": f"API request failed: {str(e)}"
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
    logging.warning("⚠️ HF_API_TOKEN not set, Hugging Face image analysis disabled")

# --------------------------------------------------------------------------------
# ✅ Environment Validation
# --------------------------------------------------------------------------------

def validate_environment():
    """Validate required environment variables"""
    required_vars = {
        "WA_TOKEN": wa_token,
        "PHONE_ID": phone_id,
        "GEN_API": gen_api,
    }
    
    missing_vars = [var for var, value in required_vars.items() if not value]
    if missing_vars:
        logging.warning(f"⚠️ Missing environment variables: {missing_vars}")
    else:
        logging.info("✅ All required environment variables are set")
    
    optional_vars = {
        "REDIS_URL": redis_url,
        "HF_API_TOKEN": HF_API_TOKEN,
        "OWNER_PHONE": owner_phone,
    }
    
    for var, value in optional_vars.items():
        if not value:
            logging.info(f"ℹ️ Optional variable not set: {var}")

validate_environment()

app = Flask(__name__)
genai.configure(api_key=gen_api)

class CustomURLExtract(URLExtract):
    def _get_cache_file_path(self):
        cache_dir = "/tmp"
        return os.path.join(cache_dir, "tlds-alpha-by-domain.txt")

extractor = CustomURLExtract(limit=1)

try:
    extractor.update()
    logging.info("✅ TLD cache updated successfully")
except Exception as e:
    logging.warning(f"⚠️ Could not update TLD cache: {e}")

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

model = genai.GenerativeModel(model_name=model_name,
                              generation_config=generation_config,
                              safety_settings=safety_settings)

convo = model.start_chat(history=[])

def save_user_states():
    """Save all user states to Redis"""
    if redis_client:
        try:
            for sender, state in user_states.items():
                redis_client.setex(f"user_state:{sender}", timedelta(days=30), json.dumps(state))
            logging.info(f"✅ User states saved to Redis: {len(user_states)} states")
        except Exception as e:
            logging.error(f"❌ Error saving user states to Redis: {e}")

def load_user_states():
    """Load user states from Redis"""
    global user_states
    if redis_client:
        try:
            keys = redis_client.keys("user_state:*")
            user_states = {}
            for key in keys:
                sender = key.replace("user_state:", "")
                state_data = redis_client.get(key)
                if state_data:
                    try:
                        state = json.loads(state_data)
                        if isinstance(state, dict) and "step" in state:
                            user_states[sender] = state
                    except json.JSONDecodeError as e:
                        logging.error(f"❌ Error decoding JSON for {sender}: {e}")
            logging.info(f"✅ Loaded {len(user_states)} user states from Redis")
        except Exception as e:
            logging.error(f"❌ Error loading user states from Redis: {e}")
            user_states = {}
    else:
        user_states = {}
        logging.warning("⚠️ Redis client not available, using in-memory storage only")

def get_user_conversation(sender):
    """Get user conversation history from Redis"""
    if redis_client:
        try:
            history = redis_client.get(f"conversation:{sender}")
            return json.loads(history) if history else []
        except Exception as e:
            logging.error(f"❌ Error getting conversation from Redis: {e}")
            return []
    return []

def save_user_conversation(sender, role, message):
    """Save user conversation to Redis with timestamp"""
    if redis_client:
        try:
            conversation = get_user_conversation(sender)
            conversation.append({
                "role": role,
                "message": message,
                "timestamp": datetime.now().isoformat()
            })
            if len(conversation) > 100:
                conversation = conversation[-100:]
            redis_client.setex(f"conversation:{sender}", timedelta(days=30), json.dumps(conversation))
        except Exception as e:
            logging.error(f"❌ Error saving conversation to Redis: {e}")

def save_user_state(sender, state):
    """Save individual user state to Redis"""
    if not redis_client:
        user_states[sender] = state
        return
        
    try:
        if isinstance(state, dict) and "step" in state:
            redis_client.setex(
                f"user_state:{sender}", 
                timedelta(days=30), 
                json.dumps(state)
            )
    except Exception as e:
        logging.error(f"❌ Error saving user state for {sender} to Redis: {e}")
        user_states[sender] = state

def get_user_state(sender):
    """Get individual user state from Redis"""
    if not redis_client:
        return user_states.get(sender)
        
    try:
        state_data = redis_client.get(f"user_state:{sender}")
        if state_data:
            state = json.loads(state_data)
            if isinstance(state, dict) and "step" in state:
                return state
        return None
    except Exception as e:
        logging.error(f"❌ Error getting user state for {sender} from Redis: {e}")
        return user_states.get(sender)

def detect_language(message):
    """Detect language based on keywords in the message"""
    language_keywords = {
        "english": ["hi", "hello", "hey", "good morning", "good afternoon", "good evening"],
        "shona": ["mhoro", "mhoroi", "makadini", "hesi", "ndinonzi"],
        "ndebele": ["sawubona", "unjani", "salibonani", "yebo"],
        "tonga": ["mwabuka buti", "mwalibizya buti", "kwasiya", "mulibuti"],
        "chinyanja": ["bwanji", "muli bwanji", "mukuli bwanji", "moni"],
        "bemba": ["muli shani", "mulishani", "mwashibukeni", "shani"],
        "lozi": ["muzuhile", "mutozi", "muzuhile cwani", "lwani"]
    }

    message_lower = message.lower()
    for lang, keywords in language_keywords.items():
        if any(keyword in message_lower for keyword in keywords):
            return lang
    return "english"

def send(answer, sender, phone_id):
    """Send message via WhatsApp API"""
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
            "body": answer
        }
    }

    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        logging.debug(f"📤 Message sent to {sender}")
    except Exception as e:
        logging.error(f"❌ Error sending message to {sender}: {e}")
        response = None

    save_user_conversation(sender, "bot", answer)
    return response

def download_whatsapp_media(media_id, file_path):
    """Download WhatsApp media by media_id using the Graph API."""
    try:
        if not media_id:
            logging.error("❌ download_whatsapp_media called with empty media_id")
            return False

        headers = {
            'Authorization': f'Bearer {wa_token}'
        }

        meta_url = f"https://graph.facebook.com/v19.0/{media_id}"
        logging.info(f"📥 Fetching media metadata for media_id={media_id}")
        meta_resp = requests.get(meta_url, headers=headers, timeout=20)
        meta_resp.raise_for_status()
        media_data = meta_resp.json()

        media_url = media_data.get("url")
        if not media_url:
            logging.error(f"❌ No media URL found for media_id={media_id}")
            return False

        logging.info(f"📥 Downloading media content from URL for media_id={media_id}")
        media_resp = requests.get(media_url, headers=headers, timeout=60)
        media_resp.raise_for_status()

        with open(file_path, 'wb') as f:
            f.write(media_resp.content)

        logging.info(f"✅ Media saved to {file_path} for media_id={media_id}")
        return True
    except Exception as e:
        logging.error(f"❌ Error downloading WhatsApp media (media_id={media_id}): {e}")
        return False

def stage_cervical_cancer(image_path):
    """Stage cervical cancer using Hugging Face model"""
    if not hf_client:
        return {
            "stage": "Error",
            "confidence": 0,
            "success": False,
            "error": "Hugging Face model not configured"
        }

    try:
        logging.info("Sending image to Hugging Face model for analysis...")
        result = hf_client.predict(image_path)
        
        logging.info(f"Hugging Face analysis result: {json.dumps(result, indent=2)}")
        
        if result["success"] and result["top_prediction"]:
            top_pred = result["top_prediction"]
            return {
                "stage": top_pred["label"],
                "confidence": float(top_pred["confidence"]),
                "success": True,
                "response_type": "classification",
                "all_predictions": result["predictions"]
            }
        else:
            error_msg = result.get("error", "No valid predictions received")
            return {
                "stage": "Error",
                "confidence": 0,
                "success": False,
                "error": error_msg
            }

    except Exception as e:
        logging.error(f"❌ Staging error: {e}")
        return {
            "stage": "Error",
            "confidence": 0,
            "success": False,
            "error": str(e)
        }

# Database setup (keep your existing database code)
db = False
if os.environ.get("DB_URL"):
    try:
        db = True
        db_url = os.environ.get("DB_URL")
        if db_url and "redis" not in db_url:
            engine = create_engine(db_url)
            Session = sessionmaker(bind=engine)
            Base = declarative_base()

            class Chat(Base):
                __tablename__ = 'chats'
                Chat_no = Column(Integer, primary_key=True)
                Sender = Column(String(255), nullable=False)
                Message = Column(String, nullable=False)
                Chat_time = Column(DateTime, default=datetime.utcnow)

            Base.metadata.create_all(engine)
            logging.info("🗃️ Database tables created")
        else:
            db = False
    except Exception as e:
        logging.error(f"❌ Error setting up database: {e}")
        db = False
else:
    db = False

def handle_language_detection(sender, prompt, phone_id):
    """Handle language detection state"""
    detected_lang = detect_language(prompt)
    user_states[sender]["language"] = detected_lang
    user_states[sender]["step"] = "worker_id"

    if detected_lang == "english":        
        send("Hello! I'm Rudo, Dawa Health's virtual assistant. Let's start with registration. What is your Worker ID?", sender, phone_id)
    
    save_user_state(sender, user_states[sender])

def handle_worker_id(sender, prompt, phone_id):
    """Handle worker ID state"""
    state = user_states[sender]
    lang = state["language"]
    
    state["worker_id"] = prompt
    state["step"] = "patient_id"
    
    if lang == "shona":
        send("Ndatenda! Patient ID yemurwere ndeyipi?", sender, phone_id)
    elif lang == "ndebele":
        send("Ngiyabonga! I-Patient ID yomguli ithini?", sender, phone_id)
    else:
        send("Thank you! What is the Patient ID?", sender, phone_id)
    
    save_user_state(sender, state)

def handle_patient_id(sender, prompt, phone_id):
    """Handle patient ID state"""
    state = user_states[sender]
    lang = state["language"]
    
    state["patient_id"] = prompt
    state["registered"] = True
    state["step"] = "awaiting_image"
    
    if lang == "shona":
        send("Ndatenda! Zvino ndapota tumirai mufananidzo wekuongororwa.", sender, phone_id)
    elif lang == "ndebele":
        send("Ngiyabonga! Manje ngicela uthumele isithombe sokuhlola.", sender, phone_id)
    else:
        send("Thank you! Now you can upload the image for analysis", sender, phone_id)
    
    save_user_state(sender, state)

def handle_cervical_image(sender, media_id, phone_id):
    """Handle cervical cancer image analysis"""
    state = user_states[sender]
    lang = state["language"]

    if state.get("processing_image"):
        return
    
    state["processing_image"] = True
    save_user_state(sender, state)

    try:
        image_path = f"/tmp/{sender}_{int(time.time())}.jpg"

        waiting_messages = {
            "shona": "📨 Ndiri kuongorora mufananidzo wenyu...",
            "ndebele": "📨 Ngiyahlola isithombe sakho...", 
            "english": "📨 Analyzing your image..."
        }

        if download_whatsapp_media(media_id, image_path):
            waiting_message = waiting_messages.get(lang, waiting_messages["english"])
            send(waiting_message, sender, phone_id)

            result = stage_cervical_cancer(image_path)

            worker_id = state.get("worker_id", "Unknown")
            patient_id = state.get("patient_id", "Unknown")

            if result["success"]:
                stage = result["stage"]
                confidence = result["confidence"]

                if lang == "shona":
                    response = f"""🔬 Hugging Face Ongororo:

📋 Worker ID: {worker_id}
👤 Patient ID: {patient_id}
🏥 Danho: {stage}
✅ Chivimbo: {confidence:.1%}

💡 Ziva: Izvi hazvitsivi kuongororwa kwechiremba."""
                elif lang == "ndebele":
                    response = f"""🔬 Imiphumela yeHugging Face:

📋 I-Worker ID: {worker_id}
👤 I-Patient ID: {patient_id}
🏥 Isigaba: {stage}
✅ Ukuthemba: {confidence:.1%}

💡 Qaphela: Lokhu akufaki esikhundleni sokuhlolwa kadokotela."""
                else:
                    response = f"""🔬 Hugging Face Analysis Results:

📋 Worker ID: {worker_id}
👤 Patient ID: {patient_id}
🏥 Stage: {stage}
✅ Confidence: {confidence:.1%}

💡 Note: This does not replace a doctor's diagnosis."""
            else:
                error_msg = result.get("error", "Unknown error")
                if lang == "shona":
                    response = f"""❌ Hatina kukwanisa kuongorora mufananidzo:

Tsaona: {error_msg}

💡 Edza kuendesa imwe mufananidzo."""
                else:
                    response = f"""❌ Analysis failed:

Error: {error_msg}

💡 Please try another image."""

            try:
                os.remove(image_path)
            except:
                pass

            send(response, sender, phone_id)
        else:
            if lang == "shona":
                send("❌ Hatina kukwanisa kugamuchira mufananidzo. Edza zvakare.", sender, phone_id)
            else:
                send("❌ Could not download image. Please try again.", sender, phone_id)

        state["step"] = "follow_up"
        questions = {
            "shona": "Unoda kuendesa imwe mufananidzo here? (Ehe/Aihwa)",
            "ndebele": "Uyafuna ukuthumela esinye isithombe? (Yebo/Cha)", 
            "english": "Would you like to submit another image? (Yes/No)"
        }
        
        question = questions.get(lang, questions["english"])
        send(question, sender, phone_id)

    except Exception as e:
        logging.error(f"❌ Error in handle_cervical_image for {sender}: {e}")
        error_msg = "An error occurred. Please try again."
        if lang == "shona":
            error_msg = "Paine dambudziko. Edza zvakare."
        
        send(f"❌ {error_msg}", sender, phone_id)
        
    finally:
        state["processing_image"] = False
        save_user_state(sender, state)

def handle_follow_up(sender, prompt, phone_id):
    """Handle follow-up after diagnosis"""
    state = user_states[sender]
    lang = state["language"]
    
    prompt_lower = prompt.lower()
    
    if any(word in prompt_lower for word in ["yes", "ehe", "yebo"]):
        state["step"] = "awaiting_image"
        if lang == "shona":
            send("Tumirai imwe mufananidzo wekuongororwa.", sender, phone_id)
        else:
            send("Please upload another image for analysis.", sender, phone_id)
    else:
        state["step"] = "main_menu"
        if lang == "shona":
            send("Ndatenda nekushandisa Dawa Health. Kana uine mimwe mibvunzo, tendera kuti ndikubatsire.", sender, phone_id)
        else:
            send("Thank you for using Dawa Health. If you have more questions, feel free to ask.", sender, phone_id)
    
    save_user_state(sender, state)

def handle_conversation_state(sender, prompt, phone_id, media_url=None, media_type=None):
    """Handle conversation based on current state"""
    state = user_states.get(sender)
    if not state:
        return
    
    prompt_lower = prompt.strip().lower()

    reset_keywords = ["hey", "hi", "hello", "mhoro", "mhoroi", "sawubona"]
    if prompt_lower in reset_keywords:
        user_states[sender] = {
            "step": "language_detection",
            "language": "english",
            "registered": False,
            "worker_id": None,
            "patient_id": None,
            "conversation_history": []
        }
        save_user_state(sender, user_states[sender])
        send("👋 Hello! Let's start again. What language would you like to use?", sender, phone_id)
        return

    if media_type == "image" and state["step"] == "awaiting_image":
        handle_cervical_image(sender, media_url, phone_id)
        return
    
    if state["step"] == "language_detection":
        handle_language_detection(sender, prompt, phone_id)
    elif state["step"] == "worker_id":
        handle_worker_id(sender, prompt, phone_id)
    elif state["step"] == "patient_id":
        handle_patient_id(sender, prompt, phone_id)
    elif state["step"] == "follow_up":
        handle_follow_up(sender, prompt, phone_id)
    elif state["step"] == "main_menu":
        lang = state.get("language", "english")
        fresh_convo = model.start_chat(history=[])
        try:
            fresh_convo.send_message(instructions.instructions)
            fresh_convo.send_message(prompt)
            reply = fresh_convo.last.text

            filtered_reply = re.sub(
                r'(Alright, you are now connected to the backend\.|Here are the links to the product images for Dawa Health:.*?https?://\S+)',
                '', reply, flags=re.DOTALL
            ).strip()

            if filtered_reply:
                send(filtered_reply, sender, phone_id)
            else:
                send("I'm sorry, I didn't understand that. Could you please rephrase?", sender, phone_id)

        except ResourceExhausted as e:
            logging.error(f"❌ Gemini API quota exceeded: {e}")
            send("Sorry, we're experiencing high traffic. Please try again later.", sender, phone_id)
    else:
        state["step"] = "language_detection"
        handle_language_detection(sender, prompt, phone_id)

    save_user_state(sender, state)

def message_handler(data, phone_id):
    global user_states
    
    sender = data["from"]
    logging.info(f"📩 Received message from {sender}")
    
    state = get_user_state(sender)
    if state:
        user_states[sender] = state
    else:
        if sender not in user_states:
            user_states[sender] = {
                "step": "language_detection",
                "language": "english",
                "registered": False,
                "worker_id": None,
                "patient_id": None,
                "conversation_history": []
            }
            save_user_state(sender, user_states[sender])

    if "text" in data:
        prompt = data["text"]["body"]
        save_user_conversation(sender, "user", prompt)
        handle_conversation_state(sender, prompt, phone_id)
    elif "image" in data:
        media_id = data["image"]["id"]
        handle_conversation_state(sender, "", phone_id, media_url=media_id, media_type="image")
    else:
        send("I can only process text and images at the moment.", sender, phone_id)

@app.route('/webhook', methods=['GET', 'POST'])
def webhook():
    if request.method == 'GET':
        mode = request.args.get('hub.mode')
        token = request.args.get('hub.verify_token')
        challenge = request.args.get('hub.challenge')
        if mode == 'subscribe' and token == 'hello':
            logging.info('✅ Webhook verified successfully!')
            return challenge
        else:
            logging.error('❌ Webhook verification failed!')
            return 'Verification failed', 403
    
    elif request.method == 'POST':
        data = request.get_json()
        logging.info(f'📨 Received webhook data')
        
        if data.get("object") == "whatsapp_business_account":
            try:
                for entry in data.get("entry", []):
                    for change in entry.get("changes", []):
                        value = change.get("value", {})
                        if "messages" in value:
                            for message in value["messages"]:
                                if message.get("type") in ["text", "image"]:
                                    threading.Thread(
                                        target=message_handler,
                                        args=(message, phone_id)
                                    ).start()
                return 'OK', 200
            except Exception as e:
                logging.error(f'❌ Error processing webhook: {e}')
                return 'Error processing webhook', 500
        
        return 'Unsupported event type', 400

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "redis_connected": redis_client is not None,
        "huggingface_configured": hf_client is not None,
        "gemini_configured": gen_api is not None,
        "whatsapp_configured": wa_token is not None and phone_id is not None,
        "user_states_count": len(user_states)
    }
    return jsonify(status)

# Load user states on startup
load_user_states()

logging.info("🚀 Application started successfully!")

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
