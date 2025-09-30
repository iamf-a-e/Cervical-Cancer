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
import io
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification

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
            # Method 1: Try direct Upstash connection first
            if 'upstash.io' in redis_url:
                # Extract host from Upstash URL
                parsed = urllib.parse.urlparse(redis_url)
                host = parsed.hostname
                port = 6379  # Default Redis port
                
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
                # Standard Redis URL
                redis_client = redis.from_url(
                    redis_url,
                    password=redis_token,
                    ssl=True,
                    decode_responses=True,
                    socket_connect_timeout=10,
                    socket_timeout=10,
                    retry_on_timeout=True
                )
            
            # Test the connection
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

wa_token = os.environ.get("WA_TOKEN")  # Whatsapp API Key
phone_id = os.environ.get("PHONE_ID")
gen_api = os.environ.get("GEN_API")  # Gemini API Key
owner_phone = os.environ.get("OWNER_PHONE")  # Owner's phone number with countrycode
model_name = "gemini-2.0-flash"
name = "Fae"  # The bot will consider this person as its owner or creator
bot_name = "Rudo"  # This will be the name of your bot, eg: "Hello I am Astro Bot"
AGENT = "+263719835124"  # Fixed: added quotes to make it a string

# Hugging Face Configuration
HF_MODEL_NAME = os.environ.get("HF_MODEL_NAME", "microsoft/resnet-50")
HF_API_TOKEN = os.environ.get("HF_API_TOKEN")
# Use Hugging Face Inference API or local model
HF_USE_INFERENCE_API = os.environ.get("HF_USE_INFERENCE_API", "false").lower() == "true"

# --------------------------------------------------------------------------------
# ✅ Hugging Face Model Client
# --------------------------------------------------------------------------------

class HuggingFaceClient:
    def __init__(self, model_name=HF_MODEL_NAME, use_inference_api=HF_USE_INFERENCE_API, api_token=HF_API_TOKEN):
        self.model_name = model_name
        self.use_inference_api = use_inference_api
        self.api_token = api_token
        self.processor = None
        self.model = None
        
        if not self.use_inference_api:
            # Load model locally
            try:
                logging.info(f"🔄 Loading Hugging Face model locally: {model_name}")
                self.processor = AutoImageProcessor.from_pretrained(model_name)
                self.model = AutoModelForImageClassification.from_pretrained(model_name)
                logging.info("✅ Hugging Face model loaded successfully")
            except Exception as e:
                logging.error(f"❌ Failed to load Hugging Face model: {e}")
                raise
        else:
            logging.info("🔗 Using Hugging Face Inference API")

    def predict(self, image_path):
        """Predict using either local model or Inference API"""
        try:
            if self.use_inference_api:
                return self._predict_inference_api(image_path)
            else:
                return self._predict_local(image_path)
        except Exception as e:
            logging.error(f"❌ Prediction error: {e}")
            raise

    def _predict_local(self, image_path):
        """Predict using locally loaded model"""
        if not self.model or not self.processor:
            raise ValueError("Model not loaded locally")
            
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        
        # Preprocess image
        inputs = self.processor(image, return_tensors="pt")
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = outputs.logits.softmax(dim=-1)
            
        # Get top prediction
        probs, indices = torch.topk(predictions, k=3)
        
        # Convert to readable results
        results = []
        for i in range(len(indices[0])):
            label = self.model.config.id2label[indices[0][i].item()]
            confidence = probs[0][i].item()
            results.append({
                "label": label,
                "confidence": confidence
            })
        
        return {
            "success": True,
            "predictions": results,
            "top_prediction": results[0] if results else None,
            "response_type": "classification"
        }

    def _predict_inference_api(self, image_path):
        """Predict using Hugging Face Inference API"""
        if not self.api_token:
            raise ValueError("Hugging Face API token required for Inference API")
            
        # Read and encode image
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")
        
        # API endpoint
        api_url = f"https://api-inference.huggingface.co/models/{self.model_name}"
        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "inputs": image_data
        }
        
        response = requests.post(api_url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        
        result = response.json()
        
        # Format response
        if isinstance(result, list):
            predictions = result
        elif isinstance(result, dict) and "predictions" in result:
            predictions = result["predictions"]
        else:
            predictions = [result]
        
        # Sort by confidence score (assuming the API returns scores)
        sorted_predictions = sorted(predictions, key=lambda x: x.get('score', 0), reverse=True)
        
        formatted_results = []
        for pred in sorted_predictions[:3]:  # Top 3
            formatted_results.append({
                "label": pred.get('label', 'Unknown'),
                "confidence": pred.get('score', 0)
            })
        
        return {
            "success": True,
            "predictions": formatted_results,
            "top_prediction": formatted_results[0] if formatted_results else None,
            "response_type": "classification"
        }

# Initialize Hugging Face client
hf_client = None
try:
    hf_client = HuggingFaceClient()
    logging.info("✅ Hugging Face client initialized successfully")
except Exception as e:
    logging.error(f"❌ Failed to initialize Hugging Face client: {e}")
    hf_client = None

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
    
    # Check optional but important vars
    optional_vars = {
        "REDIS_URL": redis_url,
        "HF_API_TOKEN": HF_API_TOKEN,
        "OWNER_PHONE": owner_phone,
    }
    
    for var, value in optional_vars.items():
        if not value:
            logging.info(f"ℹ️ Optional variable not set: {var}")

# Validate environment on startup
validate_environment()
    
app = Flask(__name__)
genai.configure(api_key=gen_api)

class CustomURLExtract(URLExtract):
    def _get_cache_file_path(self):
        cache_dir = "/tmp"
        return os.path.join(cache_dir, "tlds-alpha-by-domain.txt")

extractor = CustomURLExtract(limit=1)

# Initialize TLD cache
try:
    extractor.update()
    logging.info("✅ TLD cache updated successfully")
except Exception as e:
    logging.warning(f"⚠️ Could not update TLD cache: {e}. Using built-in fallback.")

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
                        else:
                            logging.warning(f"⚠️ Invalid state structure for {sender}: {state}")
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
            logging.debug(f"💾 Saved conversation for {sender}")
        except Exception as e:
            logging.error(f"❌ Error saving conversation to Redis: {e}")

def save_user_state(sender, state):
    """Save individual user state to Redis with validation and retry"""
    if not redis_client:
        user_states[sender] = state
        logging.debug("💾 Redis client not available, using in-memory storage")
        return
        
    try:
        if isinstance(state, dict) and "step" in state:
            redis_client.setex(
                f"user_state:{sender}", 
                timedelta(days=30), 
                json.dumps(state)
            )
            logging.debug(f"💾 Saved state for {sender}: {state['step']}")
        else:
            logging.error(f"❌ Invalid state structure for {sender}: {state}")
    except Exception as e:
        logging.error(f"❌ Error saving user state for {sender} to Redis: {e}")
        user_states[sender] = state

def get_user_state(sender):
    """Get individual user state from Redis with better error handling"""
    if not redis_client:
        return user_states.get(sender)
        
    try:
        state_data = redis_client.get(f"user_state:{sender}")
        if state_data:
            state = json.loads(state_data)
            if isinstance(state, dict) and "step" in state:
                logging.debug(f"📥 Loaded state for {sender}: {state['step']}")
                return state
            else:
                logging.warning(f"⚠️ Invalid state structure for {sender}: {state}")
                return None
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
    return "english"  # Default to English

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

def remove(*file_paths):
    """Remove files if they exist"""
    for file in file_paths:
        if os.path.exists(file):
            os.remove(file)

def download_image(url, file_path):
    """Download image from URL"""
    try:
        headers = {
            'Authorization': f'Bearer {wa_token}'
        }
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        with open(file_path, 'wb') as f:
            f.write(response.content)
        logging.debug(f"📥 Image downloaded to {file_path}")
        return True
    except Exception as e:
        logging.error(f"❌ Error downloading image: {e}")
        return False

def download_whatsapp_media(media_id, file_path):
    """Download WhatsApp media by media_id using the Graph API."""
    try:
        if not media_id:
            logging.error("❌ download_whatsapp_media called with empty media_id")
            return False

        headers = {
            'Authorization': f'Bearer {wa_token}'
        }

        # Step 1: Get media metadata to retrieve the actual CDN URL
        meta_url = f"https://graph.facebook.com/v19.0/{media_id}"
        logging.info(f"📥 Fetching media metadata for media_id={media_id}")
        meta_resp = requests.get(meta_url, headers=headers, timeout=20)
        meta_resp.raise_for_status()
        media_data = meta_resp.json()

        media_url = media_data.get("url")
        if not media_url:
            logging.error(f"❌ No media URL found for media_id={media_id}. Response: {media_data}")
            return False

        # Step 2: Download the media bytes from the returned URL
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
            return {
                "stage": "Error",
                "confidence": 0,
                "success": False,
                "error": "No valid predictions received"
            }

    except Exception as e:
        logging.error(f"❌ Staging error: {e}")
        return {
            "stage": "Error",
            "confidence": 0,
            "success": False,
            "error": str(e)
        }

# Database setup (optional) - Keep your existing database code
db = False
if os.environ.get("DB_URL"):
    try:
        db = True
        db_url = os.environ.get("DB_URL")
        if db_url and "redis" not in db_url:
            engine = create_engine(db_url)
            Session = sessionmaker(bind=engine)
            Base = declarative_base()
            scheduler = sched.scheduler(time.time, time.sleep)
            report_time = datetime.now().replace(hour=22, minute=00, second=0, microsecond=0)

            class Chat(Base):
                __tablename__ = 'chats'
                Chat_no = Column(Integer, primary_key=True)
                Sender = Column(String(255), nullable=False)
                Message = Column(String, nullable=False)
                Chat_time = Column(DateTime, default=datetime.utcnow)

            logging.info("🗃️ Creating tables if they do not exist...")
            Base.metadata.create_all(engine)

            def insert_chat(sender, message):
                logging.info("💾 Inserting chat into database")
                try:
                    session = Session()
                    chat = Chat(Sender=sender, Message=message)
                    session.add(chat)
                    session.commit()
                    logging.info("✅ Chat inserted successfully")
                except Exception as e:
                    logging.error(f"❌ Error inserting chat: {e}")
                    session.rollback()
                finally:
                    session.close()

            def get_chats(sender):
                try:
                    session = Session()
                    chats = session.query(Chat.Message).filter(Chat.Sender == sender).all()
                    return [chat[0] for chat in chats]
                except Exception as e:
                    logging.error(f"❌ Error getting chats: {e}")
                    return []
                finally:
                    session.close()

            def delete_old_chats():
                try:
                    session = Session()
                    cutoff_date = datetime.now() - timedelta(days=14)
                    session.query(Chat).filter(Chat.Chat_time < cutoff_date).delete()
                    session.commit()
                    logging.info("✅ Old chats deleted successfully")
                except Exception as e:
                    logging.error(f"❌ Error deleting old chats: {e}")
                    session.rollback()
                finally:
                    session.close()

            def create_report(phone_id):
                logging.info("📊 Creating report")
                try:
                    today = datetime.today().strftime('%d-%m-%Y')
                    session = Session()
                    query = session.query(Chat.Message).filter(func.date_trunc('day', Chat.Chat_time) == today).all()
                    if query:
                        chats = '\n\n'.join([chat[0] for chat in query])
                        send(chats, owner_phone, phone_id)
                except Exception as e:
                    logging.error(f"❌ Error creating report: {e}")
                finally:
                    session.close()
        else:
            logging.warning("⚠️ DB_URL appears to be a Redis URL, SQLAlchemy database disabled")
            db = False
    except Exception as e:
        logging.error(f"❌ Error setting up database: {e}")
        db = False
else:
    db = False
    logging.info("ℹ️ DB_URL not set, database functionality disabled")

def handle_language_detection(sender, prompt, phone_id):
    """Handle language detection state"""
    detected_lang = detect_language(prompt)
    user_states[sender]["language"] = detected_lang
    user_states[sender]["step"] = "worker_id"
    user_states[sender]["needs_language_confirmation"] = False

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
    elif lang == "tonga":
        send("Twatotela! Patient ID ya muwandi iyi?", sender, phone_id)
    elif lang == "chinyanja":
        send("Zikomo! Patient ID ya wodwalayo ndi yotani?", sender, phone_id)
    elif lang == "bemba":
        send("Natotela! Patient ID ya mulewele shani?", sender, phone_id)
    elif lang == "lozi":
        send("Ni itumezi! Patient ID ya muwali ki i?", sender, phone_id)
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
    elif lang == "tonga":
        send("Twatotela! Nomba tumizya ciswaswani cekuongolesya.", sender, phone_id)
    elif lang == "chinyanja":
        send("Zikomo! Tsopano chonde tumizani chithunzi choyeserera.", sender, phone_id)
    elif lang == "bemba":
        send("Natotela! Nomba napapata tumishanye icinskana cekupekuleshya.", sender, phone_id)
    elif lang == "lozi":
        send("Ni itumezi! Kacenu, ni lu tumela sitapi sa ku kekula.", sender, phone_id)
    else:
        send("Thank you! Now you can upload the image for amplified VIA analysis", sender, phone_id)
    
    save_user_state(sender, state)

def handle_cervical_image(sender, media_id, phone_id):
    """Handle cervical cancer image analysis with Hugging Face"""
    state = user_states[sender]
    lang = state["language"]

    if state.get("processing_image"):
        print(f"⚠️ Already processing image for {sender}, skipping duplicate")
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

💡 Edza kuendesa imwe mufananidzo kana kumbobvunza chiremba."""
                else:
                    response = f"""❌ Analysis failed:

Error: {error_msg}

💡 Please try another image or consult a doctor."""

            try:
                os.remove(image_path)
            except:
                pass

            send(response, sender, phone_id)

        else:
            if lang == "shona":
                send("❌ Hatina kukwanisa kugamuchira mufananidzo. Edza zvakare.", sender, phone_id)
            elif lang == "ndebele":
                send("❌ Asikwazanga ukulanda isithombe. Zama futhi.", sender, phone_id)
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
        print(f"❌ Error in handle_cervical_image for {sender}: {e}")
        error_msg = "An error occurred during processing. Please try again."
        if lang == "shona":
            error_msg = "Paine dambudziko pakuongorora mufananidzo. Edza zvakare."
        elif lang == "ndebele":
            error_msg = "Kube nephutha ekucutshungeni isithombe. Zama futhi."
        
        send(f"❌ {error_msg}", sender, phone_id)
        
    finally:
        state["processing_image"] = False
        save_user_state(sender, state)

def handle_follow_up(sender, prompt, phone_id):
    """Handle follow-up after diagnosis"""
    state = user_states[sender]
    lang = state["language"]
    
    prompt_lower = prompt.lower()
    
    if any(word in prompt_lower for word in ["yes", "ehe", "yebo", "hongu", "ndinoda"]):
        state["step"] = "awaiting_image"
        if lang == "shona":
            send("Tumirai imwe mufananidzo wekuongororwa.", sender, phone_id)
        else:
            send("Please upload another image for analysis.", sender, phone_id)
    
    else:
        state["step"] = "main_menu"
        if lang == "shona":
            send("Ndatenda nekushandisa Dawa Health neHugging Face technology. Kana uine mimwe mibvunzo, tendera kuti ndikubatsire.", sender, phone_id)
        else:
            send("Thank you for using Dawa Health with Hugging Face technology. If you have more questions, feel free to ask.", sender, phone_id)
    
    save_user_state(sender, state)

def handle_conversation_state(sender, prompt, phone_id, media_url=None, media_type=None):
    """Handle conversation based on current state"""
    state = user_states.get(sender)
    if not state:
        logging.error(f"❌ No state found for {sender}")
        return
    
    prompt_lower = prompt.strip().lower()

    reset_keywords = ["hey", "hi", "hello", "mhoro", "mhoroi", "sawubona", "unjani"]
    if prompt_lower in reset_keywords:
        user_states[sender] = {
            "step": "language_detection",
            "language": "english",
            "needs_language_confirmation": False,
            "registered": False,
            "worker_id": None,
            "patient_id": None,
            "conversation_history": []
        }
        save_user_state(sender, user_states[sender])
        send("👋 Hello! Let's start again. What language would you like to use?", sender, phone_id)
        return

    logging.info(f"💬 Processing message from {sender}, current step: {state['step']}")

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
                if lang == "shona":
                    send("Ndine urombo, handina kunzwisisa. Ungataura zvakare here?", sender, phone_id)
                else:
                    send("I'm sorry, I didn't understand that. Could you please rephrase your question?", sender, phone_id)

        except ResourceExhausted as e:
            logging.error(f"❌ Gemini API quota exceeded: {e}")
            if lang == "shona":
                send("Ndine urombo, tiri kushandisa traffic yakawanda. Edza zvakare gare gare.", sender, phone_id)
            else:
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
        logging.info(f"📥 Loaded existing state for {sender}: {state['step']}")
    else:
        if sender not in user_states:
            user_states[sender] = {
                "step": "language_detection",
                "language": "english",
                "needs_language_confirmation": False,
                "registered": False,
                "worker_id": None,
                "patient_id": None,
                "conversation_history": []
            }
            save_user_state(sender, user_states[sender])
            logging.info(f"🆕 Created new state for {sender}: language_detection")
        else:
            logging.info(f"📥 Using in-memory state for {sender}: {user_states[sender]['step']}")

    if "text" in data:
        prompt = data["text"]["body"]
        save_user_conversation(sender, "user", prompt)
        handle_conversation_state(sender, prompt, phone_id)
    elif "image" in data:
        media_id = data["image"]["id"]
        handle_conversation_state(sender, "", phone_id, media_url=media_id, media_type="image")
    else:
        logging.info(f"❓ Unsupported message type from {sender}")
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
        logging.info(f'📨 Received webhook data: {json.dumps(data, indent=2)}')
        
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

if __name__ == '__main__':
    load_user_states()
    app.run(host='0.0.0.0', port=8080, debug=True)
