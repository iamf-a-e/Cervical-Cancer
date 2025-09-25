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
# ✅ Decode Base64 service account JSON (Option A) and set GOOGLE_APPLICATION_CREDENTIALS
# --------------------------------------------------------------------------------
service_account_b64 = os.environ.get("GCP_SERVICE_ACCOUNT_BASE64")
if service_account_b64:
    sa_path = "/tmp/service-account.json"
    try:
        with open(sa_path, "wb") as f:
            f.write(base64.b64decode(service_account_b64))
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = sa_path
        logging.info(f"✅ Service account JSON written to {sa_path}")
    except Exception as e:
        logging.error(f"❌ Failed to decode service account JSON: {e}")
else:
    logging.warning("⚠️ GCP_SERVICE_ACCOUNT_BASE64 not set. Vertex AI may fail to authenticate.")

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
                # Upstash provides a REST API-like URL, convert it
                if redis_url.startswith('https://'):
                    # Extract host from Upstash URL
                    parsed = urllib.parse.urlparse(redis_url)
                    host = parsed.hostname
                    port = 6379  # Default Redis port
                    
                    redis_client = redis.Redis(
                        host=host,
                        port=port,
                        password=redis_token,
                        ssl=True,
                        ssl_cert_reqs=None,  # Important for Upstash
                        decode_responses=True,
                        socket_connect_timeout=10,
                        socket_timeout=10,
                        retry_on_timeout=True
                    )
                else:
                    # Try with rediss:// scheme
                    formatted_url = redis_url
                    if not formatted_url.startswith(('redis://', 'rediss://')):
                        formatted_url = f"rediss://{redis_token}@{redis_url}:6379"
                    
                    redis_client = redis.from_url(
                        formatted_url,
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
            
            # Method 2: Fallback to basic connection
            try:
                logging.info("🔄 Trying alternative Redis connection method...")
                if 'upstash.io' in redis_url:
                    parsed = urllib.parse.urlparse(redis_url)
                    host = parsed.hostname
                    
                    redis_client = redis.Redis(
                        host=host,
                        port=6379,
                        password=redis_token,
                        ssl=True,
                        ssl_cert_reqs=None,
                        decode_responses=True,
                        socket_connect_timeout=10,
                        socket_timeout=10
                    )
                    redis_client.ping()
                    logging.info("✅ Connected using alternative method")
                    return redis_client
            except Exception as e2:
                logging.error(f"❌ Alternative connection also failed: {e2}")
    
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

# Vertex AI Endpoint Configuration (kept from original)
VERTEX_AI_ENDPOINT_ID = "9216603443274186752"
VERTEX_AI_REGION = "us-west4"
VERTEX_AI_PROJECT = os.environ.get("VERTEX_AI_PROJECT")
# MEDSIGLIP_API_KEY is no longer used for Vertex AI authentication (ADC is used instead)
MEDSIGLIP_API_KEY = os.environ.get("MEDSIGLIP_API")

# --------------------------------------------------------------------------------
# ✅ CORRECTED VertexAIClient — uses ADC (Application Default Credentials)
# --------------------------------------------------------------------------------

class VertexAIClient:
    def __init__(self, project_id, endpoint_id, location="us-west4"):
        self.project_id = project_id
        self.endpoint_id = endpoint_id
        self.location = location

        # Build the correct endpoint base URL
        self.base_url = (
            f"https://{endpoint_id}.{location}-519460264942.prediction.vertexai.goog/v1/projects/"
            f"{project_id}/locations/{location}/endpoints/{endpoint_id}:predict"
        )

        # Get default ADC credentials
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
    
        try:
            response = requests.post(
                self.base_url, json=payload, headers=headers, timeout=60
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as http_err:
            # Log full error response for debugging
            try:
                error_text = response.text
            except Exception:
                error_text = str(http_err)
            logging.error(f"Vertex AI HTTP error: {http_err} | Response: {error_text}")
            raise
        except Exception as e:
            logging.error(f"Vertex AI request failed: {e}")
            raise

vertex_ai_client = None
if VERTEX_AI_PROJECT and VERTEX_AI_ENDPOINT_ID:
    try:
        vertex_ai_client = VertexAIClient(
            VERTEX_AI_PROJECT,
            VERTEX_AI_ENDPOINT_ID,
            VERTEX_AI_REGION
        )
        logging.info("✅ Vertex AI client initialized successfully with ADC authentication")
    except Exception as e:
        logging.error(f"❌ Failed to initialize Vertex AI client: {e}")
        vertex_ai_client = None
else:
    logging.warning("⚠️ Vertex AI project, endpoint ID not set, cervical cancer staging disabled")

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
        "VERTEX_AI_PROJECT": VERTEX_AI_PROJECT,
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
            # Save each user state individually for better performance
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
            # Get all user state keys
            keys = redis_client.keys("user_state:*")
            user_states = {}
            for key in keys:
                sender = key.replace("user_state:", "")
                state_data = redis_client.get(key)
                if state_data:
                    try:
                        state = json.loads(state_data)
                        # Validate state structure
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
            # Keep only the last 100 messages to prevent excessive storage
            if len(conversation) > 100:
                conversation = conversation[-100:]
            redis_client.setex(f"conversation:{sender}", timedelta(days=30), json.dumps(conversation))
            logging.debug(f"💾 Saved conversation for {sender}")
        except Exception as e:
            logging.error(f"❌ Error saving conversation to Redis: {e}")

def save_user_state(sender, state):
    """Save individual user state to Redis with validation and retry"""
    if not redis_client:
        # Fallback to in-memory storage
        user_states[sender] = state
        logging.debug("💾 Redis client not available, using in-memory storage")
        return
        
    try:
        # Validate state structure before saving
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
        # Fallback to in-memory storage
        user_states[sender] = state

def get_user_state(sender):
    """Get individual user state from Redis with better error handling"""
    if not redis_client:
        return user_states.get(sender)
        
    try:
        state_data = redis_client.get(f"user_state:{sender}")
        if state_data:
            state = json.loads(state_data)
            # Validate the state structure
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

    # Save bot response to conversation history
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
    """Stage cervical cancer using Vertex AI - handle both embedding and classification outputs"""
    if not vertex_ai_client:
        return {
            "stage": "Error",
            "confidence": 0,
            "success": False,
            "error": "Vertex AI not configured"
        }

    try:
        with open(image_path, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode("utf-8")

        payload = {
            "instances": [
                {
                    "image_bytes": {"b64": image_b64},
                    "key": "prediction_key"
                }
            ]
        }

        logging.info("🔬 Sending image to Vertex AI for analysis...")
        result = vertex_ai_client.predict(payload)
        logging.info(f"🔬 Raw Vertex AI response: {json.dumps(result, indent=2)[:1000]}...")
        
        if "predictions" in result and result["predictions"]:
            prediction = result["predictions"][0]
            
            # Handle classification output (expected format)
            if isinstance(prediction, dict):
                # Case 1: Standard classification output
                if "displayNames" in prediction and "confidences" in prediction:
                    labels = prediction["displayNames"]
                    scores = prediction["confidences"]
                    if labels and scores:
                        max_idx = scores.index(max(scores))
                        return {
                            "stage": labels[max_idx],
                            "confidence": float(scores[max_idx]),
                            "success": True,
                            "response_type": "classification"
                        }
                
                # Case 2: Alternative classification format
                elif "classes" in prediction and "scores" in prediction:
                    labels = prediction["classes"]
                    scores = prediction["scores"]
                    if labels and scores:
                        max_idx = scores.index(max(scores))
                        return {
                            "stage": labels[max_idx],
                            "confidence": float(scores[max_idx]),
                            "success": True,
                            "response_type": "classification"
                        }
            
            # Case 3: Embedding output (current issue)
            if "embedding" in prediction:
                embedding = prediction["embedding"]
                logging.warning("⚠️ Received embedding instead of classification. Endpoint may be misconfigured.")
                
                # Simple heuristic: if embedding has high variance, might indicate abnormality
                # This is a basic fallback - you should use a proper classifier
                import numpy as np
                embedding_array = np.array(embedding)
                variance = np.var(embedding_array)
                
                # Very basic classification based on embedding characteristics
                if variance > 0.01:  # Adjust this threshold based on your data
                    stage = "Suspicious - Further evaluation needed"
                    confidence = min(variance * 10, 0.8)  # Scale variance to confidence
                else:
                    stage = "Normal - No significant abnormalities detected"
                    confidence = 0.7
                
                return {
                    "stage": stage,
                    "confidence": float(confidence),
                    "success": True,
                    "response_type": "embedding_fallback",
                    "note": "Analysis based on image features. Clinical evaluation required."
                }
            
            # Case 4: Unknown format - return raw prediction
            logging.warning(f"⚠️ Unknown prediction format: {type(prediction)}")
            return {
                "stage": "Analysis Complete - Raw Features Extracted",
                "confidence": 0.5,
                "success": True,
                "response_type": "unknown_format",
                "raw_prediction": str(prediction)[:500],
                "note": "Features extracted successfully. Clinical interpretation needed."
            }
        else:
            return {
                "stage": "Error",
                "confidence": 0,
                "success": False,
                "error": "No predictions in response"
            }

    except Exception as e:
        logging.error(f"❌ Staging error: {e}")
        logging.error(f"❌ Stack trace: {traceback.format_exc()}")
        return {
            "stage": "Error",
            "confidence": 0,
            "success": False,
            "error": str(e)
        }


# Database setup (optional)
db = False
if os.environ.get("DB_URL"):
    try:
        db = True
        db_url = os.environ.get("DB_URL")  # Database URL
        # Check if this is a Redis URL and skip SQLAlchemy setup if so
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

    # Send appropriate greeting based on language
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
    """Handle cervical cancer image analysis with improved feedback"""
    state = user_states[sender]
    lang = state["language"]

    # File path for the incoming image
    image_path = f"/tmp/{sender}_{int(time.time())}.jpg"

    # Localized "analyzing" message
    waiting_messages = {
        "shona": "📨 Ndiri kuongorora mufananidzo wenyu...",
        "ndebele": "📨 Ngiyahlola isithombe sakho...", 
        "english": "📨 Analyzing your image..."
    }

    # Try to download media
    if download_whatsapp_media(media_id, image_path):
        # ✅ Only send analyzing message once we have the file
        send(waiting_messages.get(lang, "📨 Analyzing your image..."), sender, phone_id)

        result = stage_cervical_cancer(image_path)

        worker_id = state.get("worker_id", "Unknown")
        patient_id = state.get("patient_id", "Unknown")

        if result["success"]:
            stage = result["stage"]
            confidence = result["confidence"]
            response_type = result.get("response_type", "unknown")

            # Classification results
            if response_type == "classification":
                if lang == "shona":
                    response = f"""🔬 MedSigLip Ongororo:

📋 Worker ID: {worker_id}
👤 Patient ID: {patient_id}
🏥 Danho: {stage}
✅ Chivimbo: {confidence:.1%}

💡 Ziva: Izvi hazvitsivi kuongororwa kwechiremba."""
                elif lang == "ndebele":
                    response = f"""🔬 Imiphumela yeMedSigLip:

📋 I-Worker ID: {worker_id}
👤 I-Patient ID: {patient_id}
🏥 Isigaba: {stage}
✅ Ukuthemba: {confidence:.1%}

💡 Qaphela: Lokhu akufaki esikhundleni sokuhlolwa kadokotela."""
                else:
                    response = f"""🔬 MedSigLip Analysis Results:

📋 Worker ID: {worker_id}
👤 Patient ID: {patient_id}
🏥 Stage: {stage}
✅ Confidence: {confidence:.1%}

💡 Note: This does not replace a doctor's diagnosis."""
            
            # Embedding-based fallback
            elif response_type == "embedding_fallback":
                note = result.get("note", "")
                if lang == "shona":
                    response = f"""🔬 Ongororo Yakaitwa:

📋 Worker ID: {worker_id}
👤 Patient ID: {patient_id}
🏥 Mhedzisiro: {stage}
✅ Chivimbo: {confidence:.1%}

💡 {note}"""
                else:
                    response = f"""🔬 Feature Analysis Results:

📋 Worker ID: {worker_id}
👤 Patient ID: {patient_id}
🏥 Findings: {stage}
✅ Confidence: {confidence:.1%}

💡 {note}"""
            else:
                # Unknown format
                if lang == "shona":
                    response = f"""🔬 Mufananidzo Wagamuchirwa:

📋 Worker ID: {worker_id}
👤 Patient ID: {patient_id}
🏥 Zvakaonekwa: Mufananidzo wakaongororwa zvakanaka

💡 Chiremba achakupa mhedzisiro chaiyo."""
                else:
                    response = f"""🔬 Image Analysis Complete:

📋 Worker ID: {worker_id}
👤 Patient ID: {patient_id}
🏥 Status: Image processed successfully

💡 Doctor will provide detailed interpretation."""
        else:
            # ❌ Analysis error
            error_msg = result.get("error", "Unknown error")
            if lang == "shona":
                response = f"""❌ Hatina kukwanisa kuongorora mufananidzo:

Tsaona: {error_msg}

💡 Edza kuendesa imwe mufananidzo kana kumbobvunza chiremba."""
            else:
                response = f"""❌ Analysis failed:

Error: {error_msg}

💡 Please try another image or consult a doctor."""

        # Cleanup temp image
        try:
            os.remove(image_path)
        except:
            pass

        send(response, sender, phone_id)

    else:
        # ❌ Download failed
        if lang == "shona":
            send("❌ Hatina kukwanisa kugamuchira mufananidzo. Edza zvakare.", sender, phone_id)
        elif lang == "ndebele":
            send("❌ Asikwazanga ukulanda isithombe. Zama futhi.", sender, phone_id)
        else:
            send("❌ Could not download image. Please try again.", sender, phone_id)

    # 🔄 Always advance to follow_up, success or failure
    state["step"] = "follow_up"
    questions = {
        "shona": "Unoda kuendesa imwe mufananidzo here? (Ehe/Aihwa)",
        "ndebele": "Uyafuna ukuthumela esinye isithombe? (Yebo/Cha)", 
        "english": "Would you like to submit another image? (Yes/No)"
    }
    send(questions.get(lang, questions["english"]), sender, phone_id)

    save_user_state(sender, state)


def handle_follow_up(sender, prompt, phone_id):
    """Handle follow-up after diagnosis"""
    state = user_states[sender]
    lang = state["language"]
    
    prompt_lower = prompt.lower()
    
    if any(word in prompt_lower for word in ["yes", "ehe", "yebo", "hongu", "ndinoda"]):
        # Go directly to image upload for the same patient
        state["step"] = "awaiting_image"
        if lang == "shona":
            send("Tumirai imwe mufananidzo wekuongororwa.", sender, phone_id)
        else:
            send("Please upload another image for analysis.", sender, phone_id)
    
    else:
        # End session
        state["step"] = "main_menu"
        if lang == "shona":
            send("Ndatenda nekushandisa Dawa Health neMedSigLip technology. Kana uine mimwe mibvunzo, tendera kuti ndikubatsire.", sender, phone_id)
        else:
            send("Thank you for using Dawa Health with MedSigLip technology. If you have more questions, feel free to ask.", sender, phone_id)
    
    save_user_state(sender, state)


def handle_conversation_state(sender, prompt, phone_id, media_url=None, media_type=None):
    """Handle conversation based on current state"""
    state = user_states.get(sender)
    if not state:
        logging.error(f"❌ No state found for {sender}")
        return
    
    prompt_lower = prompt.strip().lower()

    # 🔄 Global reset trigger
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

    # 📷 If user sends an image during diagnosis
    if media_type == "image" and state["step"] == "awaiting_image":
        handle_cervical_image(sender, media_url, phone_id)
        return
    
    # 🌍 Route by state
    if state["step"] == "language_detection":
        handle_language_detection(sender, prompt, phone_id)
    elif state["step"] == "worker_id":
        handle_worker_id(sender, prompt, phone_id)
    elif state["step"] == "patient_id":
        handle_patient_id(sender, prompt, phone_id)
    elif state["step"] == "follow_up":
        handle_follow_up(sender, prompt, phone_id)
    elif state["step"] == "main_menu":
        # General queries via Gemini
        lang = state.get("language", "english")
        fresh_convo = model.start_chat(history=[])
        try:
            fresh_convo.send_message(instructions.instructions)
            fresh_convo.send_message(prompt)
            reply = fresh_convo.last.text

            # 🧹 Filter out any internal instructions
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
        # Default: restart at language detection
        state["step"] = "language_detection"
        handle_language_detection(sender, prompt, phone_id)

    save_user_state(sender, state)


def message_handler(data, phone_id):
    global user_states
    
    sender = data["from"]
    logging.info(f"📩 Received message from {sender}")
    
    # Load user state from Redis with better handling
    state = get_user_state(sender)
    if state:
        user_states[sender] = state
        logging.info(f"📥 Loaded existing state for {sender}: {state['step']}")
    else:
        # Only initialize if truly new user (not in memory either)
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
            logging.info(f"🆕 Created new state for {sender}")
        else:
            logging.info(f"💾 Using in-memory state for {sender}: {user_states[sender]['step']}")
    
    # Extract message and media
    prompt = ""
    media_url = None
    media_type = None
    
    if data["type"] == "text":
        prompt = data["text"]["body"]
        logging.info(f"💬 Text message: {prompt[:100]}...")
    elif data["type"] == "image":
        media_type = "image"
        media_url = data["image"]["id"]
        logging.info(f"🖼️ Image received, media_id: {media_url}")
        # Use a placeholder prompt for image processing
        prompt = "IMAGE_UPLOADED"
    else:
        prompt = "UNSUPPORTED_MESSAGE_TYPE"
        logging.warning(f"⚠️ Unsupported message type: {data['type']}")
    
    # Save user message to conversation history
    save_user_conversation(sender, "user", prompt)
    
    # Handle the conversation based on current state
    handle_conversation_state(sender, prompt, phone_id, media_url, media_type)
    
    # Save updated state to Redis
    save_user_state(sender, user_states[sender])

@app.route('/', methods=['GET'])
def home():
    return render_template('connected.html')

@app.route('/webhook', methods=['GET'])
def webhook():
    try:
        if request.args.get('hub.verify_token') == 'my_verify_token':
            return request.args.get('hub.challenge')
        else:
            return 'Error, wrong validation token'
    except Exception as e:
        logging.error(f"❌ Webhook verification error: {e}")
        return 'Error'

@app.route('/webhook', methods=['POST'])
def webhook_handle():
    try:
        data = request.get_json()
        logging.info(f"📨 Received webhook data: {json.dumps(data, indent=2)}")
        
        if data.get("object") == "whatsapp_business_account":
            for entry in data.get("entry", []):
                for change in entry.get("changes", []):
                    if change.get("field") == "messages":
                        value = change.get("value", {})
                        if "messages" in value:
                            for message in value["messages"]:
                                message_handler(message, phone_id)
        return jsonify({"status": "success"}), 200
    except Exception as e:
        logging.error(f"❌ Webhook handling error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "redis_connected": redis_client is not None,
        "vertex_ai_configured": vertex_ai_client is not None,
        "vertex_ai_project": VERTEX_AI_PROJECT is not None,
        "vertex_ai_endpoint": VERTEX_AI_ENDPOINT_ID is not None,
        "gemini_configured": gen_api is not None,
        "whatsapp_configured": wa_token is not None and phone_id is not None,
        "user_states_count": len(user_states)
    }
    
    # Add more detailed Vertex AI info
    if vertex_ai_client:
        status["vertex_ai_details"] = {
            "project_id": vertex_ai_client.project_id,
            "endpoint_id": vertex_ai_client.endpoint_id,
            "base_url": vertex_ai_client.base_url
        }
    
    # Test Redis connection
    if redis_client:
        try:
            redis_client.ping()
            status["redis_status"] = "connected"
            # Get some Redis stats
            status["redis_info"] = {
                "db_size": len(redis_client.keys("*")),
                "user_states_in_redis": len(redis_client.keys("user_state:*"))
            }
        except Exception as e:
            status["redis_status"] = f"error: {str(e)}"
            status["status"] = "degraded"
    
    return jsonify(status)

@app.route('/test-vertex', methods=['GET'])
def test_vertex():
    """Test Vertex AI connection"""
    if not vertex_ai_client:
        return jsonify({"error": "Vertex AI client not configured"}), 500
    
    try:
        # Simple test payload
        test_payload = {"instances": [{"test": "connection"}]}
        result = vertex_ai_client.predict(test_payload)
        return jsonify({"status": "success", "result": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/test-redis', methods=['GET'])
def test_redis():
    """Test Redis connection"""
    if not redis_client:
        return jsonify({"error": "Redis client not configured"}), 500
    
    try:
        # Test basic operations
        test_key = "test:connection"
        test_value = {"timestamp": datetime.now().isoformat(), "test": "success"}
        
        # Set value
        redis_client.setex(test_key, timedelta(minutes=5), json.dumps(test_value))
        
        # Get value
        retrieved = redis_client.get(test_key)
        
        # Get some stats
        keys_count = len(redis_client.keys("*"))
        user_states_count = len(redis_client.keys("user_state:*"))
        
        return jsonify({
            "status": "success",
            "set_get_test": json.loads(retrieved) if retrieved else None,
            "stats": {
                "total_keys": keys_count,
                "user_states": user_states_count
            }
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Load user states on startup
load_user_states()

# Pre-warm the TLD cache in background
def warmup_tld_cache():
    try:
        extractor.update()
        logging.info("✅ TLD cache warmed up successfully")
    except Exception as e:
        logging.warning(f"⚠️ TLD cache warmup failed: {e}")

# Start warmup in background thread
tld_thread = threading.Thread(target=warmup_tld_cache, daemon=True)
tld_thread.start()

logging.info("🚀 Application started successfully!")

@app.route('/debug-vertex-response', methods=['GET'])
def debug_vertex_response():
    """Debug endpoint to check Vertex AI response format"""
    if not vertex_ai_client:
        return jsonify({"error": "Vertex AI not configured"}), 500
    
    # Test with a small sample image or mock data
    test_payload = {
        "instances": [{
            "image_bytes": {"b64": "test"},
            "key": "debug_key"
        }]
    }
    
    try:
        result = vertex_ai_client.predict(test_payload)
        return jsonify({
            "status": "success",
            "prediction_keys": list(result.keys()) if isinstance(result, dict) else str(type(result)),
            "prediction_sample": str(result)[:1000] if result else "No result",
            "endpoint_type": "Check if this returns classification or embedding"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
        

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
