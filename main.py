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
from google.oauth2 import service_account

logging.basicConfig(level=logging.INFO)

# Initialize Redis connection for Upstash with proper error handling
redis_url = os.environ.get("UPSTASH_REDIS_URL")
redis_token = os.environ.get("UPSTASH_REDIS_TOKEN")

redis_client = None
if redis_url and redis_token:
    try:
        # Ensure the URL has the correct scheme
        if not redis_url.startswith(('redis://', 'rediss://')):
            if redis_url.startswith('http://'):
                redis_url = redis_url.replace('http://', 'redis://')
            elif redis_url.startswith('https://'):
                redis_url = redis_url.replace('https://', 'rediss://')
            else:
                # Assume it's a hostname without scheme
                redis_url = f"rediss://{redis_url}"
        
        # Use from_url for better Upstash Redis compatibility
        redis_client = redis.from_url(
            redis_url,
            password=redis_token,
            ssl=True,
            decode_responses=True,
            socket_connect_timeout=5,
            socket_timeout=5
        )
        # Test the connection
        redis_client.ping()
        logging.info("Successfully connected to Upstash Redis")
    except Exception as e:
        logging.error(f"Failed to connect to Upstash Redis: {e}")
        redis_client = None
else:
    redis_client = None
    logging.warning("UPSTASH_REDIS_URL or UPSTASH_REDIS_TOKEN not set, Redis functionality disabled")

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

# Vertex AI REST API configuration (REPLACED the large aiplatform package)
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
        """Setup Google Cloud credentials"""
        try:
            # Try to get credentials from environment variable first
            creds_path = VERTEX_AI_CREDENTIALS_PATH
            
            if creds_path and os.path.exists(creds_path):
                self.credentials = service_account.Credentials.from_service_account_file(
                    creds_path,
                    scopes=['https://www.googleapis.com/auth/cloud-platform']
                )
                logging.info("Vertex AI credentials loaded successfully from file")
            else:
                # Try to get credentials from environment variable (base64 encoded)
                creds_json = os.environ.get("GOOGLE_CREDENTIALS_JSON")
                if creds_json:
                    try:
                        # If it's base64 encoded, decode it
                        if len(creds_json) > 1000:  # Likely base64 encoded
                            creds_json = base64.b64decode(creds_json).decode('utf-8')
                        
                        import json as json_lib
                        creds_dict = json_lib.loads(creds_json)
                        self.credentials = service_account.Credentials.from_service_account_info(
                            creds_dict,
                            scopes=['https://www.googleapis.com/auth/cloud-platform']
                        )
                        logging.info("Vertex AI credentials loaded from environment variable")
                    except Exception as e:
                        logging.error(f"Error loading credentials from environment: {e}")
                        self.credentials = None
                else:
                    # Try using default credentials (for Google Cloud environments)
                    try:
                        self.credentials, _ = google.auth.default()
                        logging.info("Using default Google Cloud credentials")
                    except Exception as e:
                        logging.warning(f"Could not get default credentials: {e}")
                        self.credentials = None
        except Exception as e:
            logging.error(f"Error setting up credentials: {e}")
            self.credentials = None
    
    def get_access_token(self):
        """Get access token for API requests"""
        if self.credentials:
            try:
                import google.auth.transport.requests
                if not self.credentials.valid:
                    self.credentials.refresh(google.auth.transport.requests.Request())
                return self.credentials.token
            except Exception as e:
                logging.error(f"Error getting access token: {e}")
        return None
    
    def predict(self, instances):
        """Make prediction using Vertex AI REST API"""
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
        if vertex_ai_client.credentials:
            logging.info("Vertex AI client initialized successfully")
        else:
            logging.warning("Vertex AI client initialized but no credentials available")
    except Exception as e:
        logging.error(f"Failed to initialize Vertex AI client: {e}")
        vertex_ai_client = None
else:
    logging.warning("Vertex AI project or endpoint ID not set, cervical cancer staging disabled")

app = Flask(__name__)

# Configure Gemini only if API key is available
if gen_api:
    genai.configure(api_key=gen_api)
else:
    logging.warning("GEN_API not set, Gemini functionality disabled")

class CustomURLExtract(URLExtract):
    def _get_cache_file_path(self):
        cache_dir = "/tmp"
        return os.path.join(cache_dir, "tlds-alpha-by-domain.txt")

extractor = CustomURLExtract(limit=1)

# Only create Gemini model if API key is available
if gen_api:
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
else:
    model = None

def save_user_states():
    """Save all user states to Redis"""
    if redis_client:
        try:
            # Save each user state individually for better performance
            for sender, state in user_states.items():
                redis_client.setex(f"user_state:{sender}", timedelta(days=30), json.dumps(state))
            logging.info(f"User states saved to Redis: {len(user_states)} states")
        except Exception as e:
            logging.error(f"Error saving user states to Redis: {e}")

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
                            logging.warning(f"Invalid state structure for {sender}: {state}")
                    except json.JSONDecodeError as e:
                        logging.error(f"Error decoding JSON for {sender}: {e}")
            logging.info(f"Loaded {len(user_states)} user states from Redis")
        except Exception as e:
            logging.error(f"Error loading user states from Redis: {e}")
            user_states = {}
    else:
        user_states = {}
        logging.warning("Redis client not available, using in-memory storage only")

def get_user_conversation(sender):
    """Get user conversation history from Redis"""
    if redis_client:
        try:
            history = redis_client.get(f"conversation:{sender}")
            return json.loads(history) if history else []
        except Exception as e:
            logging.error(f"Error getting conversation from Redis: {e}")
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
            logging.debug(f"Saved conversation for {sender}")
        except Exception as e:
            logging.error(f"Error saving conversation to Redis: {e}")

def save_user_state(sender, state):
    """Save individual user state to Redis with validation"""
    if not redis_client:
        logging.debug("Redis client not available, skipping state save")
        return
        
    try:
        # Validate state structure before saving
        if isinstance(state, dict) and "step" in state:
            redis_client.setex(
                f"user_state:{sender}", 
                timedelta(days=30), 
                json.dumps(state)
            )
            logging.debug(f"Saved state for {sender}: {state['step']}")
        else:
            logging.error(f"Invalid state structure for {sender}: {state}")
    except Exception as e:
        logging.error(f"Error saving user state for {sender} to Redis: {e}")

def get_user_state(sender):
    """Get individual user state from Redis with better error handling"""
    if not redis_client:
        return None
        
    try:
        state_data = redis_client.get(f"user_state:{sender}")
        if state_data:
            state = json.loads(state_data)
            # Validate the state structure
            if isinstance(state, dict) and "step" in state:
                logging.debug(f"Loaded state for {sender}: {state['step']}")
                return state
            else:
                logging.warning(f"Invalid state structure for {sender}: {state}")
                return None
        return None
    except Exception as e:
        logging.error(f"Error getting user state for {sender} from Redis: {e}")
        return None

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
    if not wa_token:
        logging.error("WA_TOKEN not set, cannot send message")
        return None
        
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
        logging.debug(f"Message sent to {sender}")
    except Exception as e:
        logging.error(f"Error sending message to {sender}: {e}")
        response = None

    # Save bot response to conversation history
    save_user_conversation(sender, "bot", answer)

    return response

def remove(*file_paths):
    """Remove files if they exist"""
    for file in file_paths:
        if os.path.exists(file):
            os.remove(file)

def download_whatsapp_image(media_id, file_path):
    """Download image from WhatsApp Media API"""
    if not wa_token:
        logging.error("WA_TOKEN not set, cannot download image")
        return False
        
    try:
        # Step 1: Get the media URL
        media_url = f"https://graph.facebook.com/v19.0/{media_id}"
        headers = {
            'Authorization': f'Bearer {wa_token}'
        }
        
        logging.info(f"Getting media URL for media_id: {media_id}")
        response = requests.get(media_url, headers=headers)
        response.raise_for_status()
        media_data = response.json()
        
        # Step 2: Download the actual image content
        download_url = media_data.get('url')
        if not download_url:
            logging.error(f"No download URL found in media data: {media_data}")
            return False
            
        logging.info(f"Downloading image from: {download_url}")
        download_response = requests.get(download_url, headers=headers)
        download_response.raise_for_status()
        
        # Step 3: Save the image
        with open(file_path, 'wb') as f:
            f.write(download_response.content)
        
        # Verify the file was created and has content
        if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
            logging.info(f"Image successfully downloaded to {file_path} ({os.path.getsize(file_path)} bytes)")
            return True
        else:
            logging.error(f"Downloaded file is empty or doesn't exist: {file_path}")
            return False
            
    except Exception as e:
        logging.error(f"Error downloading WhatsApp image: {e}")
        return False

def stage_cervical_cancer(image_path):
    """Stage cervical cancer using Vertex AI REST API"""
    if not vertex_ai_client:
        logging.error("Vertex AI client not configured")
        return {
            "stage": "Error",
            "confidence": 0,
            "success": False,
            "error": "Vertex AI client not configured"
        }
    
    try:
        # Verify the image exists and has content
        if not os.path.exists(image_path):
            logging.error(f"Image file does not exist: {image_path}")
            return {
                "stage": "Error",
                "confidence": 0,
                "success": False,
                "error": "Image file does not exist"
            }
        
        file_size = os.path.getsize(image_path)
        if file_size == 0:
            logging.error(f"Image file is empty: {image_path}")
            return {
                "stage": "Error",
                "confidence": 0,
                "success": False,
                "error": "Image file is empty"
            }
        
        logging.info(f"Processing image: {image_path} ({file_size} bytes)")
        
        # Read and encode the image
        with open(image_path, "rb") as f:
            image_data = f.read()
        
        # Prepare the prediction instance for base64 image
        instance = {
            "image_bytes": {"b64": base64.b64encode(image_data).decode()}
        }
        
        logging.info("Sending request to Vertex AI endpoint...")
        
        # Make prediction using REST API
        prediction_result = vertex_ai_client.predict([instance])
        
        logging.info(f"Vertex AI response: {prediction_result}")
        
        if "error" in prediction_result:
            logging.error(f"Vertex AI error: {prediction_result['error']}")
            return {
                "stage": "Error",
                "confidence": 0,
                "success": False,
                "error": prediction_result["error"]
            }
        
        # Process the prediction results
        if "predictions" in prediction_result and len(prediction_result["predictions"]) > 0:
            results = prediction_result["predictions"][0]
            
            # Adjust these keys based on your actual model's output format
            stage = results.get('stage', results.get('class', 'Unknown'))
            confidence = results.get('confidence', results.get('score', 0))
            
            logging.info(f"Prediction successful: Stage={stage}, Confidence={confidence}")
            
            return {
                "stage": stage,
                "confidence": float(confidence),
                "success": True
            }
        else:
            logging.error("No predictions returned from model")
            return {
                "stage": "Error",
                "confidence": 0,
                "success": False,
                "error": "No predictions returned from model"
            }
            
    except Exception as e:
        logging.error(f"Error in cervical cancer staging: {e}")
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

            logging.info("Creating tables if they do not exist...")
            Base.metadata.create_all(engine)

            def insert_chat(sender, message):
                logging.info("Inserting chat into database")
                try:
                    session = Session()
                    chat = Chat(Sender=sender, Message=message)
                    session.add(chat)
                    session.commit()
                    logging.info("Chat inserted successfully")
                except Exception as e:
                    logging.error(f"Error inserting chat: {e}")
                    session.rollback()
                finally:
                    session.close()

            def get_chats(sender):
                try:
                    session = Session()
                    chats = session.query(Chat.Message).filter(Chat.Sender == sender).all()
                    return [chat[0] for chat in chats]
                except Exception as e:
                    logging.error(f"Error getting chats: {e}")
                    return []
                finally:
                    session.close()

            def delete_old_chats():
                try:
                    session = Session()
                    cutoff_date = datetime.now() - timedelta(days=14)
                    session.query(Chat).filter(Chat.Chat_time < cutoff_date).delete()
                    session.commit()
                    logging.info("Old chats deleted successfully")
                except Exception as e:
                    logging.error(f"Error deleting old chats: {e}")
                    session.rollback()
                finally:
                    session.close()

            def create_report(phone_id):
                logging.info("Creating report")
                try:
                    today = datetime.today().strftime('%d-%m-%Y')
                    session = Session()
                    query = session.query(Chat.Message).filter(func.date_trunc('day', Chat.Chat_time) == today).all()
                    if query:
                        chats = '\n\n'.join([chat[0] for chat in query])
                        send(chats, owner_phone, phone_id)
                except Exception as e:
                    logging.error(f"Error creating report: {e}")
                finally:
                    session.close()
        else:
            logging.warning("DB_URL appears to be a Redis URL, SQLAlchemy database disabled")
            db = False
    except Exception as e:
        logging.error(f"Error setting up database: {e}")
        db = False
else:
    db = False
    logging.info("DB_URL not set, database functionality disabled")

def handle_language_detection(sender, prompt, phone_id):
    """Handle language detection state"""
    detected_lang = detect_language(prompt)
    user_states[sender]["language"] = detected_lang
    user_states[sender]["step"] = "worker_id"
    user_states[sender]["needs_language_confirmation"] = False

    # Send appropriate greeting based on language
    if detected_lang == "shona":
        send("Mhoro! Ndinonzi Rudo, mubatsiri wepamhepo weDawa Health. Reggai titange nekunyoresa. Worker ID yenyu ndeyipi?", sender, phone_id)
    elif detected_lang == "ndebele":
        send("Sawubona! Ngingu Rudo, isiphathamandla se-Dawa Health. Masige saqala ngokubhalisa. I-Worker ID yakho ithini?", sender, phone_id)
    elif detected_lang == "tonga":
        send("Mwabuka buti! Nine Rudo, munisanga wa Dawa Health. Tuyambile mukubhaliska. Worker ID yobe iyi?", sender, phone_id)
    elif detected_lang == "chinyanja":
        send("Moni! Ndine Rudo, katandizi wa Dawa Health. Tiyambireni ndikulembetsani. Worker ID yanu ndi yotani?", sender, phone_id)
    elif detected_lang == "bemba":
        send("Mwashibukeni! Nine Rudo, umushishi wa Dawa Health. Tulembefye. Worker ID yobe ili shani?", sender, phone_id)
    elif detected_lang == "lozi":
        send("Muzuhile! Nine Rudo, musiyami wa Dawa Health. Re kae ku sa felisize. Worker ID ya hao ki i?", sender, phone_id)
    else:
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
        send("Thank you! Now please send the image for diagnosis.", sender, phone_id)
    
    save_user_state(sender, state)

def handle_cervical_image(sender, media_id, phone_id):
    """Handle cervical cancer image for staging"""
    state = user_states[sender]
    lang = state["language"]
    
    # Download the image
    image_path = f"/tmp/{sender}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    
    if lang == "shona":
        send("Ndiri kugamuchira mufananidzo wenyu. Ndapota mirira, ndiri kuongorora.", sender, phone_id)
    else:
        send("I've received your image. Please wait while I analyze it.", sender, phone_id)
    
    # Download the image from WhatsApp
    if download_whatsapp_image(media_id, image_path):
        # Stage the cervical cancer
        result = stage_cervical_cancer(image_path)
        
        if result["success"]:
            stage = result["stage"]
            confidence = result["confidence"]
            
            # Add worker ID and patient ID to the result
            worker_id = state.get("worker_id", "Unknown")
            patient_id = state.get("patient_id", "Unknown")
            
            if lang == "shona":
                response = f"Mhedzisiro yekuongorora:\n- Worker ID: {worker_id}\n- Patient ID: {patient_id}\n- Danho: {stage}\n- Chivimbo: {confidence:.2%}\n\nNote: Izvi hazvitsivi kuongororwa kwechiremba. Unofanira kuona chiremba kuti uwane kuongororwa kwakazara."
            else:
                response = f"Diagnosis results:\n- Worker ID: {worker_id}\n- Patient ID: {patient_id}\n- Stage: {stage}\n- Confidence: {confidence:.2%}\n\nNote: This does not replace a doctor's diagnosis. Please see a healthcare professional for a complete evaluation."
        else:
            error_msg = result.get("error", "Unknown error")
            if lang == "shona":
                response = f"Ndine urombo, handina kukwanisa kuongorora mufananidzo wenyu. Error: {error_msg}. Edza kuendesa imwe mufananidzo kana kumbobvunza chiremba."
            else:
                response = f"I'm sorry, I couldn't analyze your image. Error: {error_msg}. Please try sending another image or consult a doctor directly."
        
        # Clean up the downloaded image
        remove(image_path)
        
        send(response, sender, phone_id)
    else:
        if lang == "shona":
            send("Ndine urombo, handina kukwanisa kugamuchira mufananidzo wenyu. Edza zvakare.", sender, phone_id)
        else:
            send("I'm sorry, I couldn't download your image. Please try again.", sender, phone_id)
    
    # Ask if they want to submit another image or end the session
    state["step"] = "follow_up"
    if lang == "shona":
        send("Unoda kuendesa imwe mufananidzo here? (Reply 'Ehe' for yes or 'Aihwa' for no)", sender, phone_id)
    else:
        send("Would you like to submit another image? (Reply 'Yes' or 'No')", sender, phone_id)
    
    save_user_state(sender, state)

def handle_follow_up(sender, prompt, phone_id):
    """Handle follow-up after diagnosis"""
    state = user_states[sender]
    lang = state["language"]
    
    prompt_lower = prompt.lower()
    
    if any(word in prompt_lower for word in ["yes", "ehe", "yebo", "hongu", "ndinoda"]):
        # Reset to patient ID step for new diagnosis
        state["step"] = "patient_id"
        if lang == "shona":
            send("Patient ID yemurwere itsva ndeyipi?", sender, phone_id)
        else:
            send("What is the Patient ID for the new patient?", sender, phone_id)
    else:
        # End session
        state["step"] = "main_menu"
        if lang == "shona":
            send("Ndatenda nekushandisa Dawa Health. Kana uine mimwe mibvunzo, tendera kuti ndikubatsire.", sender, phone_id)
        else:
            send("Thank you for using Dawa Health. If you have more questions, feel free to ask.", sender, phone_id)
    
    save_user_state(sender, state)

def handle_conversation_state(sender, prompt, phone_id, media_id=None, media_type=None):
    """Handle conversation based on current state"""
    state = user_states.get(sender)
    if not state:
        logging.error(f"No state found for {sender}")
        return
    
    logging.info(f"Processing message from {sender}, current step: {state['step']}")
    
    # Check if we have an image for cervical cancer staging
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
    elif state["step"] == "main_menu":
        # For main menu, use Gemini for general queries
        if not model:
            lang = state.get("language", "english")
            if lang == "shona":
                send("Ndine urombo, basa reGemini harisi kushanda parizvino. Ndapota edza gare gare.", sender, phone_id)
            else:
                send("I'm sorry, the Gemini service is currently unavailable. Please try again later.", sender, phone_id)
            return
            
        lang = state.get("language", "english")
        fresh_convo = model.start_chat(history=[])
        try:
            fresh_convo.send_message(instructions.instructions)
            fresh_convo.send_message(prompt)
            reply = fresh_convo.last.text
            
            # Filter out any internal instructions
            filtered_reply = re.sub(r'(Alright, you are now connected to the backend\.|Here are the links to the product images for Dawa Health:.*?https?://\S+)', '', reply, flags=re.DOTALL)
            filtered_reply = filtered_reply.strip()
            
            if filtered_reply:
                send(filtered_reply, sender, phone_id)
            else:
                if lang == "shona":
                    send("Ndine urombo, handina kunzwisisa. Ungataura zvakare here?", sender, phone_id)
                else:
                    send("I'm sorry, I didn't understand that. Could you please rephrase your question?", sender, phone_id)
                    
        except ResourceExhausted as e:
            logging.error(f"Gemini API quota exceeded: {e}")
            if lang == "shona":
                send("Ndine urombo, tiri kushandisa traffic yakawanda. Edza zvakare gare gare.", sender, phone_id)
            else:
                send("Sorry, we're experiencing high traffic. Please try again later.", sender, phone_id)
        except Exception as e:
            logging.error(f"Error with Gemini: {e}")
            if lang == "shona":
                send("Ndine urombo, pane chakakanganisika. Edza zvakare gare gare.", sender, phone_id)
            else:
                send("I'm sorry, something went wrong. Please try again later.", sender, phone_id)
    else:
        # Default to language detection if state is unknown
        state["step"] = "language_detection"
        handle_language_detection(sender, prompt, phone_id)
    
    save_user_state(sender, state)

def message_handler(data, phone_id):
    global user_states
    
    sender = data["from"]
    logging.info(f"Received message from {sender}")
    
    # Load user state from Redis with better handling
    state = get_user_state(sender)
    if state:
        user_states[sender] = state
        logging.info(f"Loaded existing state for {sender}: {state['step']}")
    else:
        # Only initialize if truly new user (not in memory either)
        if sender not in user_states:
            user_states[sender] = {
                "step": "language_detection",
                "language": "english",
                "needs_language_confirmation": False,
                "worker_id": None,
                "patient_id": None,
                "registered": False
            }
            logging.info(f"Created new state for {sender}: language_detection")
    
    # Handle different message types
    if "text" in data:
        prompt = data["text"]["body"]
        logging.info(f"Text message from {sender}: {prompt}")
        
        # Save user message to conversation history
        save_user_conversation(sender, "user", prompt)
        
        # Handle conversation state
        handle_conversation_state(sender, prompt, phone_id)
        
    elif "image" in data:
        media_id = data["image"]["id"]
        logging.info(f"Image message from {sender}, media_id: {media_id}")
        
        # Save user message to conversation history
        save_user_conversation(sender, "user", "[IMAGE]")
        
        # Handle conversation state with image
        handle_conversation_state(sender, "", phone_id, media_id=media_id, media_type="image")
        
    else:
        logging.warning(f"Unhandled message type from {sender}: {data.keys()}")
        send("I'm sorry, I can only process text and image messages at the moment.", sender, phone_id)

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/webhook", methods=["GET"])
def webhook_get():
    if request.args.get("hub.verify_token") == "Rudo":
        return request.args.get("hub.challenge")
    return "Authentication failed. Invalid Token."

@app.route("/webhook", methods=["POST"])
def webhook_post():
    try:
        data = request.get_json()
        logging.info(f"Received webhook data: {data}")
        
        if data.get("object") == "whatsapp_business_account":
            for entry in data.get("entry", []):
                for change in entry.get("changes", []):
                    value = change.get("value", {})
                    if "messages" in value:
                        for message in value["messages"]:
                            message_handler(message, phone_id)
        
        return jsonify({"status": "success"}), 200
    except Exception as e:
        logging.error(f"Error in webhook: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "redis": redis_client is not None,
            "gemini": model is not None,
            "vertex_ai": vertex_ai_client is not None,
            "whatsapp": bool(wa_token and phone_id)
        }
    }
    return jsonify(status)

# Load user states on startup
load_user_states()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
