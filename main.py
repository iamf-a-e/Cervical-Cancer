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

# Initialize Redis connection for Upstash
redis_url = os.environ.get("UPSTASH_REDIS_URL")
redis_token = os.environ.get("UPSTASH_REDIS_TOKEN")

if redis_url and redis_token:
    try:
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
            if VERTEX_AI_CREDENTIALS_PATH and os.path.exists(VERTEX_AI_CREDENTIALS_PATH):
                self.credentials = service_account.Credentials.from_service_account_file(
                    VERTEX_AI_CREDENTIALS_PATH,
                    scopes=['https://www.googleapis.com/auth/cloud-platform']
                )
                logging.info("Vertex AI credentials loaded successfully")
            else:
                # Try using default credentials
                self.credentials = service_account.Credentials.from_service_account_file(
                    os.environ.get('GOOGLE_APPLICATION_CREDENTIALS', ''),
                    scopes=['https://www.googleapis.com/auth/cloud-platform']
                )
                logging.info("Using default Google Cloud credentials")
        except Exception as e:
            logging.error(f"Error setting up credentials: {e}")
            self.credentials = None
    
    def get_access_token(self):
        """Get access token for API requests"""
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
        logging.info("Vertex AI client initialized successfully")
    except Exception as e:
        logging.error(f"Failed to initialize Vertex AI client: {e}")
        vertex_ai_client = None
else:
    logging.warning("Vertex AI project or endpoint ID not set, cervical cancer staging disabled")

app = Flask(__name__)
genai.configure(api_key=gen_api)

class CustomURLExtract(URLExtract):
    def _get_cache_file_path(self):
        cache_dir = "/tmp"
        return os.path.join(cache_dir, "tlds-alpha-by-domain.txt")

extractor = CustomURLExtract(limit=1)

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
        logging.debug(f"Image downloaded to {file_path}")
        return True
    except Exception as e:
        logging.error(f"Error downloading image: {e}")
        return False

def stage_cervical_cancer(image_path):
    """Stage cervical cancer using Vertex AI REST API"""
    if not vertex_ai_client:
        return {
            "stage": "Error",
            "confidence": 0,
            "success": False,
            "error": "Vertex AI client not configured"
        }
    
    try:
        # Read and encode the image
        with open(image_path, "rb") as f:
            image_data = f.read()
        
        # Prepare the prediction instance for base64 image
        instance = {
            "image_bytes": {"b64": base64.b64encode(image_data).decode()}
        }
        
        # Make prediction using REST API
        prediction_result = vertex_ai_client.predict([instance])
        
        if "error" in prediction_result:
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
            
            return {
                "stage": stage,
                "confidence": float(confidence),
                "success": True
            }
        else:
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

def handle_cervical_image(sender, image_url, phone_id):
    """Handle cervical cancer image for staging"""
    state = user_states[sender]
    lang = state["language"]
    
    # Download the image
    image_path = f"/tmp/{sender}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    
    if lang == "shona":
        send("Ndiri kugamuchira mufananidzo wenyu. Ndapota mirira, ndiri kuongorora.", sender, phone_id)
    else:
        send("I've received your image. Please wait while I analyze it.", sender, phone_id)
    
    if download_image(image_url, image_path):
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
            if lang == "shona":
                response = "Ndine urombo, handina kukwanisa kuongorora mufananidzo wenyu. Edza kuendesa imwe mufananidzo kana kumbobvunza chiremba."
            else:
                response = "I'm sorry, I couldn't analyze your image. Please try sending another image or consult a doctor directly."
        
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

def handle_conversation_state(sender, prompt, phone_id, media_url=None, media_type=None):
    """Handle conversation based on current state"""
    state = user_states.get(sender)
    if not state:
        logging.error(f"No state found for {sender}")
        return
    
    logging.info(f"Processing message from {sender}, current step: {state['step']}")
    
    # Check if we have an image for cervical cancer staging
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
        # For main menu, use Gemini for general queries
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
                "registered": False,
                "worker_id": None,
                "patient_id": None,
                "conversation_history": []
            }
            save_user_state(sender, user_states[sender])
            logging.info(f"Created new state for {sender}")
        else:
            logging.info(f"Using in-memory state for {sender}: {user_states[sender]['step']}")
    
    # Extract message and media
    prompt = ""
    media_url = None
    media_type = None
    
    if data["type"] == "text":
        prompt = data["text"]["body"]
        logging.info(f"Text message: {prompt[:100]}...")
    elif data["type"] == "image":
        media_type = "image"
        media_url = data["image"]["id"]
        # For WhatsApp, we need to download the image using the Media API
        prompt = "[Image received]"
        logging.info(f"Image received: {media_url}")
    
    # Save to conversation history
    save_user_conversation(sender, "user", prompt if prompt else "[Media message]")
    
    # Handle based on current state
    handle_conversation_state(sender, prompt, phone_id, media_url, media_type)
    
    # Daily report and cleanup
    if db:
        scheduler.enterabs(report_time.timestamp(), 1, create_report, (phone_id,))
        scheduler.run(blocking=False)
        delete_old_chats()

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
            logging.info(f"Webhook received: {json.dumps(data, indent=2)}")
            
            entry = data["entry"][0]
            changes = entry["changes"][0]
            value = changes["value"]
            
            # Check if messages exist in the webhook data
            if "messages" in value:
                message_data = value["messages"][0]
                phone_id = value["metadata"]["phone_number_id"]
                message_handler(message_data, phone_id)
        except Exception as e:
            logging.error(f"Error in webhook: {e}")
        return jsonify({"status": "ok"}), 200

@app.route("/download_media/<media_id>", methods=["GET"])
def download_media(media_id):
    """Endpoint to download media from WhatsApp"""
    try:
        url = f"https://graph.facebook.com/v19.0/{media_id}"
        headers = {
            'Authorization': f'Bearer {wa_token}'
        }
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        media_data = response.json()
        
        # Download the actual media content
        media_url = media_data.get("url")
        if media_url:
            media_response = requests.get(media_url, headers=headers)
            media_response.raise_for_status()
            
            # Save the media to a temporary file
            media_path = f"/tmp/{media_id}.jpg"
            with open(media_path, 'wb') as f:
                f.write(media_response.content)
            
            return jsonify({"success": True, "path": media_path})
        else:
            return jsonify({"success": False, "error": "No URL found in media data"})
    except Exception as e:
        logging.error(f"Error downloading media: {e}")
        return jsonify({"success": False, "error": str(e)})

@app.route("/test-redis", methods=["GET"])
def test_redis():
    """Test Redis connectivity"""
    if redis_client:
        try:
            # Test basic operations
            test_key = f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            redis_client.set(test_key, "test_value", ex=10)
            value = redis_client.get(test_key)
            
            # Test state operations
            test_state = {"step": "test", "language": "english"}
            redis_client.setex("test_state:123", timedelta(minutes=1), json.dumps(test_state))
            saved_state = redis_client.get("test_state:123")
            
            return jsonify({
                "status": "success", 
                "redis_working": value == "test_value",
                "state_operations": saved_state is not None,
                "user_states_count": len(user_states)
            })
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)})
    else:
        return jsonify({"status": "error", "message": "Redis client not initialized"})

@app.route("/user-states", methods=["GET"])
def get_user_states():
    """Get current user states for debugging"""
    return jsonify({
        "user_states": user_states,
        "redis_connected": redis_client is not None
    })

if __name__ == "__main__":
    # Load states at startup
    load_user_states()
    logging.info(f"Application started with {len(user_states)} loaded user states")
    app.run(debug=True, port=8000)
