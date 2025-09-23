import google.generativeai as genai
from flask import Flask, request, jsonify, render_template
import requests
import os
import fitz
import sched
import time
import logging
from mimetypes import guess_type
from datetime import datetime,timedelta
from urlextract import URLExtract
from training import instructions, product_images
from sqlalchemy import create_engine, Column, Integer, String, DateTime, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from google.api_core.exceptions import ResourceExhausted
from training import products, pregnancy_data, pregnancy_data_shona, pregnancy_data_ndebele, pregnancy_data_tonga, pregnancy_data_chinyanja, pregnancy_data_bemba, pregnancy_data_lozi


logging.basicConfig(level=logging.INFO)
user_states = {}  

db=False
wa_token=os.environ.get("WA_TOKEN") # Whatsapp API Key
phone_id = os.environ.get("PHONE_ID")
gen_api=os.environ.get("GEN_API") # Gemini API Key
owner_phone=os.environ.get("OWNER_PHONE") # Owner's phone number with countrycode
model_name="gemini-2.0-flash"
name="Fae" #The bot will consider this person as its owner or creator
bot_name="Rudo" #This will be the name of your bot, eg: "Hello I am Astro Bot"
AGENT = +263719835124


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
  {"category": "HARM_CATEGORY_HARASSMENT","threshold": "BLOCK_MEDIUM_AND_ABOVE"},
  {"category": "HARM_CATEGORY_HATE_SPEECH","threshold": "BLOCK_MEDIUM_AND_ABOVE"},  
  {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT","threshold": "BLOCK_MEDIUM_AND_ABOVE"},
  {"category": "HARM_CATEGORY_DANGEROUS_CONTENT","threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]

model = genai.GenerativeModel(model_name=model_name,
                              generation_config=generation_config,
                              safety_settings=safety_settings)

convo = model.start_chat(history=[])
convo.send_message(instructions.instructions)



def send(answer,sender,phone_id):
    url = f"https://graph.facebook.com/v19.0/{phone_id}/messages"
    headers = {
        'Authorization': f'Bearer {wa_token}',
        'Content-Type': 'application/json'
    }
    type="text"
    body="body"
    content=answer
    image_urls=product_images.image_urls
    if "product_image" in answer:
        for product in image_urls.keys():
            if product in answer:
                answer=answer.replace("product_image",image_urls[product])
                urls=extractor.find_urls(answer)
                if len(urls)>0:
                    mime_type,_=guess_type(urls[0].split("/")[-1])
                    type=mime_type.split("/")[0]
                    body="link"
                    content=urls[0]
                    answer=answer.replace(urls[0],"\n")
                    break
    data = {
        "messaging_product": "whatsapp",
        "to": sender,
        "type": type,
        type: {
            body:content,
            **({"caption":answer} if type!="text" else {})
            },
        }
    response = requests.post(url, headers=headers, json=data)
    if db:
        insert_chat("Bot",answer)
    return response

def remove(*file_paths):
    for file in file_paths:
        if os.path.exists(file):
            os.remove(file)
        else:pass

if db:
    db_url=os.environ.get("DB_URL") # Database URL
    engine=create_engine(db_url)
    Session=sessionmaker(bind=engine)
    Base=declarative_base()
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
            return chats
        except:pass
        finally:
            session.close()

    def delete_old_chats():
        try:
            session = Session()
            cutoff_date = datetime.now() - timedelta(days=14)
            session.query(Chat).filter(Chat.Chat_time < cutoff_date).delete()
            session.commit()
            logging.info("Old chats deleted successfully")
        except:
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
                chats = '\n\n'.join(query)
                send(chats, owner_phone, phone_id)
        except Exception as e:
            logging.error(f"Error creating report: {e}")
        finally:
            session.close()
            
else:pass

def message_handler(data, phone_id):
    global user_states  # Reference the global state dictionary
    
    # Initialize user_states if it doesn't exist
    if 'user_states' not in globals():
        user_states = {}
    
    sender = data["from"]
    
    # Extract message text (handle different message types)
    if data["type"] == "text":
        prompt = data["text"]["body"]
    else:
        prompt = ""  # For non-text messages, we'll process them separately
        

    msg = prompt.lower()
    language_keywords = {
        "english": ["hie", "hi", "hey"],
        "shona": ["mhoro", "mhoroi", "makadini", "hesi"],
        "ndebele": ["sawubona", "unjani", "salibonani"],
        "tonga": ["mwabuka buti", "mwalibizya buti", "kwasiya", "mulibuti"],
        "chinyanja": ["bwanji", "muli bwanji", "mukuli bwanji"],
        "bemba": ["muli shani", "mulishani", "mwashibukeni"],
        "lozi": ["muzuhile", "mutozi", "muzuhile cwani"]
    }

    user_language = "english"  # default
    for lang, keywords in language_keywords.items():
        if any(word in msg for word in keywords):
            user_language = lang
            break

    # ✅ Select correct pregnancy data
    language_map = {
        "english": pregnancy_data.pregnancy_data,
        "shona": pregnancy_data_shona.pregnancy_data_shona,
        "ndebele": pregnancy_data_ndebele.pregnancy_data_ndebele,
        "tonga": pregnancy_data_tonga.pregnancy_data_tonga,
        "chinyanja": pregnancy_data_chinyanja.pregnancy_data_chinyanja,
        "bemba": pregnancy_data_bemba.pregnancy_data_bemba,
        "lozi": pregnancy_data_lozi.pregnancy_data_lozi
    }

    reply_text = language_map.get(user_language, pregnancy_data.pregnancy_data)
    
    
    # Handle incoming messages from customers (non-agent numbers)
    if sender != str(AGENT):
        # Check if customer is already talking to agent
        if user_states.get(sender) == "talking-to-agent":
            # Forward customer message to agent
            send(f"Customer {sender}: {prompt}", AGENT, phone_id)
            return
        
        # Process normal messages (your existing logic)
        if data["type"] == "text":
            if db:
                insert_chat(sender, prompt)
            convo.send_message(prompt)
        else:
            # Handle media messages (your existing PDF/image/audio processing)
            media_url_endpoint = f'https://graph.facebook.com/v19.0/{data[data["type"]]["id"]}/'
            headers = {'Authorization': f'Bearer {wa_token}'}
            media_response = requests.get(media_url_endpoint, headers=headers)
            media_url = media_response.json()["url"]
            media_download_response = requests.get(media_url, headers=headers)
            
            if data["type"] == "audio":
                filename = "/tmp/temp_audio.mp3"
            elif data["type"] == "image":
                filename = "/tmp/temp_image.jpg"
            elif data["type"] == "document":
                doc = fitz.open(stream=media_download_response.content, filetype="pdf")
                for _, page in enumerate(doc):
                    destination = "/tmp/temp_image.jpg"
                    pix = page.get_pixmap()
                    pix.save(destination)
                    file = genai.upload_file(path=destination, display_name="tempfile")
                    response = model.generate_content(["Read this document carefully and explain it in detail", file])
                    answer = response._result.candidates[0].content.parts[0].text
                    convo.send_message(f'''Direct image input has limitations,
                                        so this message is created by an LLM model based on the document sent by the user, 
                                        reply to the customer assuming you saw that document.
                                        (Warn the customer and stop the chat if it is not related to the business): {answer}''')
                    remove(destination)
            else:
                send("This format is not supported by the bot ☹", sender, phone_id)
                return
            
            if data["type"] in ["image", "audio"]:
                with open(filename, "wb") as temp_media:
                    temp_media.write(media_download_response.content)
                file = genai.upload_file(path=filename, display_name="tempfile")
                
                if data["type"] == "image":
                    response = model.generate_content(["What is in this image?", file])
                    answer = response.text
                    convo.send_message(f'''Customer has sent an image,
                                        So here is the LLM's reply based on the image sent by the customer: {answer}\n\n''')
                    urls = extractor.find_urls(convo.last.text)
                    if len(urls) > 0:
                        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
                        response = requests.get(urls[0], headers=headers)
                        img_path = "/tmp/prod_image.jpg"
                        with open(img_path, "wb") as temp_media:
                            temp_media.write(response.content)
                        img = genai.upload_file(path=img_path, display_name="prodfile")
                        response = model.generate_content(["Are the things in both images exactly the same? Explain in detail", img, file])
                        answer = response.text
                        convo.send_message(f'''This is the message from AI after comparing the two images: {answer}''')
                else:
                    response = model.generate_content(["What is the content of this audio file?", file])
                    answer = response.text
                    convo.send_message(f'''Direct media input has limitations,
                                            so this message is created by an LLM model based on the audio sent by the user, 
                                            reply to the customer assuming you heard that audio.
                                            (Warn the customer and stop the chat if it is not related to the business): {answer}''')
                
                remove("/tmp/temp_image.jpg", "/tmp/temp_audio.mp3", "/tmp/prod_image.jpg")
            
            # Clean up uploaded files
            files = genai.list_files()
            for file in files:
                file.delete()
        
        reply = convo.last.text
        
        # Check if customer needs agent
        if any(keyword in prompt.lower() for keyword in ["agent", "human", "representative"]):
            send("Please wait while I connect you to a human agent...", sender, phone_id)
            send(f"New customer request from {sender}. Message: '{prompt}'. Send 'accept' to start chatting.\n\nSend 'exit' when you're done chatting.", AGENT, phone_id)
            user_states[sender] = "waiting-for-agent"
            return
        
        # Send normal bot response
        if "unable_to_solve_query" in reply:
            send(f"Customer {sender} is not satisfied", owner_phone, phone_id)
            reply = reply.replace("unable_to_solve_query", '\n')
            send(reply, sender, phone_id)
        else:
            send(reply, sender, phone_id)
    
    # Handle incoming messages from agent
    elif sender == str(AGENT):
        prompt = data["text"]["body"].lower().strip()
        
        # Agent accepts a customer
        if prompt == "accept":
            # Find first waiting customer
            customer = next((k for k, v in user_states.items() if v == "waiting-for-agent"), None)
            if customer:
                user_states[customer] = "talking-to-agent"
                user_states["current_customer"] = customer
                send("You are now connected to a human agent.", customer, phone_id)
                send(f"You are now chatting with: {customer}", AGENT, phone_id)
                # Forward the original message that triggered agent request
                if db:
                    last_msg = get_chats(customer)[-1][0] if get_chats(customer) else "No previous messages"
                    send(f"Customer's original message: {last_msg}", AGENT, phone_id)
            else:
                send("No customers waiting at the moment.", AGENT, phone_id)
            return
        
        # Agent ends conversation
        elif prompt == "exit":
            customer = user_states.get("current_customer")
            if customer:
                send("You're now chatting with Rudo, our virtual assistant again.", customer, phone_id)
                send("The conversation has been handed back to the bot.", AGENT, phone_id)
                user_states[customer] = "bot"
                user_states.pop("current_customer", None)
            else:
                send("No active customer conversation to exit.", AGENT, phone_id)
            return
        
        # Forward agent message to customer
        customer = user_states.get("current_customer")
        if customer:
            send(data["text"]["body"], customer, phone_id)
        else:
            send("No customer is currently connected to you.", AGENT, phone_id)
    
    # Daily report and cleanup
    if db:
        scheduler.enterabs(report_time.timestamp(), 1, create_report, (phone_id,))
        scheduler.run(blocking=False)
        delete_old_chats()
    
    try:
        convo.send_message(instructions.instructions)
    except ResourceExhausted as e:
        print("Gemini API quota exceeded. Error:", e)
        send("Sorry, we're experiencing high traffic. Please try again later.", sender, phone_id)
        return

    if db:
        scheduler.enterabs(report_time.timestamp(), 1, create_report, (phone_id,))
        scheduler.run(blocking=False)
        delete_old_chats()

    try:
        convo.send_message(instructions.instructions)
    except ResourceExhausted as e:
        print("Gemini API quota exceeded. Error:", e)
        send("Sorry, we're experiencing high traffic. Please try again later.", sender, phone_id)
        return  
        
        
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
            data = request.get_json()["entry"][0]["changes"][0]["value"]["messages"][0]
            phone_id=request.get_json()["entry"][0]["changes"][0]["value"]["metadata"]["phone_number_id"]
            message_handler(data,phone_id)
        except :pass
        return jsonify({"status": "ok"}), 200

    
        
if __name__ == "__main__":
    app.run(debug=True, port=8000)











