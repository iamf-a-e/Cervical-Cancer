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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --------------------------------------------------------------------------------
#  Service Account Authentication (GCP) - Enhanced Debugging
# --------------------------------------------------------------------------------
service_account_b64 = os.environ.get("GCP_SERVICE_ACCOUNT_BASE64")
logging.info(f"GCP_SERVICE_ACCOUNT_BASE64 present: {bool(service_account_b64)}")
logging.info(f"GOOGLE_APPLICATION_CREDENTIALS env: {os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')}")

if service_account_b64:
    sa_path = "/tmp/service-account.json"
    try:
        with open(sa_path, "wb") as f:
            f.write(base64.b64decode(service_account_b64))
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = sa_path
        logging.info(f"✅ Service account JSON written to {sa_path}")
        
        # Verify file was created properly
        if os.path.exists(sa_path):
            file_size = os.path.getsize(sa_path)
            logging.info(f"✅ Service account file exists, size: {file_size} bytes")
            # Read first 100 chars to verify it's valid JSON
            with open(sa_path, "r") as f:
                content_preview = f.read(100)
                logging.info(f"✅ File preview: {content_preview}...")
        else:
            logging.error("❌ Service account file was not created")
            
    except Exception as e:
        logging.error(f"❌ Failed to decode service account JSON: {e}")
        logging.error(traceback.format_exc())
else:
    logging.warning("⚠️ GCP_SERVICE_ACCOUNT_BASE64 not set.")

# --------------------------------------------------------------------------------
#  Environment Variables - Enhanced Debugging
# --------------------------------------------------------------------------------
wa_token = os.environ.get("WA_TOKEN")  # WhatsApp API Key
phone_id = os.environ.get("PHONE_ID")
gen_api = os.environ.get("GEN_API")
owner_phone = os.environ.get("OWNER_PHONE")
model_name = "gemini-2.0-flash"
bot_name = "Rudo"

VERTEX_AI_ENDPOINT_ID = os.environ.get("VERTEX_AI_ENDPOINT_ID")
VERTEX_AI_REGION = os.environ.get("VERTEX_AI_REGION", "us-west4")
VERTEX_AI_PROJECT = os.environ.get("VERTEX_AI_PROJECT")

# Debug environment variables
logging.info("=== Environment Variables Debug ===")
logging.info(f"VERTEX_AI_PROJECT: {VERTEX_AI_PROJECT}")
logging.info(f"VERTEX_AI_ENDPOINT_ID: {VERTEX_AI_ENDPOINT_ID}")
logging.info(f"VERTEX_AI_REGION: {VERTEX_AI_REGION}")
logging.info(f"WA_TOKEN present: {bool(wa_token)}")
logging.info(f"PHONE_ID: {phone_id}")

# --------------------------------------------------------------------------------
#  Vertex AI Client - Enhanced Debugging
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
        logging.info(f"🔧 Vertex AI Base URL: {self.base_url}")
        
        try:
            # Test credentials availability
            self.credentials, self.project = default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
            logging.info(f"✅ GCP Credentials obtained, project: {self.project}")
            
            if not self.credentials.valid:
                logging.info("🔄 Refreshing credentials...")
                self.credentials.refresh(Request())
                logging.info("✅ Credentials refreshed")
            else:
                logging.info("✅ Credentials are valid")
                
        except Exception as e:
            logging.error(f"❌ Failed to obtain GCP credentials: {e}")
            logging.error(traceback.format_exc())
            raise

    def get_auth_header(self):
        if not self.credentials.valid:
            logging.info("🔄 Refreshing token for auth header...")
            self.credentials.refresh(Request())
        token_preview = self.credentials.token[:20] + "..." if self.credentials.token else "None"
        logging.info(f"🔑 Token preview: {token_preview}")
        return {"Authorization": f"Bearer {self.credentials.token}"}

    def predict(self, payload):
        headers = self.get_auth_header()
        headers["Content-Type"] = "application/json"
        
        # Log prediction attempt
        logging.info(f"🚀 Sending prediction request to Vertex AI")
        logging.info(f"📊 Payload keys: {list(payload.keys())}")
        if "instances" in payload and payload["instances"]:
            instance = payload["instances"][0]
            logging.info(f"📸 Image data present: {'content' in instance}")
            if 'content' in instance:
                logging.info(f"📏 Image data length: {len(instance['content'])} chars")
        
        try:
            response = requests.post(self.base_url, json=payload, headers=headers, timeout=60)
            logging.info(f"✅ Vertex AI response status: {response.status_code}")
            response.raise_for_status()
            result = response.json()
            logging.info(f"📈 Prediction response keys: {list(result.keys())}")
            return result
        except requests.exceptions.RequestException as e:
            logging.error(f"❌ Vertex AI API request failed: {e}")
            if hasattr(e, 'response') and e.response is not None:
                logging.error(f"❌ Response text: {e.response.text}")
            raise

vertex_ai_client = None
vertex_ai_initialized = False

if VERTEX_AI_PROJECT and VERTEX_AI_ENDPOINT_ID:
    try:
        logging.info("🔄 Initializing Vertex AI client...")
        vertex_ai_client = VertexAIClient(VERTEX_AI_PROJECT, VERTEX_AI_ENDPOINT_ID, VERTEX_AI_REGION)
        vertex_ai_initialized = True
        logging.info("✅ Vertex AI client initialized successfully")
        
        # Test with a simple payload to verify connectivity
        try:
            test_payload = {"instances": [{"content": "test"}]}
            # Don't actually send, just test URL construction
            logging.info(f"🧪 Vertex AI client test - URL constructed correctly")
        except Exception as test_e:
            logging.error(f"❌ Vertex AI client test failed: {test_e}")
            
    except Exception as e:
        logging.error(f"❌ Vertex AI init failed: {e}")
        logging.error(traceback.format_exc())
        vertex_ai_client = None
else:
    logging.error("❌ Missing Vertex AI environment variables:")
    if not VERTEX_AI_PROJECT:
        logging.error("   - VERTEX_AI_PROJECT is not set")
    if not VERTEX_AI_ENDPOINT_ID:
        logging.error("   - VERTEX_AI_ENDPOINT_ID is not set")

# --------------------------------------------------------------------------------
#  Cervical Cancer Staging Function - Enhanced Debugging
# --------------------------------------------------------------------------------
def stage_cervical_cancer(image_path):
    logging.info(f"🔍 Starting cervical cancer staging for image: {image_path}")
    
    # Enhanced Vertex AI client check
    if not vertex_ai_client:
        logging.error("❌ Vertex AI client is None - configuration failed")
        logging.error("   Possible reasons:")
        logging.error("   - Missing VERTEX_AI_PROJECT environment variable")
        logging.error("   - Missing VERTEX_AI_ENDPOINT_ID environment variable") 
        logging.error("   - GCP credentials not properly configured")
        logging.error("   - Vertex AI client initialization exception")
        return {
            "stage": "Error", 
            "confidence": 0, 
            "success": False, 
            "error": "Vertex AI not configured",
            "debug_info": {
                "vertex_ai_project_set": bool(VERTEX_AI_PROJECT),
                "vertex_ai_endpoint_set": bool(VERTEX_AI_ENDPOINT_ID),
                "vertex_ai_initialized": vertex_ai_initialized,
                "service_account_configured": bool(service_account_b64)
            }
        }
    
    try:
        # Verify image file exists and is readable
        if not os.path.exists(image_path):
            logging.error(f"❌ Image file not found: {image_path}")
            return {"stage": "Error", "confidence": 0, "success": False, "error": "Image file not found"}
        
        file_size = os.path.getsize(image_path)
        logging.info(f"📸 Image file size: {file_size} bytes")
        
        if file_size == 0:
            logging.error("❌ Image file is empty")
            return {"stage": "Error", "confidence": 0, "success": False, "error": "Image file is empty"}
        
        with open(image_path, "rb") as f:
            image_data = f.read()
            image_b64 = base64.b64encode(image_data).decode("utf-8")
        
        logging.info(f"📊 Base64 encoded image length: {len(image_b64)} characters")
        
        payload = {"instances": [{"content": image_b64}]}
        logging.info("🚀 Sending request to Vertex AI endpoint...")
        
        result = vertex_ai_client.predict(payload)
        logging.info(f"✅ Received response from Vertex AI")
        
        if "predictions" in result and result["predictions"]:
            prediction = result["predictions"][0]
            logging.info(f"📈 Prediction keys: {list(prediction.keys())}")
            
            if "displayNames" in prediction and "confidences" in prediction:
                labels = prediction["displayNames"]
                scores = prediction["confidences"]
                max_idx = scores.index(max(scores))
                
                logging.info(f"🏷️ Available labels: {labels}")
                logging.info(f"📊 Confidence scores: {scores}")
                logging.info(f"🎯 Selected label: {labels[max_idx]} with confidence: {scores[max_idx]}")
                
                return {
                    "stage": labels[max_idx],
                    "confidence": float(scores[max_idx]),
                    "success": True,
                    "response_type": "classification"
                }
            else:
                logging.error("❌ Unexpected prediction format - missing displayNames or confidences")
                logging.error(f"Prediction structure: {prediction}")
        else:
            logging.error("❌ No predictions in response")
            logging.error(f"Response structure: {result}")
            
        return {"stage": "Error", "confidence": 0, "success": False, "error": "Unexpected response format"}
        
    except Exception as e:
        logging.error(f"❌ Error during cervical cancer staging: {e}")
        logging.error(traceback.format_exc())
        return {
            "stage": "Error", 
            "confidence": 0, 
            "success": False, 
            "error": str(e),
            "traceback": traceback.format_exc()
        }

# --------------------------------------------------------------------------------
#  Enhanced Health Check Endpoint
# --------------------------------------------------------------------------------
@app.route('/health', methods=['GET'])
def health_check():
    vertex_status = "healthy" if vertex_ai_client else "unhealthy"
    vertex_details = {
        "project_set": bool(VERTEX_AI_PROJECT),
        "endpoint_set": bool(VERTEX_AI_ENDPOINT_ID),
        "client_initialized": vertex_ai_initialized,
        "client_exists": vertex_ai_client is not None
    }
    
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "vertex_ai": {
            "status": vertex_status,
            "details": vertex_details,
            "project": VERTEX_AI_PROJECT,
            "endpoint_id": VERTEX_AI_ENDPOINT_ID,
            "region": VERTEX_AI_REGION
        },
        "whatsapp": {
            "token_set": wa_token is not None,
            "phone_id_set": phone_id is not None
        },
        "service_account": {
            "configured": bool(service_account_b64),
            "path": os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        },
        "user_states": len(user_states),
        "environment_ready": all([VERTEX_AI_PROJECT, VERTEX_AI_ENDPOINT_ID, wa_token, phone_id])
    })



if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
