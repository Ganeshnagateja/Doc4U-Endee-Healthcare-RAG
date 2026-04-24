import os
import warnings

# Suppress FutureWarning from google.api_core before importing Google libraries.
warnings.filterwarnings('ignore', category=FutureWarning)

import jwt
import bcrypt
import pdfplumber
import base64
import io
import feedparser
import requests
from functools import wraps
from datetime import datetime, timedelta
from dotenv import load_dotenv
from flask import Flask, request, jsonify, send_from_directory
from flask_pymongo import PyMongo
from flask_cors import CORS
from bson import ObjectId
import google.generativeai as genai
from endee_rag import rag_service

# --- 1. Initialization ---
load_dotenv()
app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app) 

# --- 2. Configuration ---
app.config["MONGO_URI"] = os.getenv("MONGO_URI")
app.config["JWT_SECRET_KEY"] = os.getenv("JWT_SECRET_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not app.config["MONGO_URI"] or not app.config["JWT_SECRET_KEY"] or not GOOGLE_API_KEY:
    raise EnvironmentError("Error: MONGO_URI, JWT_SECRET_KEY, or GOOGLE_API_KEY is not set. Check your .env file.")

# --- 3. Database & AI Setup ---
try:
    mongo = PyMongo(app)
    users_collection = mongo.db.users
    sessions_collection = mongo.db.chat_sessions
    print("[OK] MongoDB initialized successfully")
except Exception as e:
    print(f"[WARN] MongoDB connection failed: {e}")
    print("  Running in offline mode. Features requiring database will not work.")
    mongo = None
    users_collection = None
    sessions_collection = None

genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-2.5-flash') 


# --- 4. Security & Helper Functions ---

def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        if 'Authorization' in request.headers:
            auth_header = request.headers['Authorization'].strip()
            parts = auth_header.split(" ", 1)
            if len(parts) == 2 and parts[0].lower() == 'bearer':
                token = parts[1]
        if not token:
            return jsonify({'message': 'Token is missing!'}), 401
        if users_collection is None:
            return jsonify({'error': 'Database is not available. Please try again later.'}), 503
        try:
            data = jwt.decode(token, app.config['JWT_SECRET_KEY'], algorithms=["HS256"])
            current_user = users_collection.find_one({"_id": ObjectId(data['user_id'])})
            if not current_user:
                return jsonify({'message': 'User not found.'}), 401
            kwargs['current_user'] = current_user
        except jwt.ExpiredSignatureError:
            return jsonify({'message': 'Token has expired!'}), 401
        except jwt.InvalidTokenError:
            return jsonify({'message': 'Token is invalid!'}), 401
        return f(*args, **kwargs)
    return decorated

def extract_pdf_text(pdf_file_bytes):
    try:
        with pdfplumber.open(io.BytesIO(pdf_file_bytes)) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text() or ""
            return text
    except Exception as e:
        print(f"Error extracting PDF text: {e}")
        return None

# --- 5. Authentication Endpoints (Unchanged) ---

@app.route('/signup', methods=['POST'])
def signup():
    if users_collection is None:
        return jsonify({'error': 'Database is not available. Please try again later.'}), 503
    
    data = request.get_json()
    name = data.get('name')
    email = data.get('email')
    password = data.get('password')
    if not name or not email or not password:
        return jsonify({'message': 'Missing fields.'}), 400
    if users_collection.find_one({'email': email}):
        return jsonify({'message': 'Email already in use.'}), 400
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    users_collection.insert_one({
        'name': name,
        'email': email,
        'password': hashed_password
    })
    return jsonify({'message': 'User created successfully!'}), 201

@app.route('/login', methods=['POST'])
def login():
    if users_collection is None:
        return jsonify({'error': 'Database is not available. Please try again later.'}), 503
    
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')
    if not email or not password:
        return jsonify({'message': 'Missing email or password.'}), 400
    user = users_collection.find_one({'email': email})
    if not user:
        return jsonify({'message': 'Invalid email or password.'}), 400
    if bcrypt.checkpw(password.encode('utf-8'), user['password']):
        token = jwt.encode({
            'user_id': str(user['_id']),
            'exp': datetime.utcnow() + timedelta(hours=24)
        }, app.config['JWT_SECRET_KEY'], algorithm="HS256")
        return jsonify({
            'message': 'Login successful!',
            'token': token,
            'name': user['name']
        }), 200
    else:
        return jsonify({'message': 'Invalid email or password.'}), 400

# --- 6. Chatbot Endpoint (MODIFIED) ---

@app.route('/ask_bot', methods=['POST'])
@token_required 
def ask_bot(current_user):
    if sessions_collection is None:
        return jsonify({'error': 'Database is not available. Please try again later.'}), 503
    
    user_id = current_user['_id']
    message = request.form.get('message', '')
    file = request.files.get('file')

    # 1. Retrieve full history from DB (will be used for saving)
    chat_session = sessions_collection.find_one({'user_id': user_id})
    history_to_save = chat_session['history'] if chat_session else []

    # --- THIS IS THE FIX ---
    # Create a "clean" history just for the AI, removing our custom 'file' key
    history_for_api = []
    for item in history_to_save:
        history_for_api.append({
            'role': item['role'],
            'parts': item['parts']
        })
    # --- END OF FIX ---

    # 2. Prepare content for AI and for Storage
    user_content_parts = [] # For sending to the AI
    file_doc_for_db = None  # For saving to MongoDB

    attached_filename = None
    if file and file.filename:
        attached_filename = file.filename
        file_bytes = file.read()
        try:
            base64_data = base64.b64encode(file_bytes).decode('utf-8')
            file_doc_for_db = {
                "filename": file.filename,
                "mime_type": file.mimetype,
                "data": base64_data
            }
        except Exception as e:
            print(f"Error encoding file to Base64: {e}")
            file_doc_for_db = None

        if file.mimetype == 'application/pdf':
            pdf_text = extract_pdf_text(file_bytes)
            if pdf_text:
                user_content_parts.append(f"Context from attached PDF: {pdf_text}")
        
        elif file.mimetype in ['image/jpeg', 'image/png', 'image/gif']:
            image_part = {
                'mime_type': file.mimetype,
                'data': file_bytes
            }
            user_content_parts.append(image_part)
        
        file.close()

    # Index uploaded PDF text into Endee so future turns can retrieve it semantically.
    if file and attached_filename and file.mimetype == 'application/pdf':
        try:
            if 'pdf_text' in locals() and pdf_text:
                rag_service.ingest_document(
                    pdf_text,
                    title=attached_filename,
                    source=f"Uploaded PDF: {attached_filename}",
                    user_id=str(user_id),
                    doc_type="user_upload"
                )
        except Exception as e:
            print(f"Endee ingestion warning: {e}")

    retrieved_contexts = []
    if message:
        retrieved_contexts = rag_service.search(message, user_id=str(user_id), k=4)
        if retrieved_contexts:
            rag_context = "\n\n".join([
                f"Source {idx+1}: {ctx.title} ({ctx.source})\n{ctx.text}"
                for idx, ctx in enumerate(retrieved_contexts)
            ])
            safety_prompt = (
                "You are Doc4U, a healthcare information assistant. Use the retrieved context below "
                "when it is relevant. Do not claim to diagnose. Mention emergency red flags and advise "
                "professional medical help when needed. Cite source names naturally when using retrieved facts.\n\n"
                f"Retrieved Endee context:\n{rag_context}\n\nUser question: {message}"
            )
            user_content_parts.append(safety_prompt)
        else:
            user_content_parts.append(message)

    if not user_content_parts:
         return jsonify({'reply': "I received an empty message. Please try again."})

    # 3. Call the Google AI API
    try:
        # Pass the CLEAN history (history_for_api) to start_chat
        chat = model.start_chat(history=history_for_api)
        response = chat.send_message(user_content_parts)
        bot_reply = response.text
    except Exception as e:
        print(f"Google AI Error: {e}") 
        return jsonify({'error': f'Error calling AI model: {str(e)}'}), 500

    # 4. Update and save history
    user_text = message or f"[User attached file: {attached_filename}]"
    user_history_entry = {
        'role': 'user',
        'parts': [{'text': user_text}]
    }
    
    if file_doc_for_db:
        user_history_entry['file'] = file_doc_for_db # Add file to the user's message

    citations = [
        {
            'title': ctx.title,
            'uri': ctx.source,
            'score': ctx.score
        }
        for ctx in retrieved_contexts
    ]

    model_history_entry = {
        'role': 'model',
        'parts': [{'text': bot_reply}],
        'citations': citations
    }
    
    # Append new entries to the FULL history list
    history_to_save.append(user_history_entry)
    history_to_save.append(model_history_entry)

    try:
        # Save the FULL history (history_to_save) back to the DB
        sessions_collection.update_one(
            {'user_id': user_id},
            {'$set': {'history': history_to_save, 'last_updated': datetime.utcnow()}},
            upsert=True
        )
    except Exception as e:
        print(f"MongoDB Error: {e}")
        if "BSON document too large" in str(e):
            return jsonify({'error': 'File is too large to save. The 16MB limit was exceeded. Your message was not saved.'}), 500
        else:
            return jsonify({'error': 'A database error occurred.'}), 500
    
    return jsonify({'reply': bot_reply, 'citations': citations, 'endee_status': rag_service.status()})


# --- 7. Chat History Endpoints (Unchanged) ---
@app.route('/get_history', methods=['GET'])
@token_required
def get_history(current_user):
    if sessions_collection is None:
        return jsonify({'error': 'Database is not available. Please try again later.'}), 503
    
    user_id = current_user['_id']
    chat_session = sessions_collection.find_one({'user_id': user_id})
    history = chat_session['history'] if chat_session else []
    return jsonify({'history': history}), 200

@app.route('/clear_history', methods=['POST'])
@token_required
def clear_history(current_user):
    if sessions_collection is None:
        return jsonify({'error': 'Database is not available. Please try again later.'}), 503
    
    user_id = current_user['_id']
    sessions_collection.delete_one({'user_id': user_id})
    return jsonify({'message': 'History cleared successfully'}), 200


# --- 8. Static File Serving (Unchanged) ---
@app.route('/')
def serve_login():
    return send_from_directory('.', 'login.html')

@app.route('/chatbot.html')
def serve_chatbot(): 
    return send_from_directory('.', 'chatbot.html')

@app.route('/favicon.ico')
def favicon():
    """Return a simple 1x1 transparent PNG to suppress 404 errors"""
    import base64
    # 1x1 transparent PNG
    png_data = base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==')
    return png_data, 200, {'Content-Type': 'image/png'}


# ---------------------------------------------------------
# ENDEE VECTOR DATABASE / RAG ENDPOINTS
# ---------------------------------------------------------
@app.route('/api/endee/status', methods=['GET'])
def endee_status():
    return jsonify(rag_service.status()), 200

@app.route('/api/rag/search', methods=['GET'])
@token_required
def rag_search(current_user):
    query = request.args.get('q', '').strip()
    top_k = int(request.args.get('top_k', 4))
    if not query:
        return jsonify({'error': 'Missing search query. Use /api/rag/search?q=your question'}), 400
    results = rag_service.search(query, user_id=str(current_user['_id']), k=top_k)
    return jsonify({
        'query': query,
        'results': [
            {
                'title': r.title,
                'source': r.source,
                'score': r.score,
                'text': r.text[:600],
                'metadata': r.metadata or {}
            }
            for r in results
        ],
        'endee_status': rag_service.status()
    }), 200

@app.route('/api/rag/ingest-text', methods=['POST'])
@token_required
def rag_ingest_text(current_user):
    data = request.get_json() or {}
    title = data.get('title', 'User note')
    text = data.get('text', '')
    if not text.strip():
        return jsonify({'error': 'Text is required for ingestion.'}), 400
    chunks = rag_service.ingest_document(
        text,
        title=title,
        source=data.get('source', f'Manual note: {title}'),
        user_id=str(current_user['_id']),
        doc_type='manual_note'
    )
    return jsonify({'message': 'Text ingested into Endee retrieval layer.', 'chunks_indexed': chunks, 'endee_status': rag_service.status()}), 201

# ---------------------------------------------------------
# WHO VACCINATION DATA
# ---------------------------------------------------------
@app.route("/api/vaccine-data/<country_code>", methods=["GET"])
def vaccine_data(country_code):
    try:
        url = "https://ghoapi.azureedge.net/api/WHOSIS_000001"
        res = requests.get(url)
        data = res.json().get('value', [])
        #print("WHO API Response:", data[:3])   # show first 3 records
        country_code = country_code.upper()

        records = [r for r in data if r.get("SpatialDim") == country_code]

        if records:
            latest = max(records, key=lambda x: x.get("TimeDim", 0))

            value = latest.get("NumericValue")
            year = latest.get("TimeDim")

            return jsonify({
                "country": country_code,
                "indicator": "Vaccination Coverage (WHOSIS_000001)",
                "year": year,
                "coverage_percent": value,
                "message": f"In {year}, vaccination coverage for {country_code} was {value}%. (WHO data)"
            })
        return jsonify({"error": f"No vaccination data found for {country_code}."})

    except Exception as e:
        print("[ERROR] Error fetching WHO vaccination data:", e)
        return jsonify({"error": "Failed to fetch vaccination data.", "details": str(e)}), 500


# ---------------------------------------------------------
# MOCK VACCINE REMINDER
# ---------------------------------------------------------
@app.route("/api/vaccine-reminder/<name>/<int:age>", methods=["GET"])
def vaccine_reminder(name, age):
    try:
        today = datetime.now()
        if age < 1:
            vaccine_type = "DTP1"
            next_date = today + timedelta(days=30)
        elif age < 5:
            vaccine_type = "Polio Booster"
            next_date = today + timedelta(days=90)
        else:
            vaccine_type = "Flu Vaccine"
            next_date = today + timedelta(days=365)

        return jsonify({
            "name": name,
            "recommended_vaccine": vaccine_type,
            "reminder_date": next_date.strftime("%Y-%m-%d"),
            "message": f"Hi {name}, your next {vaccine_type} shot is due on {next_date.strftime('%Y-%m-%d')} 💉"
        })

    except Exception as e:
        print("[ERROR] Error generating reminder:", e)
        return jsonify({"error": "Failed to generate reminder.", "details": str(e)}), 500


# ---------------------------------------------------------
# HEALTH ALERTS
# ---------------------------------------------------------
@app.route("/api/health-alerts", methods=["GET"])
def health_alerts():
    try:
        rss_url = "https://www.who.int/feeds/entity/csr/don/en/rss.xml"
        feed = feedparser.parse(rss_url)

        alerts = []
        for entry in feed.entries[:5]:
            alerts.append({
                "title": entry.title,
                "link": entry.link,
                "published": entry.published
            })

        return jsonify({
            "source": "WHO Disease Outbreak News",
            "alerts": alerts
        })

    except Exception as e:
        print("[ERROR] Error fetching WHO alerts:", e)
        return jsonify({"error": "Failed to fetch WHO alerts.", "details": str(e)}), 500




# --- 9. Run the App (Unchanged) ---
if __name__ == '__main__':
    app.run(debug=True, port=3000)
