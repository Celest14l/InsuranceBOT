# filename: app.py
import os
import time
import json
import speech_recognition as sr
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnableSequence
from gtts import gTTS 
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_socketio import SocketIO
from flask_cors import CORS
import faiss
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import PyPDF2
from pydub import AudioSegment
import pickle
import re # Import regular expressions for sentence splitting

# --- Configuration & Setup ---
print("🚀 Starting ValuEnable InsureBot Server...")

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    print("❌ Groq API Key missing. Please create a .env file with GROQ_API_KEY=<your_key>. Exiting.")
    exit(1)

# Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
AUDIO_DIR = os.path.join(BASE_DIR, "responses_output")
DATA_DIR = os.path.join(BASE_DIR, "hackathon_data")
RECORDINGS_DIR = os.path.join(DATA_DIR, "recordings")
UPLOAD_DIR = os.path.join(BASE_DIR, "user_uploads")
CACHE_DIR = os.path.join(BASE_DIR, "cache")
ERROR_LOG_FILE = os.path.join(BASE_DIR, "error_log.txt")

os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RECORDINGS_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# --- Utility & Core Logic Functions (Moved to top) ---
def log_error(error_msg):
    with open(ERROR_LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}: {error_msg}\n")

# --- Flask App Initialization ---
app = Flask(__name__, static_folder=BASE_DIR, static_url_path='')
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# --- Global In-Memory Stores & Models ---
memory = ConversationBufferWindowMemory(k=10, memory_key="chat_history", return_messages=True)
conversation = None
rag_index = None
rag_texts = []
response_counter = 1
conversation_stage = "GREETING" 

# --- Load Models (Once on startup) ---
print("🧠 Loading AI models...")
try:
    groq_chat = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="llama3-8b-8192")
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    print("✅ All models loaded successfully.")
except Exception as e:
    log_error(f"🔥 Critical error during model loading: {e}")
    exit(1)

# --- RAG Setup & Other Functions ---

def extract_text_from_pdf(file_path):
    text = ""
    if not os.path.exists(file_path):
        log_error(f"PDF file not found at {file_path}")
        return ""
    try:
        with open(file_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        log_error(f"Error reading PDF '{file_path}': {e}")
    return text

def transcribe_and_integrate_recordings():
    transcripts = []
    recognizer = sr.Recognizer()
    if not os.path.exists(RECORDINGS_DIR):
        print("⚠️ Recordings directory not found. Skipping transcription.")
        return transcripts
    print(f"🎤 Transcribing audio from {RECORDINGS_DIR}...")
    for filename in os.listdir(RECORDINGS_DIR):
        if filename.lower().endswith(".wav"):
            file_path = os.path.join(RECORDINGS_DIR, filename)
            try:
                with sr.AudioFile(file_path) as source:
                    audio = recognizer.record(source)
                    text = recognizer.recognize_google(audio, language="en-IN")
                    transcripts.append(f"Transcript from {filename}: {text}")
                    print(f"   - Transcribed {filename}")
            except Exception as e:
                log_error(f"An error occurred during transcription of {filename}: {e}")
    return transcripts

def setup_rag():
    global rag_index, rag_texts
    index_cache_path = os.path.join(CACHE_DIR, "faiss_index.bin")
    texts_cache_path = os.path.join(CACHE_DIR, "rag_texts.pkl")

    if os.path.exists(index_cache_path) and os.path.exists(texts_cache_path):
        print("✅ Found RAG cache. Loading from disk...")
        rag_index = faiss.read_index(index_cache_path)
        with open(texts_cache_path, "rb") as f:
            rag_texts = pickle.load(f)
        print("✅ RAG index loaded from cache successfully.")
        return

    print("🛠️ No cache found. Building RAG index from scratch...")
    kb_path = os.path.join(DATA_DIR, "Knowledge Base.txt")
    script_path = os.path.join(DATA_DIR, "Calling Script.pdf")
    try:
        if os.path.exists(kb_path):
            with open(kb_path, 'r', encoding='utf-8') as f:
                kb_content = f.read()
            rag_texts.extend([section.strip() for section in kb_content.split('\n\n') if section.strip()])
        script_content = extract_text_from_pdf(script_path)
        if script_content:
            rag_texts.extend([section.strip() for section in script_content.split('\n\n') if section.strip()])
        recording_transcripts = transcribe_and_integrate_recordings()
        rag_texts.extend(recording_transcripts)
        if not rag_texts:
            log_error("No text found in any provided documents to build RAG index.")
            return
        print(f"🧠 Building vector index from {len(rag_texts)} total text chunks...")
        embeddings = embedder.encode(rag_texts, show_progress_bar=True)
        dimension = embeddings.shape[1]
        rag_index = faiss.IndexFlatL2(dimension)
        rag_index.add(np.array(embeddings).astype('float32'))
        
        print("💾 Saving RAG index to cache...")
        faiss.write_index(rag_index, index_cache_path)
        with open(texts_cache_path, "wb") as f:
            pickle.dump(rag_texts, f)
        print("✅ RAG index built and cached successfully.")

    except Exception as e:
        log_error(f"Failed to build RAG index: {e}")

def retrieve_rag_context(query, k=4):
    if not rag_index or not rag_texts: return ""
    try:
        query_embedding = embedder.encode([query])
        _, indices = rag_index.search(np.array(query_embedding).astype('float32'), k)
        context = "\n---\n".join([rag_texts[i] for i in indices[0] if i < len(rag_texts)])
        return context
    except Exception as e:
        log_error(f"Error during RAG retrieval: {e}")
        return ""

def generate_bot_response(user_input):
    global conversation, conversation_stage
    
    # **** NEW: More Robust State Transition Logic ****
    if conversation_stage == "GREETING":
        conversation_stage = "POLICY_CONFIRMATION"

    stage_instructions = {
        "GREETING": "Your first and only task is to greet the user and confirm you are speaking to the correct person (Branch 1.0 of the script). Start with 'Hello'.",
        "POLICY_CONFIRMATION": "You have already greeted the user. Your task now is to confirm their policy details and ask why the premium is pending (Branch 2.0 of the script). DO NOT greet them again.",
        "HANDLING_OBJECTION": "The user has an issue or question. Use the Knowledge Base and call transcripts to provide a helpful, empathetic rebuttal. Your goal is to convince them to pay.",
        "PAYMENT_FOLLOWUP": "The user has agreed to pay. Your goal is to ask how they plan to make the payment and guide them through the process, as per Branch 5.0 of the Calling Script.",
        "CLOSING": "The conversation is ending. Your goal is to provide the helpline information and give a polite closing statement, as per Branch 9.0 of the Calling Script."
    }
    
    current_instruction = stage_instructions.get(conversation_stage, stage_instructions["HANDLING_OBJECTION"])

    system_prompt = (
        "You are 'Veena,' a professional insurance agent for 'ValuEnable'. Your behavior is strictly governed by these rules: "
        "1. **Follow the Current Goal:** Your immediate task is defined by 'Current Conversation Goal'. You MUST perform this task. "
        f"**Current Conversation Goal:** {current_instruction} "
        "2. **Check History:** Before you speak, review the 'chat_history'. DO NOT repeat information you have already given or ask questions you have already asked. "
        "3. **No Extra Salutations:** Greet the user ONCE at the very beginning of the call. DO NOT use words like 'Namaste' or 'Good evening' in subsequent turns. "
        "4. **Be Concise:** Keep responses under 35 words. Break up longer explanations into multiple short sentences. "
        "5. **Use Provided Knowledge:** Base your answers ONLY on the 'Relevant Information from documents'. "
        "6. **Guide the Conversation:** Always try to end your turn with a question to move the conversation to the next logical step."
    )
    prompt_template = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("Relevant Information from documents:\n{rag_context}\n\nCustomer says: {human_input}"),
    ])
    conversation = RunnableSequence(prompt_template | groq_chat)

    rag_context = retrieve_rag_context(user_input)
    
    try:
        inputs = { "human_input": user_input, "rag_context": rag_context, "chat_history": memory.load_memory_variables({})["chat_history"] }
        ai_message = conversation.invoke(inputs)
        response_text = ai_message.content.strip()
        memory.save_context({"human_input": user_input}, {"output": response_text})
        
        # Update state based on keywords in the bot's response
        if "confirm your policy details" in response_text.lower() or "premium of" in response_text.lower():
             conversation_stage = "HANDLING_OBJECTION"
        elif "how you plan to make the payment" in response_text.lower():
             conversation_stage = "PAYMENT_FOLLOWUP"
        elif "helpline" in response_text.lower() or "thank you for your valuable time" in response_text.lower():
             conversation_stage = "CLOSING"

        return response_text
    except Exception as e:
        log_error(f"LLM conversation error: {e}")
        return "I'm sorry, I'm having a little trouble right now. Could you please try again?"

def text_to_speech(text):
    global response_counter
    file_name = f"response_{response_counter}.wav"
    output_path = os.path.join(AUDIO_DIR, file_name)
    response_counter += 1
    
    try:
        sentences = re.split(r'(?<=[.?!])\s+', text)
        pause = AudioSegment.silent(duration=250)
        combined_audio = AudioSegment.empty()
        temp_files = []

        for i, sentence in enumerate(sentences):
            if not sentence: continue
            
            tts = gTTS(text=sentence, lang='en', tld='co.in')
            temp_mp3_path = os.path.join(AUDIO_DIR, f"temp_{i}.mp3")
            tts.save(temp_mp3_path)
            temp_files.append(temp_mp3_path)
            
            audio_segment = AudioSegment.from_mp3(temp_mp3_path)
            combined_audio += audio_segment + pause

        combined_audio.export(output_path, format="wav")
        
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)
            
        return file_name
    except Exception as e:
        log_error(f"gTTS dynamic speech generation error: {e}")
        return None

# --- API Endpoints ---

@app.route('/')
def serve_index():
    return send_from_directory(BASE_DIR, 'index.html')

@app.route('/talk', methods=['POST'])
def talk():
    t_start = time.time()
    
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    audio_file = request.files['audio']
    original_audio_path = os.path.join(UPLOAD_DIR, f"user_input_original_{int(t_start)}")
    audio_file.save(original_audio_path)

    converted_wav_path = os.path.join(UPLOAD_DIR, f"user_input_converted_{int(t_start)}.wav")
    try:
        audio = AudioSegment.from_file(original_audio_path)
        audio.export(converted_wav_path, format="wav")
    except Exception as e:
        log_error(f"Audio conversion failed: {e}")
        return jsonify({'error': 'Failed to process audio file.'}), 500
    
    t_after_conversion = time.time()

    recognizer = sr.Recognizer()
    user_input_text = ""
    try:
        with sr.AudioFile(converted_wav_path) as source:
            audio_data = recognizer.record(source)
            user_input_text = recognizer.recognize_google(audio_data, language="en-IN")
    except Exception as e:
        log_error(f"STT Error: {e}")
        return jsonify({'error': 'Could not understand audio'}), 400
    finally:
        if os.path.exists(original_audio_path): os.remove(original_audio_path)
        if os.path.exists(converted_wav_path): os.remove(converted_wav_path)

    t_after_stt = time.time()
    print(f"🎤 User said: {user_input_text}")
    
    bot_response_text = generate_bot_response(user_input_text)
    t_after_llm = time.time()
    print(f"🤖 Bot responds: {bot_response_text}")

    bot_audio_filename = text_to_speech(bot_response_text)
    t_after_tts = time.time()
    
    audio_url = f"{request.host_url}audio/{bot_audio_filename}" if bot_audio_filename else None
    
    print("\n--- LATENCY BREAKDOWN ---")
    print(f"Audio Conversion: {t_after_conversion - t_start:.2f}s")
    print(f"Speech-to-Text:   {t_after_stt - t_after_conversion:.2f}s")
    print(f"LLM Inference:      {t_after_llm - t_after_stt:.2f}s")
    print(f"Text-to-Speech:   {t_after_tts - t_after_llm:.2f}s")
    print("-------------------------")
    total_latency = t_after_tts - t_start
    print(f"⏱️ Total Latency:    {total_latency:.2f}s")
    print("-------------------------\n")
    
    return jsonify({ 'user_transcript': user_input_text, 'response_text': bot_response_text, 'audio_url': audio_url, 'latency': total_latency })

@app.route('/audio/<filename>')
def serve_audio(filename):
    return send_from_directory(AUDIO_DIR, filename)

@socketio.on('connect')
def handle_connect(): print('✅ Client connected')

@socketio.on('disconnect')
def handle_disconnect(): print('❌ Client disconnected')

# --- Main Execution ---
if __name__ == "__main__":
    setup_rag()
    print("🌐 Starting Flask-SocketIO server on http://127.0.0.1:5001")
    print("🚀 Open your browser and go to http://127.0.0.1:5001 to use the application.")
    socketio.run(app, host='0.0.0.0', port=5001, debug=True, use_reloader=False)
