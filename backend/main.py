from fastapi import FastAPI, Request, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, HTMLResponse, StreamingResponse
from pydantic import BaseModel
import sys
import os
from pathlib import Path
import time

# Add the parent directory to Python path
sys.path.append(str(Path(__file__).parent.parent))

from api.llm import LLMHandler
from api.pdf_processor import PDFProcessor
from api.rag import RAGSystem
from api.tts import TextToSpeech
from api.stt import SpeechToText
import logging
import uvicorn
import json
import base64

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(title="RAG API", description="RAG System API for PDF Processing and Querying")

# Store server start timestamp
SERVER_START_TIME = str(int(time.time()))

# Create necessary directories
DATA_DIR = Path(__file__).parent / "data"
RAW_PDFS_DIR = DATA_DIR / "raw_pdfs"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"

# Create directories if they don't exist
RAW_PDFS_DIR.mkdir(parents=True, exist_ok=True)
EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

# Initialize components
llm_handler = LLMHandler()
pdf_processor = PDFProcessor()
rag_system = RAGSystem(pdf_processor, llm_handler)
tts = TextToSpeech()
stt = SpeechToText()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files

@app.get("/")
async def read_root():
    return HTMLResponse(content=open("frontend/src/index.html").read())

@app.get("/server_timestamp")
async def get_server_timestamp():
    """Return server start timestamp for client cache management."""
    return JSONResponse({"timestamp": SERVER_START_TIME})

@app.post("/api/chat")
async def api_chat(request: Request):
    """Main chat endpoint that the frontend expects."""
    try:
        data = await request.json()
        text = data.get("text", "")
        history = data.get("history", [])
        is_speech = data.get("is_speech", False)

        if not text:
            raise HTTPException(status_code=400, detail="No text provided")

        # Generate response using RAG system
        response = rag_system.generate_response(text, history)

        return JSONResponse(response)
    except Exception as e:
        logger.error(f"Error in chat: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
async def chat(request: Request):
    """Legacy chat endpoint for backwards compatibility."""
    return await api_chat(request)

@app.post("/api/stream_audio")
async def api_stream_audio(request: Request):
    """Stream audio endpoint that the frontend expects."""
    try:
        data = await request.json()
        text = data.get("text", "")
        history = data.get("history", [])

        if not text:
            raise HTTPException(status_code=400, detail="No text provided")

        # Generate audio using TTS
        audio_path = tts.text_to_speech(text)

        # Read the audio file and convert to base64
        with open(audio_path, "rb") as audio_file:
            audio_data = audio_file.read()
            audio_base64 = base64.b64encode(audio_data).decode()

        # Clean up the temporary file
        tts.cleanup(audio_path)

        return JSONResponse({
            "status": "success",
            "audio": audio_base64
        })
    except Exception as e:
        logger.error(f"Audio streaming error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to generate audio: {str(e)}")

@app.post("/stream_audio")
async def stream_audio(request: Request):
    """Legacy stream audio endpoint for backwards compatibility."""
    return await api_stream_audio(request)

@app.post("/sheets")
async def handle_sheets(request: Request):
    """Handle sheets mode requests."""
    try:
        data = await request.json()
        text = data.get("text", "")
        history = data.get("history", [])

        if not text:
            raise HTTPException(status_code=400, detail="No text provided")

        # For now, return a simple success response with the text formatted as a table
        # This can be extended to integrate with actual Google Sheets API
        response_text = f"""### Sheets Mode Response

| Input | Response |
|:------|:---------|
| {text} | This feature will be implemented to connect with Google Sheets |

The sheets functionality is ready for integration with Google Sheets API."""

        return JSONResponse({
            "status": "success",
            "message": response_text
        })
    except Exception as e:
        logger.error(f"Sheets error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to process sheets request: {str(e)}")

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    try:
        logger.info("Received audio transcription request")

        # Read audio data
        audio_data = await file.read()

        # Transcribe audio
        text, detected_language = await stt.transcribe_audio(audio_data)

        logger.info(f"Transcribed text: {text}, Language: {detected_language}")

        return JSONResponse({
            "status": "success",
            "text": text,
            "language": detected_language
        })
    except Exception as e:
        logger.error(f"Transcription error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to transcribe audio: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)