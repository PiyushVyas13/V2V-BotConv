from fastapi import FastAPI, Request, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, HTMLResponse, StreamingResponse
from pydantic import BaseModel
from llm import LLMHandler
from pdf_processor import PDFProcessor
from rag import RAGSystem
from fastapi.templating import Jinja2Templates
import os
from typing import List, Optional, Dict, Any
from pathlib import Path
import logging
import uvicorn
import json
from gtts import gTTS  # Import gTTS for text-to-speech
import io  # For handling in-memory bytes
import base64  # For encoding audio data to base64

# Set up logging with more detailed format
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

# Create necessary directories
DATA_DIR = Path("DATA")
RAW_PDFS_DIR = DATA_DIR / "raw_pdfs"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"

# Create directories if they don't exist
RAW_PDFS_DIR.mkdir(parents=True, exist_ok=True)
EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

# Verify embeddings, exclude .gitkeep
embedding_dirs = [d for d in EMBEDDINGS_DIR.glob("*") if d.is_dir() and d.name != ".gitkeep"]
if embedding_dirs:
    logger.info(f"Found {len(embedding_dirs)} existing embedding directories:")
    for dir in embedding_dirs:
        logger.info(f"- {dir.name}")
else:
    logger.warning("No existing embeddings found!")

# Set up templates
templates = Jinja2Templates(directory="templates")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG system components
try:
    logger.info("Initializing RAG system components...")
    pdf_processor = PDFProcessor(data_dir="DATA")
    llm_handler = LLMHandler()  # Will use Azure OpenAI deployment from env vars
    rag_system = RAGSystem(pdf_processor, llm_handler)

    # Load existing embeddings
    if embedding_dirs:
        logger.info("Loading existing embeddings...")
        if rag_system.process_documents():
            logger.info("Successfully loaded existing embeddings")
        else:
            logger.warning("Failed to load existing embeddings")
    else:
        logger.warning("No embeddings found to load")
except Exception as e:
    logger.error(f"Failed to initialize RAG system: {str(e)}")
    raise

class Message(BaseModel):
    role: str
    text: str
    timestamp: Optional[str] = None
    isSpeech: Optional[bool] = False

class ChatRequest(BaseModel):
    text: str
    history: Optional[List[Message]] = []
    is_speech: Optional[bool] = False

class ChatResponse(BaseModel):
    responses: List[str]

class SheetRequest(BaseModel):
    text: str
    history: Optional[List[dict]] = []

@app.get("/", response_class=HTMLResponse)
async def get_index(request: Request):
    logger.info("Serving index page")
    return templates.TemplateResponse("PGP.html", {"request": request})

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        logger.info(f"Received chat request: {request.text}")
        
        # Check if we have any processed documents
        if not rag_system.documents_processed:
            return {
                "responses": ["No documents have been processed. Please upload a PDF first."]
            }

        # Query the RAG system
        context = rag_system.query(request.text)
        if not context:
            logger.warning("No response generated for query")
            return {
                "responses": ["No relevant information found in the documents. Please try a different query."]
            }
        
        # Generate response using RAG system
        response = rag_system.generate_response(
            request.text, 
            context
        )
        
        logger.info("Successfully generated response")
        return response

    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        return {
            "responses": ["I apologize, but I encountered an error. Please try again."]
        }

@app.post("/sheets")
async def process_sheets(request: SheetRequest):
    try:
        logger.info(f"Received sheets request: {request.text}")
        
        if not request.text:
            raise HTTPException(status_code=400, detail="No input provided")

        # Import here to avoid circular imports
        from sheet import llm_parse, get_last_sr_no, update_sheet

        last_sr_no = await get_last_sr_no()
        payload = await llm_parse(request.text, last_sr_no)
        await update_sheet(payload)

        return JSONResponse({
            "status": "success",
            "message": "Successfully updated Google Sheets"
        })
    except Exception as e:
        logger.error(f"Sheets error: {str(e)}")
        return JSONResponse({
            "status": "error",
            "message": str(e)
        }, status_code=500)

@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        if not file.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are allowed")
        
        # Save the uploaded file
        file_path = RAW_PDFS_DIR / file.filename
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Process the PDF
        if rag_system.process_document(str(file_path)):
            return {"message": f"Successfully processed {file.filename}"}
        else:
            raise HTTPException(status_code=500, detail="Failed to process PDF")

    except Exception as e:
        logger.error(f"PDF upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/stream_audio")
async def stream_audio(request: Request):
    try:
        data = await request.json()
        text = data.get("text", "")
        history = data.get("history", [])

        if not text:
            raise HTTPException(status_code=400, detail="No text provided")

        # Use the TextToSpeech instance from tts.py
        from tts import tts
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

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    try:
        logger.info("Received audio transcription request")

        # Read audio data
        audio_data = await file.read()

        # Use the SpeechToText instance from stt.py
        from stt import stt
        text, detected_language = await stt.transcribe_audio(audio_data)

        logger.info(f"Transcribed text: {text}, Language: {detected_language}")

        return JSONResponse(
            {
                "status": "success",
                "text": text,
                "language": detected_language
            }
        )
    except Exception as e:
        logger.error(f"Transcription error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to transcribe audio: {str(e)}")

def start():
    """Start the FastAPI server with uvicorn"""
    logger.info("Starting FastAPI server...")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8003,
        reload=True,
        log_level="info"
    )

if __name__ == "__main__":
    start()