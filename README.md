# PGP-GPT Application

A web application that combines RAG (Retrieval-Augmented Generation) with voice interaction capabilities.

## Project Structure

```
.
├── frontend/
│   ├── src/
│   │   └── index.html
│   ├── public/
│   │   └── static/
│   └── package.json
├── backend/
│   ├── api/
│   │   └── main.py
│   ├── core/
│   │   ├── llm.py
│   │   ├── rag.py
│   │   ├── stt.py
│   │   └── tts.py
│   ├── data/
│   │   ├── raw_pdfs/
│   │   └── embeddings/
│   └── requirements.txt
└── vercel.json
```

## Deployment Instructions

### Prerequisites

1. Install Vercel CLI:
```bash
npm install -g vercel
```

2. Install Python dependencies:
```bash
cd backend
pip install -r requirements.txt
```

3. Install frontend dependencies:
```bash
cd frontend
npm install
```

### Environment Variables

Create a `.env` file in the backend directory with the following variables:
```
OPENAI_API_KEY=your_openai_api_key
ELEVENLABS_API_KEY=your_elevenlabs_api_key
```

### Local Development

1. Start the backend server:
```bash
cd backend
uvicorn api.main:app --reload
```

2. Start the frontend development server:
```bash
cd frontend
npm start
```

### Deploying to Vercel

1. Login to Vercel:
```bash
vercel login
```

2. Deploy the project:
```bash
vercel
```

3. Set environment variables in Vercel dashboard:
   - Go to your project settings
   - Add the environment variables from your `.env` file

## Features

- PDF document processing and embedding
- RAG-based question answering
- Voice interaction (Speech-to-Text and Text-to-Speech)
- Real-time chat interface
- Streaming audio responses

## API Endpoints

- `POST /api/chat`: Chat endpoint for text-based interaction
- `POST /api/stream_audio`: Endpoint for streaming audio responses
- `POST /api/transcribe`: Endpoint for speech-to-text transcription

## License

MIT 