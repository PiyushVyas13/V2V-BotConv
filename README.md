# AI Chatbot with PDF Processing

This project implements an AI chatbot that can process and answer questions about PDF documents using RAG (Retrieval Augmented Generation). OPEN AI

## Features

- PDF document processing and embedding generation
- RAG-based question answering
- Voice input/output support
- Automatic PDF processing pipeline

## Directory Structure

```
.
├── DATA/
│   ├── raw_pdfs/     # Place your PDFs here
│   └── embeddings/   # Generated embeddings
├── templates/        # Frontend templates
├── main.py          # FastAPI application
├── pdf_processor.py # PDF processing logic
├── llm.py          # LLM integration
├── tts.py          # Text-to-speech
└── stt.py          # Speech-to-text
```

## Adding New PDFs

1. Place your PDF files in the `DATA/raw_pdfs/` directory
2. The system will automatically:
   - Process new PDFs
   - Generate embeddings
   - Update the vector store
   - Make the content available for querying

## CI/CD Pipeline

The project includes a GitHub Actions workflow that automatically:
1. Detects new or modified PDFs
2. Processes them to generate embeddings
3. Updates the vector store
4. Commits changes back to the repository

### Manual Trigger

You can manually trigger the PDF processing pipeline:
1. Go to the "Actions" tab in your GitHub repository
2. Select "PDF Processing Pipeline"
3. Click "Run workflow"

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Set up environment variables in `.env`:
   ```
   OPENAI_API_KEY=your_openai_api_key
   ELEVENLABS_API_KEY=your_elevenlabs_api_key
   ```

3. Run the server:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000
   ```

## Usage

1. Access the web interface at `http://localhost:8000`
2. Add PDFs to `DATA/raw_pdfs/`
3. Ask questions about the PDF content
4. Use voice input/output as needed

## Contributing

1. Fork the repository
2. Add your PDFs to `DATA/raw_pdfs/`
3. Create a pull request
4. The CI/CD pipeline will automatically process your PDFs

## License

MIT License 