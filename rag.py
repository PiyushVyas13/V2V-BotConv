import os
from typing import List, Dict, Optional
import logging
from pdf_processor import PDFProcessor
from llm import LLMHandler
from pathlib import Path
from collections import deque

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RAGSystem:
    def __init__(self, 
                 pdf_processor: PDFProcessor,
                 llm_handler: LLMHandler,
                 memory_size: int = 7):
        """
        Initialize the RAG system.
        
        Args:
            pdf_processor: Initialized PDFProcessor instance
            llm_handler: Initialized LLMHandler instance
            memory_size: Number of previous questions to keep in memory
        """
        if not isinstance(pdf_processor, PDFProcessor):
            raise TypeError("pdf_processor must be an instance of PDFProcessor")
        if not isinstance(llm_handler, LLMHandler):
            raise TypeError("llm_handler must be an instance of LLMHandler")
            
        self.pdf_processor = pdf_processor
        self.llm_handler = llm_handler
        self.documents_processed = False
        self.memory_size = memory_size
        self.conversation_memory = deque(maxlen=memory_size)
        logger.info(f"RAG system initialized successfully with memory size {memory_size}")
    
    def add_to_memory(self, query: str, response: str) -> None:
        """
        Add a question-answer pair to the conversation memory.
        
        Args:
            query: User's question
            response: System's response
        """
        self.conversation_memory.append({
            "query": query,
            "response": response
        })
        logger.debug(f"Added to memory. Current memory size: {len(self.conversation_memory)}")
    
    def get_memory_context(self) -> str:
        """
        Get the conversation memory as a formatted string for context.
        
        Returns:
            str: Formatted conversation history
        """
        if not self.conversation_memory:
            return ""
            
        context = "Previous conversation:\n"
        for i, memory in enumerate(self.conversation_memory, 1):
            context += f"{i}. Q: {memory['query']}\n   A: {memory['response']}\n"
        return context
    
    def process_documents(self) -> bool:
        """
        Process all PDFs in the raw_pdfs directory.
        
        Returns:
            bool: True if documents were processed successfully
        """
        logger.info("Processing all PDFs in DATA/raw_pdfs directory...")
        try:
            self.pdf_processor.process_pdfs()
            self.documents_processed = True
            logger.info("PDF processing completed successfully")
            return True
        except Exception as e:
            logger.error(f"Error processing PDFs: {str(e)}")
            self.documents_processed = False
            return False
    
    def process_document(self, pdf_path: str) -> bool:
        """
        Process a single PDF and update the document store.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            bool: True if document was processed successfully
        """
        logger.info(f"Processing document: {pdf_path}")
        try:
            self.pdf_processor.process_pdf(pdf_path)
            self.documents_processed = True
            logger.info(f"Document processed and document store updated: {pdf_path}")
            return True
        except Exception as e:
            logger.error(f"Error processing document {pdf_path}: {str(e)}")
            return False
    
    def query(self, query: str, k: int = 5) -> Optional[List[Dict]]:
        """
        Query the RAG system.
        
        Args:
            query: User query
            k: Number of context chunks to retrieve
            
        Returns:
            Optional[List[Dict]]: List of relevant document chunks with scores, or None if error
        """
        if not query or not isinstance(query, str):
            logger.error("Invalid query: must be a non-empty string")
            return None
            
        if not self.documents_processed:
            logger.warning("No documents have been processed yet")
            return None
        
        try:
            # Retrieve relevant context
            context = self.pdf_processor.search(query, k=k)
            
            if not context:
                logger.info("No relevant context found for query")
                return None
                
            return context
            
        except Exception as e:
            logger.error(f"Error querying RAG system: {str(e)}")
            return None

    def generate_response(self, query: str, context: List[Dict]) -> Dict:
        """
        Generate a response using the LLM handler.
        
        Args:
            query: User query
            context: Retrieved context from RAG system
            
        Returns:
            Dict: Response containing text and optional audio
        """
        try:
            # Get conversation memory context
            memory_context = self.get_memory_context()
            
            # Generate response with memory context
            response = self.llm_handler.generate_response(
                query, 
                context, 
                conversation_history=memory_context
            )
            
            # Add to memory if response was successful
            if response and response["responses"]:
                self.add_to_memory(query, response["responses"][0])
            
            return response
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return {
                "responses": ["I apologize, but I encountered an error while generating the response. Please try again."],
                "audio": None
            }

def main():
    """Run the RAG system independently."""
    try:
        # Initialize components
        pdf_processor = PDFProcessor()
        llm_handler = LLMHandler(model_name="gpt-3.5-turbo")
        
        # Initialize RAG system
        rag = RAGSystem(pdf_processor, llm_handler)
        
        # Process all PDFs in raw_pdfs directory
        if not rag.process_documents():
            print("\nNo documents were processed. Please check the logs for errors.")
            return
        
        # Interactive query loop
        print("\nRAG System Ready! Type 'quit' to exit.")
        print("Enter your questions about the processed PDFs:")
        
        while True:
            query = input("\nYour question: ").strip()
            
            if query.lower() == 'quit':
                break
            
            if not query:
                continue
            
            print("\nGenerating response...")
            try:
                context = rag.query(query)
                if context:
                    response = rag.generate_response(query, context)
                    print("\nResponse:", response["responses"][0])
                else:
                    print("\nNo relevant information found in the documents.")
            except Exception as e:
                logger.error(f"Error generating response: {str(e)}")
                print("\nAn error occurred while generating the response.")
                
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main() 