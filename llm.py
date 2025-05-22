import os
from typing import List, Dict, Optional
from openai import OpenAI
from dotenv import load_dotenv
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class LLMHandler:
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        """
        Initialize the LLM handler.
        
        Args:
            model_name: Name of the OpenAI model to use
        """
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
            
        self.client = OpenAI(api_key=self.api_key)
        self.model_name = model_name
        logger.info(f"LLM handler initialized with model: {model_name}")
    
    def generate_response(self, 
                         query: str, 
                         context: List[Dict], 
                         conversation_history: Optional[str] = None) -> Dict:
        """
        Generate a response using the LLM.
        
        Args:
            query: User query
            context: Retrieved context from RAG system
            conversation_history: Optional conversation history string
            
        Returns:
            Dict: Response containing text and optional audio
        """
        try:
            # Format the context
            context_text = "\n\n".join([f"Document {i+1}:\n{chunk['text']}" 
                                      for i, chunk in enumerate(context)])
            
            # Create the system message with context and conversation history
            system_message = "You are a helpful AI assistant. Use the following context to answer the user's question."
            if conversation_history:
                system_message += f"\n\n{conversation_history}"
            if context_text:
                system_message += f"\n\nRelevant context:\n{context_text}"

            # Check if the query is asking for a comparison or table format
            query_lower = query.lower()
            is_comparison = any(keyword in query_lower for keyword in ["compare", "differences", "versus", "vs", "table", "format as table"])

            if is_comparison:
                system_message += """
When formatting tables, follow these rules:
1. Use proper Markdown table syntax with headers and alignment
2. Include a clear title for the table
3. Ensure all columns are properly aligned
4. Use consistent formatting for similar data
5. Add brief explanations if needed
6. Format the table like this:

| Header 1 | Header 2 | Header 3 |
|----------|----------|----------|
| Data 1   | Data 2   | Data 3   |
| Data 4   | Data 5   | Data 6   |

For document comparisons, use this format:

📄 Document 1 Name
| Aspect | Details |
|--------|---------|
| Point 1 | Info 1  |
| Point 2 | Info 2  |

📄 Document 2 Name
| Aspect | Details |
|--------|---------|
| Point 1 | Info 1  |
| Point 2 | Info 2  |
"""

            # Single response
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": query}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            return {"responses": [response.choices[0].message.content.strip()], "audio": None}
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return {"responses": ["I apologize, but I encountered an error while generating the response. Please try again."], "audio": None}