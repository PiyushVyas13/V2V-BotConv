import os
from typing import List, Dict, Optional
from openai import AzureOpenAI
from dotenv import load_dotenv
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class LLMHandler:
    def __init__(self, model_name: str = None):
        """
        Initialize the LLM handler with Azure OpenAI.
        
        Args:
            model_name: Name of the Azure OpenAI deployment (will use env var if not provided)
        """
        self.endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
        self.deployment_name = model_name or os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o")
        
        if not self.endpoint:
            raise ValueError("AZURE_OPENAI_ENDPOINT not found in environment variables")
        if not self.api_key:
            raise ValueError("AZURE_OPENAI_API_KEY not found in environment variables")
        if not self.deployment_name:
            raise ValueError("AZURE_OPENAI_DEPLOYMENT_NAME not found in environment variables")
            
        # Initialize Azure OpenAI client
        self.client = AzureOpenAI(
            api_key=self.api_key,
            api_version=self.api_version,
            azure_endpoint=self.endpoint
        )
        logger.info(f"Azure OpenAI LLM handler initialized with deployment: {self.deployment_name}")
    
    def generate_response(self, 
                         query: str, 
                         context: List[Dict], 
                         conversation_history: Optional[str] = None,
                         chat_history: Optional[List[Dict]] = None) -> Dict:
        """
        Generate a response using Azure OpenAI.
        
        Args:
            query: User query
            context: Retrieved context from RAG system
            conversation_history: Optional conversation history string
            chat_history: Optional chat history list for additional context
            
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
            if chat_history:
                # Add recent chat history for additional context
                recent_history = "\n\nRecent conversation:\n"
                for msg in chat_history[-3:]:  # Only use last 3 messages for context
                    role = msg.get('role', 'user')
                    text = msg.get('text', '')
                    recent_history += f"{role.title()}: {text}\n"
                system_message += recent_history
            if context_text:
                system_message += f"\n\nRelevant context:\n{context_text}"

            # Check if the query is asking for a comparison or table format
            query_lower = query.lower()
            is_comparison = any(keyword in query_lower for keyword in ["compare", "differences", "versus", "vs", "table", "format as table"])

            if is_comparison:
                system_message += """
When formatting tables, follow these rules:
1. Use proper Markdown table syntax with headers and alignment
2. Include a clear title for the table using ### or #### heading
3. Ensure all columns are properly aligned using :---: for center, :--- for left, ---: for right
4. Use consistent formatting for similar data
5. Add brief explanations if needed
6. Format the table like this:

### Table Title
| Header 1 | Header 2 | Header 3 |
|:--------:|:---------|---------:|
| Data 1   | Data 2   | Data 3   |
| Data 4   | Data 5   | Data 6   |

For document comparisons, use this format:

### Document Comparison
📄 Document 1 Name
| Aspect | Details |
|:-------|:--------|
| Point 1 | Info 1  |
| Point 2 | Info 2  |

📄 Document 2 Name
| Aspect | Details |
|:-------|:--------|
| Point 1 | Info 1  |
| Point 2 | Info 2  |

Always ensure tables are properly aligned and formatted for readability.
"""

            # Single response using Azure OpenAI
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": query}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            return {"responses": [response.choices[0].message.content.strip()], "audio": None}
        except Exception as e:
            logger.error(f"Error generating response with Azure OpenAI: {str(e)}")
            return {"responses": ["I apologize, but I encountered an error while generating the response. Please try again."], "audio": None}