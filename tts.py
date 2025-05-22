import io
import base64
from gtts import gTTS
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def text_to_speech(text, lang='en'):
    """
    Convert input text to speech using gTTS and return audio bytes.
    
    Args:
        text (str): The text to convert to speech.
        lang (str): Language code ('en' for English, 'hi' for Hindi).
        
    Returns:
        bytes: Audio data in MP3 format, or None if an error occurs.
    """
    try:
        # Validate language
        if lang not in ['en', 'hi']:
            raise ValueError("Unsupported language. Use 'en' for English or 'hi' for Hindi.")

        # Create gTTS object
        tts = gTTS(text=text, lang=lang, slow=False)

        # Save to bytes buffer
        audio_buffer = io.BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_bytes = audio_buffer.getvalue()
        audio_buffer.close()

        return audio_bytes

    except Exception as e:
        logger.error(f"Error generating audio: {e}")
        return None