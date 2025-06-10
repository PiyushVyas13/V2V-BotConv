from gtts import gTTS
import tempfile
import os
import logging
from typing import Optional, Dict

class TextToSpeech:
    def __init__(self):
        """
        Initialize the Text-to-Speech handler with gTTS.
        """
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initialized gTTS")
        
        # Language codes supported by gTTS
        # Note: gTTS does not support explicit selection of male/female voices.
        # It uses the default voice for the selected language.
        self.supported_languages = {
            "en": "en",
            "hi": "hi",
            "gu": "gu",
            # Add more languages as needed
        }
        
        # Set default speed to 1.5x
        self.speed = 1.5

    def text_to_speech(self, text: str, language: str = "en") -> str:
        """
        Convert text to speech and save as audio file.
        
        Args:
            text (str): Text to convert to speech
            language (str): Language code (e.g., 'en', 'hi', 'gu')
            
        Returns:
            str: Path to the generated audio file
        """
        try:
            # Default to English if language not supported
            if language not in self.supported_languages:
                self.logger.warning(f"Language {language} not supported, defaulting to English")
                language = "en"

            # Create temporary file for audio
            temp_file = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
            temp_path = temp_file.name
            temp_file.close()

            # Generate speech using gTTS with precise speed control
            tts = gTTS(text=text, lang=language, slow=False)
            tts.save(temp_path)

            # Apply speed adjustment using ffmpeg
            output_path = temp_path.replace('.mp3', '_speed.mp3')
            os.system(f'ffmpeg -i {temp_path} -filter:a "atempo={self.speed}" {output_path} -y')
            os.remove(temp_path)  # Remove original file

            return output_path

        except Exception as e:
            self.logger.error(f"Error in text-to-speech conversion: {str(e)}")
            raise

    def cleanup(self, file_path: str):
        """
        Clean up temporary audio file.
        
        Args:
            file_path (str): Path to the file to clean up
        """
        try:
            if os.path.exists(file_path):
                os.unlink(file_path)
        except Exception as e:
            self.logger.error(f"Error cleaning up file {file_path}: {str(e)}")

# Create a singleton instance
tts = TextToSpeech()