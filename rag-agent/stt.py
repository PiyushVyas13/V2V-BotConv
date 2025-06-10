import speech_recognition as sr
import tempfile
import os
import logging
from typing import Optional, Tuple
import langdetect

class SpeechToText:
    def __init__(self):
        """
        Initialize the Speech-to-Text handler with SpeechRecognition.
        """
        self.logger = logging.getLogger(__name__)
        self.recognizer = sr.Recognizer()
        self.logger.info("Initialized SpeechRecognition")

    async def transcribe_audio(self, audio_data: bytes) -> Tuple[str, str]:
        """
        Transcribe audio data to text and detect language.
        
        Args:
            audio_data (bytes): Raw audio data
            
        Returns:
            Tuple[str, str]: (transcribed_text, detected_language)
        """
        try:
            # Save audio data to temporary file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_file.write(audio_data)
                temp_path = temp_file.name

            # Use SpeechRecognition to transcribe
            with sr.AudioFile(temp_path) as source:
                audio = self.recognizer.record(source)
                
                # Try Google's speech recognition first
                try:
                    text = self.recognizer.recognize_google(audio)
                except sr.UnknownValueError:
                    self.logger.warning("Google Speech Recognition could not understand audio")
                    text = ""
                except sr.RequestError as e:
                    self.logger.error(f"Could not request results from Google Speech Recognition service: {e}")
                    text = ""

            # Detect language using langdetect
            try:
                detected_language = langdetect.detect(text) if text else "en"
            except:
                detected_language = "en"

            # Clean up temporary file
            os.unlink(temp_path)

            return text, detected_language

        except Exception as e:
            self.logger.error(f"Error in transcription: {str(e)}")
            raise

    async def stream_transcribe(self, audio_stream: bytes) -> Tuple[str, str]:
        """
        Transcribe streaming audio data.
        
        Args:
            audio_stream (bytes): Streaming audio data
            
        Returns:
            Tuple[str, str]: (transcribed_text, detected_language)
        """
        try:
            return await self.transcribe_audio(audio_stream)
        except Exception as e:
            self.logger.error(f"Error in stream transcription: {str(e)}")
            raise

# Create a singleton instance
stt = SpeechToText()