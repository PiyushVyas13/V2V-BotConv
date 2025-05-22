import os
import pickle
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from datetime import datetime, timedelta
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# If modifying these scopes, delete the file token.json.
SCOPES = [
    'https://www.googleapis.com/auth/spreadsheets',  # Full access to spreadsheets
    'https://www.googleapis.com/auth/drive.file',    # Access to files created by the app
    'https://www.googleapis.com/auth/drive.readonly' # Read-only access to Drive metadata
]

class GoogleSheetsAuth:
    def __init__(self):
        self.creds = None
        self.token_path = 'token.json'
        self.credentials_path = 'credentials.json'
        self.token_expiry = None

    def authenticate(self):
        """Handles the authentication process for Google Sheets API."""
        try:
            # Check if token.json exists and is valid
            if os.path.exists(self.token_path):
                with open(self.token_path, 'rb') as token:
                    self.creds = pickle.load(token)
                    self.token_expiry = self.creds.expiry

            # If credentials don't exist or are invalid, refresh or get new ones
            if not self.creds or not self.creds.valid:
                if self.creds and self.creds.expired and self.creds.refresh_token:
                    logger.info("Refreshing expired credentials...")
                    self.creds.refresh(Request())
                else:
                    logger.info("Getting new credentials...")
                    if not os.path.exists(self.credentials_path):
                        raise FileNotFoundError(
                            f"credentials.json not found. Please download it from Google Cloud Console "
                            f"and place it in the project directory."
                        )
                    
                    flow = InstalledAppFlow.from_client_secrets_file(
                        self.credentials_path, SCOPES)
                    self.creds = flow.run_local_server(port=0)

                # Save the credentials for future use
                with open(self.token_path, 'wb') as token:
                    pickle.dump(self.creds, token)
                self.token_expiry = self.creds.expiry

            # Check if token is about to expire (within 5 minutes)
            if self.token_expiry and datetime.now() + timedelta(minutes=5) >= self.token_expiry:
                logger.info("Token is about to expire, refreshing...")
                self.creds.refresh(Request())
                with open(self.token_path, 'wb') as token:
                    pickle.dump(self.creds, token)
                self.token_expiry = self.creds.expiry

            return self.creds

        except Exception as e:
            logger.error(f"Authentication error: {str(e)}")
            raise

    def get_service(self):
        """Returns an authenticated Google Sheets service."""
        try:
            creds = self.authenticate()
            service = build('sheets', 'v4', credentials=creds)
            return service
        except Exception as e:
            logger.error(f"Error creating service: {str(e)}")
            raise

def main():
    """Test the authentication process."""
    try:
        auth = GoogleSheetsAuth()
        service = auth.get_service()
        logger.info("Successfully authenticated with Google Sheets API")
        return service
    except Exception as e:
        logger.error(f"Authentication test failed: {str(e)}")
        raise

if __name__ == '__main__':
    main() 