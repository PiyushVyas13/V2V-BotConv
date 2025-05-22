import os
import json
import logging
from GSheet import GoogleSheetsAgent
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def update_sheet_headings():
    """Updates the Google Sheet with specified headings."""
    try:
        # Ensure GOOGLE_SHEETS_ID is loaded from .env
        spreadsheet_id = os.getenv("GOOGLE_SHEETS_ID")
        if not spreadsheet_id or spreadsheet_id == "your-spreadsheet-id":
            logger.error("GOOGLE_SHEETS_ID not set or is default. Please set it in the .env file.")
            return
            
        logger.info(f"Using Spreadsheet ID: {spreadsheet_id}")

        agent = GoogleSheetsAgent(spreadsheet_id)

        headings = [
            "Sr. No.",
            "Description of the Contract",
            "Name of the first Party",
            "Name of the second Party",
            "Date of Request",
            "Purpose",
            "Agreement Commencement date",
            "Duration of the Agreement",
            "Department",
            "Internal Responsibility",
            "Status",
            "Signed Copy Received (Y/N)",
            "Uploaded on Ivalua"
        ]

        sheet_name = "Sheet1" # Assuming the default sheet name is Sheet1
        # Calculate the range based on the number of headings (A1 to the column corresponding to the number of headings)
        # Column M corresponds to the 13th heading
        range_name = f"{sheet_name}!A1:M1"

        payload = {
            "operation": "update",
            "sheet_name": sheet_name,
            "range": range_name,
            "values": [
                headings
            ],
            "metadata": {
                "timestamp": datetime.now().isoformat(), # Need datetime import
                "user": "script",
                "source": "test_script"
            }
        }

        logger.info(f"Attempting to update sheet with payload: {json.dumps(payload)}")
        result = agent.update_sheet(payload)
        logger.info(f"Sheet update successful: {json.dumps(result, indent=2)}")

    except Exception as e:
        logger.error(f"Error updating sheet: {str(e)}")

# Add necessary import for datetime
from datetime import datetime

if __name__ == "__main__":
    update_sheet_headings() 