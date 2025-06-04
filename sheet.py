from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from google.oauth2 import service_account
from googleapiclient.discovery import build
from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import Optional, List
import os
import json
import logging
import asyncio
from functools import partial
import pandas as pd # Import pandas to easily create markdown table

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

app = FastAPI(title="Sheet API", description="API for Google Sheets Integration")

# Set up templates
templates = Jinja2Templates(directory="templates")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Google Sheets setup
SPREADSHEET_ID = '1lHtR67MqdA4-zPxo7DEsypkMpKUM1RJid0ifnjk6qnQ'
SCOPES = ['https://www.googleapis.com/auth/spreadsheets']
CREDS = service_account.Credentials.from_service_account_file('credentials.json', scopes=SCOPES)
SHEET_SERVICE = build('sheets', 'v4', credentials=CREDS)

# OpenAI setup
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables")
openai_client = OpenAI(api_key=OPENAI_API_KEY)

class SheetRequest(BaseModel):
    text: str
    history: Optional[List[dict]] = []

def run_in_executor(func):
    """Decorator to run synchronous functions in an executor"""
    async def wrapper(*args, **kwargs):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, partial(func, *args, **kwargs))
    return wrapper

@run_in_executor
def llm_parse(user_input: str, last_sr_no: int) -> dict:
    system_prompt = f"""
    You are a helpful assistant that extracts structured contract details from natural language input and formats them into a JSON payload. The user will provide a description of a contract in natural language, and you need to extract the following fields:

    - "Sr. No.": Provided as an integer (last_sr_no + 1)
    - "Description of the Contract": The type or subject of the contract (e.g., "software project")
    - "Name of the first Party": The first party involved in the contract
    - "Name of the second Party": The second party involved in the contract
    - "Date of Request": The date the request was made, in DD/MM/YYYY format
    - "Purpose": The purpose of the contract
    - "Agreement Commencement date": The start date of the agreement, in DD/MM/YYYY format
    - "Duration of the Agreement": The duration of the agreement (e.g., "12 months")
    - "Department Responsibility": The department responsible for the contract
    - "Internal Status": The internal status of the contract (e.g., "In Progress")
    - "Signed Copy RECEIVED on IVALUA (Y/N)": "Y" if a signed copy was received on IVALUA, otherwise "N"
    - "Uploaded on IVALUA": The date the contract was uploaded on IVALUA, in DD/MM/YYYY format

    ### Instructions:
    - Format dates in DD/MM/YYYY.
    - Leave fields blank ("") if not mentioned.
    - Set "Sr. No." to last_sr_no + 1.
    - Return the result as a **valid JSON object only**.
    - Do not include explanations, markdown, or extra text.
    """

    try:
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}
            ],
            temperature=0.2
        )

        content = response.choices[0].message.content.strip()
        logger.info("LLM Raw Response: %s", content)

        # Try extracting JSON from content
        if content.startswith("```json"):
            content = content.split("```json")[1].split("```")[0].strip()
        elif content.startswith("```"):
            content = content.split("```")[1].split("```")[0].strip()

        payload = json.loads(content)
        payload["Sr. No."] = last_sr_no + 1
        return payload

    except Exception as e:
        logger.error("Error calling OpenAI: %s", str(e))
        raise HTTPException(status_code=500, detail=f"Error calling OpenAI: {str(e)}")

@run_in_executor
def get_last_sr_no() -> int:
    try:
        result = SHEET_SERVICE.spreadsheets().values().get(
            spreadsheetId=SPREADSHEET_ID,
            range='Sheet1!A:A'
        ).execute()
        values = result.get('values', [])
        if len(values) <= 1:
            return 0
        last_row = values[-1][0]
        return int(last_row) if last_row.isdigit() else 0
    except Exception as e:
        logger.error("Error getting last Sr. No.: %s", str(e))
        raise HTTPException(status_code=500, detail=f"Error getting last Sr. No.: {str(e)}")

@run_in_executor
def update_sheet(payload: dict) -> None:
    try:
        values = [[
            payload["Sr. No."],
            payload["Description of the Contract"],
            payload["Name of the first Party"],
            payload["Name of the second Party"],
            payload["Date of Request"],
            payload["Purpose"],
            payload["Agreement Commencement date"],
            payload["Duration of the Agreement"],
            payload["Department Responsibility"],
            payload["Internal Status"],
            payload["Signed Copy RECEIVED on IVALUA (Y/N)"],
            payload["Uploaded on IVALUA"]
        ]]
        SHEET_SERVICE.spreadsheets().values().append(
            spreadsheetId=SPREADSHEET_ID,
            range='Sheet1!A1',
            valueInputOption='RAW',
            body={'values': values}
        ).execute()
    except Exception as e:
        logger.error("Error updating sheet: %s", str(e))
        raise HTTPException(status_code=500, detail=f"Error updating sheet: {str(e)}")

# Helper function to convert payload dict to markdown table
def payload_to_markdown_table(payload: dict) -> str:
    if not payload:
        return "No data to display."
    
    # Create a pandas DataFrame from the payload for easy markdown conversion
    df = pd.DataFrame([payload])
    # Convert DataFrame to markdown table
    markdown_table = df.to_markdown(index=False)
    return markdown_table

@app.get("/")
async def get_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/sheets")
async def process_sheets(request: SheetRequest):
    try:
        if not request.text:
            raise HTTPException(status_code=400, detail="No input provided")

        last_sr_no = await get_last_sr_no()
        payload = await llm_parse(request.text, last_sr_no)
        await update_sheet(payload)

        # Convert payload to markdown table string for the response
        markdown_output = payload_to_markdown_table(payload)

        return JSONResponse({
            "status": "success",
            # Return the markdown table string in the message field
            "message": f"Successfully updated Google Sheets.\n\nHere are the extracted details:\n\n{markdown_output}"
        })
    except Exception as e:
        logger.error("Error processing sheets request: %s", str(e))
        return JSONResponse({
            "status": "error",
            "message": str(e)
        }, status_code=500)

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)