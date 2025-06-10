import requests
import time
import json
import os
from dotenv import load_dotenv

load_dotenv()

# Azure AD and Graph API config
TENANT_ID = os.getenv("TENANT_ID")
CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")
# SharePoint Graph API details
SITE_ID = os.getenv("SITE_ID")
DRIVE_ID = os.getenv("DRIVE_ID")

SCOPE = (
    "https://graph.microsoft.com/.default"  # .default scope for client credentials flow
)

TOKEN_URL = f"https://login.microsoftonline.com/{TENANT_ID}/oauth2/v2.0/token"
FILENAME = "contracts.xlsx"
FOLDER_PATH = "XCEL_SHEETS"

DOWNLOAD_URL = f"https://graph.microsoft.com/v1.0/sites/{SITE_ID}/drives/{DRIVE_ID}/root:/{FOLDER_PATH}/{FILENAME}:/content"
UPLOAD_URL = f"https://graph.microsoft.com/v1.0/sites/{SITE_ID}/drives/{DRIVE_ID}/root:/{FOLDER_PATH}/{FILENAME}:/content"

TOKEN_CACHE_FILE = "token_cache.json"


def get_token_from_cache():
    if not os.path.exists(TOKEN_CACHE_FILE):
        return None
    with open(TOKEN_CACHE_FILE, "r") as f:
        data = json.load(f)
    if time.time() >= data.get("expires_at", 0):
        # Token expired
        return None
    return data.get("access_token")


def save_token_to_cache(token_response):
    expires_in = int(token_response["expires_in"])  # usually seconds
    expires_at = time.time() + expires_in - 60  # minus 60 sec as buffer before expiry
    data = {"access_token": token_response["access_token"], "expires_at": expires_at}
    with open(TOKEN_CACHE_FILE, "w") as f:
        json.dump(data, f)


def fetch_access_token():
    print("🔑 Fetching new access token from Azure AD...")
    payload = {
        "client_id": CLIENT_ID,
        "scope": SCOPE,
        "client_secret": CLIENT_SECRET,
        "grant_type": "client_credentials",
    }
    response = requests.post(TOKEN_URL, data=payload)
    if response.status_code == 200:
        token_response = response.json()
        save_token_to_cache(token_response)
        return token_response["access_token"]
    else:
        raise Exception(
            f"Failed to fetch token: {response.status_code}, {response.text}"
        )


def get_access_token():
    token = get_token_from_cache()
    if token:
        return token
    return fetch_access_token()


def download_excel(output_path="downloaded_data.xlsx", filename=None):
    """Download Excel file from SharePoint"""
    if filename is None:
        filename = FILENAME
    
    # Build dynamic URL with the specified filename
    download_url = f"https://graph.microsoft.com/v1.0/sites/{SITE_ID}/drives/{DRIVE_ID}/root:/{FOLDER_PATH}/{filename}:/content"
    
    access_token = get_access_token()
    headers = {"Authorization": f"Bearer {access_token}"}
    print(f"📥 Downloading {filename} from SharePoint...")
    response = requests.get(download_url, headers=headers)
    if response.status_code == 200:
        with open(output_path, "wb") as f:
            f.write(response.content)
        print(f"✅ Downloaded to: {output_path}")
    else:
        raise Exception(
            f"Failed to download. Status: {response.status_code}, Msg: {response.text}"
        )


def upload_excel(local_file_path, filename=None):
    """Upload Excel file to SharePoint"""
    if filename is None:
        filename = FILENAME
    
    # Build dynamic URL with the specified filename
    upload_url = f"https://graph.microsoft.com/v1.0/sites/{SITE_ID}/drives/{DRIVE_ID}/root:/{FOLDER_PATH}/{filename}:/content"
    
    access_token = get_access_token()
    with open(local_file_path, "rb") as f:
        file_bytes = f.read()

    upload_headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    }

    print(f"📤 Uploading {local_file_path} to SharePoint as {filename}...")
    response = requests.put(upload_url, headers=upload_headers, data=file_bytes)
    if response.status_code in (200, 201):
        print(f"✅ Upload successful! File: {filename}")
        web_url = response.json().get("webUrl")
        print("🔗 Web URL:", web_url)
        return True, "Successfully synced to SharePoint", web_url
    else:
        raise Exception(
            f"Upload failed. Status: {response.status_code}, Msg: {response.text}"
        )


if __name__ == "__main__":
    download_excel("downloaded_data.xlsx")
    # modify the file here if needed...
    upload_excel("downloaded_data.xlsx")