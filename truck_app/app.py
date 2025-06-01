from flask import Flask, render_template, request, redirect, url_for, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import csv
import os
import time
import shutil
import gc
import json
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google.oauth2 import service_account
import gspread
from datetime import datetime
from truck_app.depth import analyze_container_image

app = Flask(__name__)
CORS(app)

# Define scopes for both Drive and Sheets APIs
SCOPES = [
    'https://www.googleapis.com/auth/drive',
    'https://www.googleapis.com/auth/spreadsheets'
]

# Load FOLDER_ID and MASTER_SHEET_ID from environment variables with fallback defaults
FOLDER_ID = os.getenv('FOLDER_ID', '1quKgQulsinzYKgUsP9DYOKPz2qkEGyPf')
MASTER_SHEET_ID = os.getenv('MASTER_SHEET_ID', '1GhQUVJHZ3Aon-ILoHSSK88Pphujn9eFZlH0YDjMYtp0')

def get_drive_service():
    """
    Initialize and return Google Drive API service and credentials.
    Loads credentials from SERVICE_ACCOUNT_JSON environment variable.
    Returns (drive_service, credentials) or (None, None) on failure.
    """
    try:
        credentials_json = os.getenv('SERVICE_ACCOUNT_JSON')
        print(f"[DEBUG] SERVICE_ACCOUNT_JSON value: {credentials_json[:50] if credentials_json else 'Not set'}...")  # Log first 50 chars for debugging
        if not credentials_json:
            print("[ERROR] SERVICE_ACCOUNT_JSON environment variable not set")
            return None, None
        try:
            credentials_dict = json.loads(credentials_json)
        except json.JSONDecodeError as e:
            print(f"[ERROR] Invalid SERVICE_ACCOUNT_JSON format: {e}")
            return None, None
        credentials = service_account.Credentials.from_service_account_info(
            credentials_dict, scopes=SCOPES
        )
        drive_service = build('drive', 'v3', credentials=credentials)
        return drive_service, credentials
    except Exception as e:
        print(f"[ERROR] Drive API init failed: {type(e).__name__}: {str(e)}")
        return None, None

def append_to_master_sheet(trip_id, utilization, comments, image_link):
    """
    Append trip data to the Google Sheet specified by MASTER_SHEET_ID.
    """
    try:
        drive_service, credentials = get_drive_service()
        if not credentials:
            print("[ERROR] Failed to get credentials for Sheets")
            return
        client = gspread.authorize(credentials)
        sheet = client.open_by_key(MASTER_SHEET_ID).worksheet('Sheet1')
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        sheet.append_row([trip_id, utilization, comments, image_link, timestamp])
        print(f"[SHEETS] Appended trip {trip_id} to master sheet")
    except Exception as sheet_error:
        print(f"[ERROR] Failed to write to master sheet: {sheet_error}")

def delete_file_with_retries(filepath, retries=5, delay=1):
    """
    Attempt to delete a file with retries.
    """
    for i in range(retries):
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
                print(f"[CLEANUP] Deleted: {filepath}")
                return True
            break
        except Exception as e:
            print(f"[WARN] Retry {i+1}: Could not delete {filepath}: {e}")
            time.sleep(delay)
    print(f"[FAIL] Could not delete {filepath} after {retries} retries.")
    return False

@app.route('/')
def home():
    return render_template("capture-photo.html")

@app.route('/estimation_result.html')
def estimation_result():
    return render_template("estimation_result.html")

@app.route('/googledrive.html')
def googledrive_result():
    return render_template("googledrive.html")

@app.route('/test-drive')
def test_drive():
    """
    Test route to verify Google Drive API connectivity.
    """
    drive_service, _ = get_drive_service()
    if not drive_service:
        return jsonify({'success': False, 'message': 'Drive connection failed'}), 500
    try:
        drive_service.files().list(pageSize=1).execute()
        return jsonify({'success': True, 'message': 'Drive API connected'}), 200
    except Exception as e:
        return jsonify({'success': False, 'message': f'Drive API test failed: {str(e)}'}), 500

@app.route('/api/upload-to-drive', methods=['POST'])
def upload_to_drive():
    """
    Handle file upload to Google Drive and append data to Google Sheet.
    """
    try:
        print(f"[DEBUG] Form data: {request.form}")
        print(f"[DEBUG] Files: {list(request.files.keys())}")
        trip_id = request.form.get('tripId', 'unknown')
        lower = request.form.get('lower', 0)
        upper = request.form.get('upper', 0)
        comments = request.form.get('comments', 'No comments provided')

        uploaded_files = {}
        image_file = request.files.get('image')
        image_filename = None
        if image_file:
            image_filename = f"trip_{trip_id}.jpg"
            image_file.save(image_filename)
            print(f"[INFO] Image saved: {image_filename}")

        drive_service, _ = get_drive_service()
        if not drive_service:
            return jsonify({'success': False, 'message': 'Drive connection failed'}), 500

        image_link = ''
        if image_filename:
            image_metadata = {
                'name': image_filename,
                'mimeType': 'image/jpeg',
                'parents': [FOLDER_ID]
            }
            image_media = MediaFileUpload(image_filename, mimetype='image/jpeg', resumable=True)
            try:
                image_drive_file = drive_service.files().create(
                    body=image_metadata, media_body=image_media, fields='id'
                ).execute()
                uploaded_files['image'] = image_drive_file.get('id')
                print(f"[UPLOAD] Image file ID: {uploaded_files['image']}")
            finally:
                del image_media
                gc.collect()
                time.sleep(1)

            permission = {'type': 'anyone', 'role': 'reader'}
            drive_service.permissions().create(fileId=uploaded_files['image'], body=permission).execute()
            image_link = f"https://drive.google.com/uc?id={uploaded_files['image']}&export=view"

        utilization = f"{lower}%-{upper}%"

        append_to_master_sheet(trip_id, utilization, comments, image_link)

        csv_filename = f"trip_{trip_id}.csv"
        with open(csv_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Trip ID", "Utilization", "Comments", "Image Link"])
            writer.writerow([trip_id, utilization, comments, image_link])

        time.sleep(0.5)
        temp_copy = f"copy_{csv_filename}"
        shutil.copyfile(csv_filename, temp_copy)
        print(f"[INFO] CSV copied to: {temp_copy}")

        csv_metadata = {
            'name': f'Trip_{trip_id}.csv',
            'mimeType': 'text/csv',
            'parents': [FOLDER_ID]
        }
        csv_media = MediaFileUpload(temp_copy, mimetype='text/csv', resumable=True)
        try:
            csv_drive_file = drive_service.files().create(
                body=csv_metadata, media_body=csv_media, fields='id'
            ).execute()
            uploaded_files['csv'] = csv_drive_file.get('id')
            print(f"[UPLOAD] CSV file ID: {uploaded_files['csv']}")
        finally:
            del csv_media
            gc.collect()
            time.sleep(1)

        delete_file_with_retries(csv_filename)
        delete_file_with_retries(temp_copy)
        if image_filename:
            delete_file_with_retries(image_filename)

        return jsonify({
            'success': True,
            'message': f'Files uploaded for trip {trip_id}',
            'fileIds': uploaded_files,
            'imageLink': image_link
        })

    except Exception as e:
        print(f"[ERROR] {e}")
        return jsonify({'success': False, 'message': f'Error processing data: {str(e)}'}), 500

@app.route('/estimate', methods=['POST'])
def estimate():
    """
    Estimate container occupancy from an uploaded image.
    """
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']
    image_data = file.read()

    temp_image_path = 'temp_image.jpg'
    with open(temp_image_path, 'wb') as f:
        f.write(image_data)

    try:
        occupancy_range = analyze_container_image(temp_image_path)

        if occupancy_range == 0:
            print("Invalid image uploaded")
            return jsonify({'status': 'invalid', 'message': 'Upload a valid image'}), 400

        elif occupancy_range == -1:
            return jsonify({'error': 'Failed to analyze image'}), 500

        return jsonify({
            'lower_bound': occupancy_range[0],
            'upper_bound': occupancy_range[1]
        }), 200
    except Exception as e:
        print(f"[ERROR] /estimate route crashed: {e}")
        return jsonify({'error': str(e)}), 500
    finally:
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=False)