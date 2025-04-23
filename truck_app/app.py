from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import csv
import os
import time
import shutil
import gc
from flask import render_template
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google.oauth2 import service_account
from truck_app.depth import ContainerOccupancyEstimator

# NEW: Sheets setup
import gspread
from datetime import datetime

app = Flask(__name__)
CORS(app)

estimator = ContainerOccupancyEstimator()

SCOPES = ['https://www.googleapis.com/auth/drive']
SERVICE_ACCOUNT_FILE = '/var/www/html/Occupency_Estimater/truck_app/service_account.json'
FOLDER_ID = '1AmB8hFseXch-qAREnfXFHrmTwR4sdaQv'

# NEW: Master Sheet ID
MASTER_SHEET_ID = '1ZBYlRzjtFg7WFFOn1C_wuktkdzkmGdMN6xmFVMUpbZQ'  # <-- 🔁 Replace with your sheet ID

def get_drive_service():
    try:
        credentials = service_account.Credentials.from_service_account_file(
            SERVICE_ACCOUNT_FILE, scopes=SCOPES
        )
        return build('drive', 'v3', credentials=credentials)
    except Exception as e:
        print(f"[ERROR] Drive API init failed: {e}")
        return None

def append_to_master_sheet(trip_id, utilization, comments, image_link):
    try:
        scope = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
        credentials = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=scope)
        client = gspread.authorize(credentials)
        sheet = client.open_by_key(MASTER_SHEET_ID).worksheet('Sheet1')
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        sheet.append_row([trip_id, utilization, comments, image_link, timestamp])
        print(f"[SHEETS] Appended trip {trip_id} to master sheet")
    except Exception as sheet_error:
        print(f"[ERROR] Failed to write to master sheet: {sheet_error}")

def delete_file_with_retries(filepath, retries=5, delay=1):
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


@app.route('/api/upload-to-drive', methods=['POST'])
def upload_to_drive():
    try:
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

        drive_service = get_drive_service()
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

        # NEW: Append to master Google Sheet
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
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']
    image_data = file.read()
    image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)

    if image is None:
        return jsonify({'error': 'Invalid image'}), 400

    try:
        occupancy_range, _ = estimator.estimate_occupancy_consensus(image)
        return jsonify({
            'lower_bound': (occupancy_range[0]+ 5),
            'upper_bound': occupancy_range[1]
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=False)
