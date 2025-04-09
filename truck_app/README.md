# Truck Utilization Web App

This web application allows users to upload truck container images, estimate cargo utilization using computer vision, and store the data to Google Drive and a Google Sheet for tracking.

## Features

- Upload container images via a web interface
- Estimate occupancy using depth analysis
- Save uploaded images and CSV reports to Google Drive
- Log trip data to a master Google Sheet
- Accessible from mobile devices

## Tech Stack

- Python (Flask)
- OpenCV
- Google Drive API
- Google Sheets API
- HTML/CSS frontend
- Deployment ready with Ngrok or cloud services

## Setup Instructions

1. **Clone the repo**

   ```bash
   git clone https://github.com/Dragonofthewest/Occupency_Estimater.git
   cd Occupency_Estimater

## Create a virtual environment
python -m venv .venv
#On linux use: source .venv/bin/activate
# On Windows use: .venv\Scripts\activate

## Install dependencies
pip install -r requirements.txt

## Run the Flask app
python app.py
