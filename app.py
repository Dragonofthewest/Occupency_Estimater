from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import threading
from depth import ContainerOccupancyEstimator

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Initialize the ContainerOccupancyEstimator
estimator = ContainerOccupancyEstimator()

@app.route('/estimate', methods=['POST'])
def estimate():
    """Endpoint to estimate container occupancy"""
    # Check if an image is provided
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    # Read the uploaded image file
    file = request.files['image']
    image_data = file.read()
    image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
    
    # Validate the image
    if image is None:
        return jsonify({'error': 'Invalid image'}), 400
    
    # Display the image in a non-blocking way
    image_thread = threading.Thread(args=(image.copy(),))
    image_thread.daemon = True  # Make thread a daemon so it doesn't prevent program exit
    image_thread.start()
    
    try:
        # Perform the occupancy estimation
        occupancy_range, _ = estimator.estimate_occupancy_consensus(image)
        result = {
            'lower_bound': occupancy_range[0],
            'upper_bound': occupancy_range[1]
        }
        
        # Return the result as JSON
        return jsonify(result), 200
    
    except Exception as e:
        # Handle exceptions gracefully
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Run the Flask app
    app.run(debug=True)