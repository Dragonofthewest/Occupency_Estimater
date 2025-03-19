import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def calculate_truck_occupancy(image_path):
    """
    Calculate the occupancy percentage of a truck cargo container from an image.
    
    Args:
        image_path (str): Path to the image of the truck container interior
        
    Returns:
        float: Percentage of occupied space in the container (0-100%)
    """
    # Constants - truck dimensions in feet
    TRUCK_LENGTH = 20.0  # feet
    TRUCK_WIDTH = 7.5    # feet
    TRUCK_HEIGHT = 7.5   # feet
    
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    # Convert to grayscale for processing
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply thresholding to separate cargo from empty space
    # Adjust threshold values based on your specific images
    _, thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
    
    # Apply morphological operations to clean up the image
    kernel = np.ones((5, 5), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    
    # Find contours of cargo items
    contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create a mask for the cargo
    cargo_mask = np.zeros_like(gray)
    cv2.drawContours(cargo_mask, contours, -1, 255, -1)
    
    # Calculate the total truck container volume
    total_volume = TRUCK_LENGTH * TRUCK_WIDTH * TRUCK_HEIGHT  # cubic feet
    
    # Calculate the occupied volume by analyzing the cargo mask
    # This is a simplified approach - in a real application, you would need
    # more sophisticated 3D reconstruction techniques
    occupied_pixels = np.count_nonzero(cargo_mask)
    total_pixels = cargo_mask.size
    
    # Calculate occupancy ratio
    occupancy_ratio = occupied_pixels / total_pixels
    
    # Calculate the occupied volume and percentage
    occupied_volume = occupancy_ratio * total_volume
    occupancy_percentage = occupancy_ratio * 100
    
    # Optional: visualize results
    result_image = image.copy()
    cv2.drawContours(result_image, contours, -1, (0, 255, 0), 2)
    
    # Return results
    return {
        'occupancy_percentage': occupancy_percentage,
        'occupied_volume': occupied_volume,
        'total_volume': total_volume,
        'result_image': result_image
    }

def display_results(results):
    """
    Display the results of the occupancy calculation.
    
    Args:
        results (dict): Dictionary containing calculation results
    """
    print(f"Truck dimensions: 20 ft × 7.5 ft × 7.5 ft")
    print(f"Total volume: {results['total_volume']:.2f} cubic feet")
    print(f"Estimated occupied volume: {results['occupied_volume']:.2f} cubic feet")
    print(f"Occupancy percentage: {results['occupancy_percentage']:.2f}%")
    
    # Display the result image
    plt.figure(figsize=(10, 8))
    plt.imshow(cv2.cvtColor(results['result_image'], cv2.COLOR_BGR2RGB))
    plt.title(f"Cargo Occupancy: {results['occupancy_percentage']:.2f}%")
    plt.axis('off')
    plt.show()

# Example usage
if __name__ == "__main__":
    # Replace with your actual image path
    image_path = "C:/Users/krish/Downloads/truck1.png"
    
    try:
        results = calculate_truck_occupancy(image_path)
        display_results(results)
    except Exception as e:
        print(f"Error: {e}")