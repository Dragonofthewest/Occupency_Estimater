import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

def calculate_truck_occupancy_advanced(image_path, truck_dimensions=(0, 0, 6)):
    """
    Calculate the occupancy percentage of a truck cargo container from an image.
    
    Args:
        image_path (str): Path to the image of the truck container interior
        truck_dimensions (tuple): Length, width, height of truck in feet
        
    Returns:
        dict: Results including occupancy percentage and processed images
    """
    # Truck dimensions
    length, width, height = truck_dimensions
    total_volume = length * width * height
    print(total_volume)
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    # Create a copy for visualization
    viz_image = image.copy()
    
    # Convert to RGB for better processing
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Create a mask for the cargo area
    # Step 1: Identify the container boundaries
    # This can be done using color segmentation for the blue container walls
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define range for blue color of container walls
    lower_blue = np.array([100, 50, 50])
    upper_blue = np.array([130, 255, 255])
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
    
    # Find container boundaries
    # Get the average of the blue areas to identify container walls
    y_coords, x_coords = np.where(blue_mask > 0)
    if len(x_coords) > 0 and len(y_coords) > 0:
        left_wall = np.min(x_coords)
        right_wall = np.max(x_coords)
        top_wall = np.min(y_coords)
        bottom_wall = np.max(y_coords)
    else:
        # If no blue detected, use the entire image
        left_wall, top_wall = 0, 0
        right_wall, bottom_wall = image.shape[1], image.shape[0]
    
    # Create a mask for the container area
    container_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    container_mask[top_wall:bottom_wall, left_wall:right_wall] = 255
    
    # Step 2: Identify cargo using multiple approaches
    
    # 1. Color-based segmentation for boxes (brown/orange)
    lower_brown = np.array([0, 50, 50])
    upper_brown = np.array([30, 255, 255])
    brown_mask = cv2.inRange(hsv, lower_brown, upper_brown)
    
    # 2. Detect white bags/sacks
    lower_white = np.array([0, 0, 150])
    upper_white = np.array([180, 30, 255])
    white_mask = cv2.inRange(hsv, lower_white, upper_white)
    
    # Combine masks for different cargo types
    combined_cargo_mask = cv2.bitwise_or(brown_mask, white_mask)
    
    # Apply morphological operations to clean up the mask
    kernel = np.ones((5, 5), np.uint8)
    combined_cargo_mask = cv2.morphologyEx(combined_cargo_mask, cv2.MORPH_CLOSE, kernel)
    combined_cargo_mask = cv2.morphologyEx(combined_cargo_mask, cv2.MORPH_OPEN, kernel)
    
    # Apply the container mask to limit cargo detection to container area
    final_cargo_mask = cv2.bitwise_and(combined_cargo_mask, container_mask)
    
    # Fill holes in the cargo mask
    filled_cargo_mask = ndimage.binary_fill_holes(final_cargo_mask > 0).astype(np.uint8) * 255
    
    # Step 3: Calculate occupancy
    container_area = cv2.countNonZero(container_mask)
    cargo_area = cv2.countNonZero(filled_cargo_mask)
    
    if container_area > 0:
        occupancy_ratio = cargo_area / container_area
    else:
        occupancy_ratio = 0
    
    # Visual representation
    contours, _ = cv2.findContours(filled_cargo_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(viz_image, contours, -1, (0, 255, 0), 2)
    
    # Draw container boundaries
    cv2.rectangle(viz_image, (left_wall, top_wall), (right_wall, bottom_wall), (255, 0, 0), 2)
    
    # Calculate depth estimation
    # This is a simplified approach - in reality, you would need depth sensors
    # or multiple camera views for accurate depth estimation
    
    # For this example, we'll use a simple heuristic:
    # Assume cargo fills about 80% of the container depth based on visual inspection
    depth_estimation_factor = 0.80
    
    # Apply depth factor to occupancy calculation
    volume_occupancy = occupancy_ratio * depth_estimation_factor
    occupancy_percentage = volume_occupancy * 100
    
    return {
        'occupancy_percentage': occupancy_percentage,
        'occupied_volume': volume_occupancy * total_volume,
        'total_volume': total_volume,
        'visualization': viz_image,
        'cargo_mask': filled_cargo_mask,
        'container_mask': container_mask
    }

def display_advanced_results(results):
    """
    Display the results of the advanced occupancy calculation.
    
    Args:
        results (dict): Dictionary containing calculation results
    """
    print(f"Truck dimensions: 12 ft × 7 ft × 6 ft")
    print(f"Total volume: {results['total_volume']:.2f} cubic feet")
    print(f"Estimated occupied volume: {results['occupied_volume']:.2f} cubic feet")
    print(f"Occupancy percentage: {results['occupancy_percentage']:.2f}%")
    
    # Display the result images
    plt.figure(figsize=(16, 10))
    
    plt.subplot(2, 2, 1)
    plt.imshow(cv2.cvtColor(results['visualization'], cv2.COLOR_BGR2RGB))
    plt.title("Cargo Detection")
    plt.axis('off')
    
    plt.subplot(2, 2, 2)
    plt.imshow(results['cargo_mask'], cmap='gray')
    plt.title("Cargo Mask")
    plt.axis('off')
    
    plt.subplot(2, 2, 3)
    plt.imshow(results['container_mask'], cmap='gray')
    plt.title("Container Mask")
    plt.axis('off')
    
    plt.subplot(2, 2, 4)
    plt.bar(['Empty', 'Occupied'], [100 - results['occupancy_percentage'], results['occupancy_percentage']])
    plt.title(f"Occupancy: {results['occupancy_percentage']:.1f}%")
    plt.ylim(0, 100)
    
    plt.tight_layout()
    plt.show()

# Example usage
if __name__ == "__main__":
    image_path = "C:/Users/krish/Downloads/aigen7.jpg"
    
    try:
        results = calculate_truck_occupancy_advanced(image_path)
        display_advanced_results(results)
    except Exception as e:
        print(f"Error: {e}")