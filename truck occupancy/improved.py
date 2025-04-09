import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import os

def calculate_container_occupancy(image_path, container_dimensions=(12, 7, 6), 
                                 use_ml_detection=False, model_path=None):
    """
    Calculate the occupancy percentage of a container with cardboard boxes and bags.
    
    Args:
        image_path (str): Path to the image of the container interior
        container_dimensions (tuple): Length, width, height of container in feet
        use_ml_detection (bool): Whether to use machine learning for detection
        model_path (str): Path to pre-trained model if use_ml_detection is True
        
    Returns:
        dict: Results including occupancy percentage and processed images
    """
    # Container dimensions
    length, width, height = container_dimensions
    total_volume = length * width * height
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    # Create a copy for visualization
    viz_image = image.copy()
    
    # Convert to RGB for better processing
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Step 1: Identify container boundaries
    # Method 1: Edge-based container detection
    edges = cv2.Canny(gray_image, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)
    
    # Initialize container boundaries
    left_wall, top_wall = 0, 0
    right_wall, bottom_wall = image.shape[1], image.shape[0]
    
    # If lines are detected, try to find container boundaries
    if lines is not None:
        horizontal_lines = []
        vertical_lines = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if abs(x2 - x1) > abs(y2 - y1):  # Horizontal line
                horizontal_lines.append((x1, y1, x2, y2))
            else:  # Vertical line
                vertical_lines.append((x1, y1, x2, y2))
        
        # Sort lines by position
        if vertical_lines:
            vertical_lines.sort(key=lambda l: l[0])  # Sort by x coordinate
            left_wall = max(0, vertical_lines[0][0] - 10)
            right_wall = min(image.shape[1], vertical_lines[-1][0] + 10)
        
        if horizontal_lines:
            horizontal_lines.sort(key=lambda l: l[1])  # Sort by y coordinate
            top_wall = max(0, horizontal_lines[0][1] - 10)
            bottom_wall = min(image.shape[0], horizontal_lines[-1][1] + 10)
    
    # Method 2: As fallback, use color-based container detection
    if left_wall == 0 and top_wall == 0 and right_wall == image.shape[1] and bottom_wall == image.shape[0]:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Define range for blue/gray color of container walls
        lower_blue = np.array([100, 30, 30])
        upper_blue = np.array([130, 255, 255])
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        
        # Define range for gray color of container walls
        lower_gray = np.array([0, 0, 50])
        upper_gray = np.array([180, 30, 200])
        gray_mask = cv2.inRange(hsv, lower_gray, upper_gray)
        
        # Combine masks
        wall_mask = cv2.bitwise_or(blue_mask, gray_mask)
        
        # Find container boundaries
        y_coords, x_coords = np.where(wall_mask > 0)
        if len(x_coords) > 0 and len(y_coords) > 0:
            left_wall = np.min(x_coords)
            right_wall = np.max(x_coords)
            top_wall = np.min(y_coords)
            bottom_wall = np.max(y_coords)
    
    # Create a mask for the container area, shrinking it slightly to avoid walls
    shrink_factor = int(min(image.shape[0], image.shape[1]) * 0.03)  # Adaptive shrinking
    container_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    container_mask[top_wall + shrink_factor:bottom_wall - shrink_factor, 
                  left_wall + shrink_factor:right_wall - shrink_factor] = 255
    
    # Step 2: Detect cargo (cardboard boxes and bags)
    if use_ml_detection and model_path:
        # Use ML-based detection if available
        cargo_mask = detect_cargo_with_ml(image, container_mask, model_path)
    else:
        # Use enhanced traditional methods
        cargo_mask = detect_cargo_traditional(image, container_mask)
    
    # Step 3: Calculate occupancy
    container_area = cv2.countNonZero(container_mask)
    cargo_area = cv2.countNonZero(cargo_mask)
    
    if container_area > 0:
        occupancy_ratio = cargo_area / container_area
    else:
        occupancy_ratio = 0
    
    # Draw results on visualization image
    contours, _ = cv2.findContours(cargo_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(viz_image, contours, -1, (0, 255, 0), 2)
    cv2.rectangle(viz_image, (left_wall, top_wall), (right_wall, bottom_wall), (255, 0, 0), 2)
    
    # Adjust depth estimation based on box detection
    depth_estimation_factor = estimate_depth_factor(contours, image.shape)
    volume_occupancy = occupancy_ratio * depth_estimation_factor
    occupancy_percentage = volume_occupancy * 100
    
    return {
        'occupancy_percentage': occupancy_percentage,
        'occupied_volume': volume_occupancy * total_volume,
        'total_volume': total_volume,
        'visualization': viz_image,
        'cargo_mask': cargo_mask,
        'container_mask': container_mask
    }

def detect_cargo_traditional(image, container_mask):
    """
    Detect cardboard boxes and bags using traditional computer vision techniques.
    """
    # Convert to multiple color spaces
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # 1. Texture-based detection (good for cardboard)
    # Use Gabor filter to detect cardboard texture
    gabor_kernel = cv2.getGaborKernel((15, 15), 8.0, np.pi/4, 10.0, 0.5, 0, ktype=cv2.CV_32F)
    gabor_img = cv2.filter2D(gray, cv2.CV_8UC3, gabor_kernel)
    
    # Threshold the Gabor filtered image
    _, texture_mask = cv2.threshold(gabor_img, 100, 255, cv2.THRESH_BINARY)
    
    # 2. Color-based detection (for cardboard and bags)
    # Cardboard color range (brown/beige)
    lower_cardboard = np.array([0, 20, 50])
    upper_cardboard = np.array([30, 255, 255])
    cardboard_mask = cv2.inRange(hsv, lower_cardboard, upper_cardboard)
    
    # Additional color ranges for different types of bags and boxes
    # Gray/white boxes and bags
    lower_gray = np.array([0, 0, 50])
    upper_gray = np.array([180, 50, 220])
    gray_mask = cv2.inRange(hsv, lower_gray, upper_gray)
    
    # Dark boxes and bags
    lower_dark = np.array([0, 0, 0])
    upper_dark = np.array([180, 255, 50])
    dark_mask = cv2.inRange(hsv, lower_dark, upper_dark)
    
    # 3. Edge-based detection (for box corners and edges)
    edges = cv2.Canny(gray, 50, 150)
    kernel_dilate = np.ones((5, 5), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel_dilate, iterations=1)
    
    # Combine all detection methods
    combined_mask = cv2.bitwise_or(cardboard_mask, gray_mask)
    combined_mask = cv2.bitwise_or(combined_mask, dark_mask)
    combined_mask = cv2.bitwise_or(combined_mask, texture_mask)
    combined_mask = cv2.bitwise_or(combined_mask, dilated_edges)
    
    # Apply morphological operations to clean up the mask
    kernel = np.ones((7, 7), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
    
    # Apply container mask to restrict detection to container area
    final_cargo_mask = cv2.bitwise_and(combined_mask, container_mask)
    
    # Fill holes in the cargo mask
    filled_cargo_mask = ndimage.binary_fill_holes(final_cargo_mask > 0).astype(np.uint8) * 255
    
    return filled_cargo_mask

def detect_cargo_with_ml(image, container_mask, model_path):
    """
    Detect cardboard boxes and bags using machine learning.
    This is a placeholder - you would need to implement actual ML detection.
    """
    # This is where you would implement ML-based detection
    # For now, we'll just use the traditional method
    from google.cloud import vision
    import io
    
    # Instantiate a client
    client = vision.ImageAnnotatorClient()
    
    # Read the image file
    with io.open(image_path, 'rb') as image_file:
        content = image_file.read()
    
    image = vision.Image(content=content)
    
    # Perform object detection
    response = client.object_localization(image=image)
    objects = response.localized_object_annotations
    
    # Create a mask for detected objects
    mask = np.zeros(container_mask.shape, dtype=np.uint8)
    
    # Get image dimensions
    height, width = container_mask.shape
    
    # Draw detected objects on the mask
    for obj in objects:
        if obj.name.lower() in ['box', 'package', 'container', 'carton', 'bag']:
            vertices = [(vertex.x * width, vertex.y * height) for vertex in obj.bounding_poly.normalized_vertices]
            vertices = np.array(vertices, np.int32)
            cv2.fillPoly(mask, [vertices], 255)
    
    # Apply container mask
    mask = cv2.bitwise_and(mask, container_mask)
    
    return mask

def estimate_depth_factor(contours, image_shape):
    """
    Estimate the depth factor based on detected contours.
    More sophisticated than a fixed factor.
    """
    if not contours:
        return 0.5  # Default value if no contours detected
    
    # Calculate total area of all contours
    total_area = sum(cv2.contourArea(contour) for contour in contours)
    
    # Calculate image area
    image_area = image_shape[0] * image_shape[1]
    
    # Calculate ratio of contour area to image area
    area_ratio = total_area / image_area
    
    # Adjust depth factor based on area ratio
    # The idea: larger visible area typically means objects are closer to the front
    # This is a simple heuristic and could be improved
    if area_ratio > 0.7:
        return 0.9  # Almost full
    elif area_ratio > 0.5:
        return 0.8  # Substantially full
    elif area_ratio > 0.3:
        return 0.7  # Moderately full
    else:
        return 0.6  # Less full
    
def display_results(results):
    """
    Display the results of the occupancy calculation.
    
    Args:
        results (dict): Dictionary containing calculation results
    """
    print(f"Container dimensions: {results['total_volume'] / (12 * 7)} ft × 7 ft × 6 ft")
    print(f"Total volume: {results['total_volume']:.2f} cubic feet")
    print(f"Estimated occupied volume: {results['occupied_volume']:.2f} cubic feet")
    print(f"Occupancy percentage: {results['occupancy_percentage']:.2f}%")
    
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
    image_path = "C:/Users/krish/Downloads/truck images/T12.jpg"
    
    try:
        results = calculate_container_occupancy(image_path)
        display_results(results)
    except Exception as e:
        print(f"Error: {e}")