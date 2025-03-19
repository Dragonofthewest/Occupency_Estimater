import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

def detect_cargo_area(image, gray):
    """
    Detect the cargo area in a truck image using multiple techniques.
    
    Args:
        image: Original color image
        gray: Grayscale image
        
    Returns:
        Binary mask of the cargo area
    """
    height, width = gray.shape
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Use Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)
    
    # Dilate the edges to connect broken lines
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=2)
    
    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create empty mask
    mask = np.zeros_like(gray)
    
    # Sort contours by area (largest first)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    # Try to find a rectangular contour (cargo area)
    potential_cargo_areas = []
    
    for contour in contours[:10]:  # Check the 10 largest contours
        # Skip very small contours
        if cv2.contourArea(contour) < 0.05 * (height * width):
            continue
            
        # Approximate the contour
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # If we have a polygon with 4-6 points, it could be the cargo area
        if 4 <= len(approx) <= 6:
            # Check if it's somewhat rectangular (aspect ratio check)
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = float(w) / h if h > 0 else 0
            
            # Most cargo areas have reasonable aspect ratios
            if 0.5 <= aspect_ratio <= 2.5:
                # Score the contour based on size and centrality
                area_score = cv2.contourArea(contour) / (height * width)
                center_x, center_y = x + w/2, y + h/2
                centrality_score = 1 - (abs(center_x - width/2) / (width/2) + 
                                       abs(center_y - height/2) / (height/2)) / 2
                
                potential_cargo_areas.append({
                    'contour': approx,
                    'score': area_score * 0.7 + centrality_score * 0.3
                })
    
    if potential_cargo_areas:
        # Use the highest scoring contour
        best_cargo = max(potential_cargo_areas, key=lambda x: x['score'])
        cv2.drawContours(mask, [best_cargo['contour']], -1, 255, -1)
    else:
        # Fallback: use the largest contour if no good rectangles found
        if contours:
            hull = cv2.convexHull(contours[0])
            cv2.drawContours(mask, [hull], -1, 255, -1)
    
    # If no good contours found, use a central region of the image
    if np.count_nonzero(mask) == 0:
        # Use central 70% of the image as default
        margin_h, margin_w = int(height * 0.15), int(width * 0.15)
        mask[margin_h:height-margin_h, margin_w:width-margin_w] = 255
    
    return mask

def detect_cargo_objects(image, gray, cargo_area_mask):
    """
    Detect objects within the cargo area using advanced techniques.
    
    Args:
        image: Original color image
        gray: Grayscale image
        cargo_area_mask: Binary mask of the cargo area
        
    Returns:
        Binary mask of the detected objects
    """
    # Apply the cargo area mask to the original image
    masked_gray = cv2.bitwise_and(gray, gray, mask=cargo_area_mask.astype(np.uint8))
    
    # Method 1: Adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        masked_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # Method 2: Otsu's thresholding as backup
    _, otsu_thresh = cv2.threshold(
        masked_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    
    # Combine results
    combined_thresh = cv2.bitwise_or(thresh, otsu_thresh)
    
    # Clean up the threshold image with morphological operations
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(combined_thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Find contours of objects
    contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create a mask for objects
    objects_mask = np.zeros_like(gray)
    
    # Use color image for additional analysis
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    masked_hsv = cv2.bitwise_and(hsv, hsv, mask=cargo_area_mask.astype(np.uint8))
    
    # Get the average color in the cargo area
    cargo_pixels = masked_hsv[cargo_area_mask > 0]
    if len(cargo_pixels) > 0:
        avg_hue = np.mean(cargo_pixels[:, 0])
        avg_sat = np.mean(cargo_pixels[:, 1])
        avg_val = np.mean(cargo_pixels[:, 2])
    else:
        avg_hue, avg_sat, avg_val = 0, 0, 0
    
    # Calculate the size threshold based on the cargo area
    total_cargo_pixels = np.count_nonzero(cargo_area_mask)
    min_object_size = max(100, total_cargo_pixels * 0.005)  # Min 0.5% of cargo area
    
    # Filter and draw contours
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_object_size:
            # Check if this contour is substantially different from average background
            mask = np.zeros_like(gray)
            cv2.drawContours(mask, [contour], -1, 255, -1)
            
            contour_pixels = masked_hsv[mask > 0]
            if len(contour_pixels) > 0:
                # Check if the contour is different enough from average cargo area
                contour_avg_val = np.mean(contour_pixels[:, 2])  # Value/brightness
                
                # Objects typically have different brightness than empty space
                if abs(contour_avg_val - avg_val) > 15:
                    cv2.drawContours(objects_mask, [contour], -1, 255, -1)
            else:
                # If color analysis fails, include based on size alone
                cv2.drawContours(objects_mask, [contour], -1, 255, -1)
    
    return objects_mask

def calculate_truck_occupancy_unknown_dimensions(image_path, debug=False):
    """
    Enhanced truck occupancy calculator for images with unknown dimensions.
    
    Args:
        image_path: Path to the truck image
        debug: If True, show debug visualizations
        
    Returns:
        float: Occupancy percentage
        dict: Additional information including visualizations
    """
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image at {image_path}")
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Step 1: Detect the cargo area boundaries
    cargo_area_mask = detect_cargo_area(image, gray)
    
    # Step 2: Identify objects within the cargo area
    objects_mask = detect_cargo_objects(image, gray, cargo_area_mask)
    
    # Calculate occupancy as ratio of filled pixels to total cargo area pixels
    total_area = np.count_nonzero(cargo_area_mask)
    filled_area = np.count_nonzero(objects_mask)
    
    if total_area > 0:
        occupancy_percentage = (filled_area / total_area) * 100
    else:
        occupancy_percentage = 0
    
    # Visualize results
    result_image = image.copy()
    debug_images = {}
    
    # Draw cargo area boundary in blue
    cargo_contours, _ = cv2.findContours(cargo_area_mask.astype(np.uint8), 
                                       cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result_image, cargo_contours, -1, (255, 0, 0), 2)
    
    # Draw detected objects in green
    object_contours, _ = cv2.findContours(objects_mask.astype(np.uint8), 
                                        cv2.RETR_EXTERNAL, 
                                        cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result_image, object_contours, -1, (0, 255, 0), 2)
    
    # Add text showing occupancy
    cv2.putText(
        result_image, 
        f"Occupancy: {occupancy_percentage:.2f}%", 
        (10, 30), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        1, 
        (0, 0, 255), 
        2
    )
    
    # Save debug images if requested
    if debug:
        debug_images['cargo_mask'] = cargo_area_mask
        debug_images['objects_mask'] = objects_mask
        
        # Create directory for debug images if it doesn't exist
        debug_dir = os.path.join(os.path.dirname(image_path), 'debug')
        os.makedirs(debug_dir, exist_ok=True)
        
        # Save debug images
        cv2.imwrite(os.path.join(debug_dir, 'cargo_mask.jpg'), cargo_area_mask)
        cv2.imwrite(os.path.join(debug_dir, 'objects_mask.jpg'), objects_mask)
    
    # Save results
    base_name = os.path.basename(image_path)
    name, ext = os.path.splitext(base_name)
    result_path = os.path.join(os.path.dirname(image_path), f"{name}_result{ext}")
    cv2.imwrite(result_path, result_image)
    
    # Return result information
    results = {
        'occupancy_percentage': occupancy_percentage,
        'total_area': total_area,
        'filled_area': filled_area,
        'result_image_path': result_path,
        'debug_images': debug_images if debug else {}
    }
    
    return occupancy_percentage, results

def display_results(results):
    """Display results using matplotlib"""
    result_image = cv2.imread(results['result_image_path'])
    
    plt.figure(figsize=(12, 10))
    
    # Main result
    plt.subplot(2, 2, 1)
    plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
    plt.title(f"Occupancy: {results['occupancy_percentage']:.2f}%")
    plt.axis('off')
    
    # Debug images if available
    if 'cargo_mask' in results['debug_images']:
        plt.subplot(2, 2, 2)
        plt.imshow(results['debug_images']['cargo_mask'], cmap='gray')
        plt.title("Cargo Area Detection")
        plt.axis('off')
    
    if 'objects_mask' in results['debug_images']:
        plt.subplot(2, 2, 3)
        plt.imshow(results['debug_images']['objects_mask'], cmap='gray')
        plt.title("Objects Detection")
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate truck occupancy from image with unknown dimensions")
    parser.add_argument("image_path", help="Path to the truck image")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode with visualizations")
    
    args = parser.parse_args()
    
    try:
        occupancy, results = calculate_truck_occupancy_unknown_dimensions(args.image_path, args.debug)
        print(f"Truck occupancy: {occupancy:.2f}%")
        print(f"Result image saved to: {results['result_image_path']}")
        
        if args.debug:
            display_results(results)
            
    except Exception as e:
        print(f"Error: {e}")