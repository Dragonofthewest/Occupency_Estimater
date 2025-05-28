import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from scipy import ndimage
import google.generativeai as genai
import PIL.Image
import re
import os
from typing import Optional
from dotenv import load_dotenv

class CargoOccupancyAnalyzer:
    def __init__(self, api_key: str):
        """
        Initialize the analyzer with Gemini API key
        
        Args:
            api_key (str): Your Google AI Studio API key
        """
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash')
    
    def analyze_cargo_occupancy(self, image_path: str) -> Optional[str]:
        """
        Analyze truck cargo occupancy from image
        
        Args:
            image_path (str): Path to the truck image file
            
        Returns:
            str: Percentage occupancy (e.g., "75%") or None if analysis fails
        """
        try:
            # Load and validate image
            image = PIL.Image.open(image_path)
            
            # Craft specific prompt for cargo occupancy estimation
            prompt = """
            Analyze this shipping container/truck cargo image and estimate the occupancy percentage.
            
            Instructions:
            1. Look at the entire cargo space from floor to ceiling, front to back
            2. Calculate what percentage of the TOTAL 3D volume is occupied by cargo
            3. Consider: floor coverage, height utilization, and depth usage
            4. If boxes/cargo cover 80% of floor and reach 70% of height, that's roughly 56% total volume
            5. Be generous in your estimation - if it looks well-packed, it's likely 60-80%
            6. Respond with ONLY the percentage number followed by % symbol
            7. If there is no truck or container in the image "NO"
            
            Look at this image carefully and estimate the total volume occupancy percentage.
            Respond with only the percentage value, nothing else.
            """
            
            # Generate response
            response = self.model.generate_content([prompt, image])
            
            # Extract percentage from response
            percentage = self._extract_percentage(response.text)
            return percentage
            
        except FileNotFoundError:
            print(f"Error: Image file '{image_path}' not found.")
            return None
        except Exception as e:
            print(f"Error analyzing image: {str(e)}")
            return None
    
    def _extract_percentage(self, text: str) -> Optional[str]:
        """
        Extract percentage value from response text
        
        Args:
            text (str): Response text from Gemini
            
        Returns:
            str: Cleaned percentage string or None
        """
        # Remove extra whitespace and newlines
        text = text.strip()
        
        print(f"***********\n\r{text}\n\r***************")
        # Look for percentage pattern
        percentage_match = re.search(r'(\d+(?:\.\d+)?)\s*%', text)
        
        if percentage_match:
            # Return clean percentage format
            percentage_value = float(percentage_match.group(1))
            return f"{int(percentage_value)}"
        elif text == "NO":
            return text
        # Fallback: look for just numbers and assume percentage
        number_match = re.search(r'\b(\d+(?:\.\d+)?)\b', text)
        if number_match:
            percentage_value = float(number_match.group(1))
            # Ensure it's a reasonable percentage (0-100)
            if 0 <= percentage_value <= 100:
                return f"{int(percentage_value)}%"
        
        return None


class ContainerOccupancyEstimator:
    def __init__(self):
        self.empty_container_depth = None  # Will be set during calibration
        
    def preprocess_image(self, image):
        """Preprocess the image for better edge detection and analysis"""
        # Convert to grayscale if it's not already
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply histogram equalization for better contrast
        equalized = cv2.equalizeHist(blurred)
        
        return equalized
    
    def detect_container_boundaries(self, image):
        """Detect the boundaries of the shipping container"""
        preprocessed = self.preprocess_image(image)
        
        # Apply Canny edge detection
        edges = cv2.Canny(preprocessed, 50, 150)
        
        # Dilate edges to connect potential gaps
        kernel = np.ones((3, 3), np.uint8)
        dilated_edges = cv2.dilate(edges, kernel, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Find the largest contour (assumed to be the container)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            return (x, y, w, h)
        else:
            # If no contours found, use the whole image
            h, w = image.shape[:2]
            return (0, 0, w, h)
    
    def estimate_occupancy_with_depth_map(self, image):
        """Estimate occupancy using a depth map approach"""
        # Get container boundaries
        x, y, w, h = self.detect_container_boundaries(image)
        
        # Crop to container area
        container_area = image[y:y+h, x:x+w]
        
        # Generate a simple depth map based on brightness
        # (This is a simplification - real depth would require stereo vision or depth sensors)
        gray = cv2.cvtColor(container_area, cv2.COLOR_BGR2GRAY) if len(container_area.shape) == 3 else container_area
        
        # Invert the grayscale image (assuming closer objects are brighter)
        depth_map = 255 - gray
        
        # Normalize depth map
        depth_map = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
        
        # Threshold to separate cargo from empty space
        # This threshold should be calibrated based on your specific containers
        _, binary = cv2.threshold(depth_map, 120, 255, cv2.THRESH_BINARY)
        
        # Calculate occupancy percentage
        occupancy = np.sum(binary > 0) / (binary.shape[0] * binary.shape[1]) * 100
        
        return occupancy, binary
    
    def estimate_occupancy_with_segmentation(self, image):
        """Estimate occupancy using color segmentation"""
        # Get container boundaries
        x, y, w, h = self.detect_container_boundaries(image)
        
        # Crop to container area
        container_area = image[y:y+h, x:x+w]
        
        # Reshape image for KMeans
        pixels = container_area.reshape(-1, 3)
        
        # Apply KMeans clustering to separate cargo from container
        kmeans = KMeans(n_clusters=3, random_state=42)
        labels = kmeans.fit_predict(pixels)
        
        # Reshape back to image dimensions
        segmented = labels.reshape(container_area.shape[0], container_area.shape[1])
        
        # Identify which cluster represents cargo
        # This would need to be calibrated for your specific use case
        # For now, we'll assume the cluster with the middle intensity is cargo
        cluster_means = kmeans.cluster_centers_
        cluster_intensities = np.mean(cluster_means, axis=1)
        cargo_cluster = np.argsort(cluster_intensities)[1]
        
        # Create binary mask of cargo
        cargo_mask = (segmented == cargo_cluster).astype(np.uint8) * 255
        
        # Calculate occupancy percentage
        occupancy = np.sum(cargo_mask > 0) / (cargo_mask.shape[0] * cargo_mask.shape[1]) * 100
        
        return occupancy, cargo_mask
    
    def estimate_occupancy_with_edge_density(self, image):
        """Estimate occupancy using edge density"""
        # Get container boundaries
        x, y, w, h = self.detect_container_boundaries(image)
        
        # Crop to container area
        container_area = image[y:y+h, x:x+w]
        
        # Preprocess
        preprocessed = self.preprocess_image(container_area)
        
        # Apply Canny edge detection
        edges = cv2.Canny(preprocessed, 50, 150)
        
        # Calculate edge density
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        
        # Normalize to percentage (calibration needed)
        # This is a simplification - needs to be calibrated for your specific containers
        normalized_density = min(100, edge_density * 1000)
        
        return normalized_density, edges
    
    def estimate_occupancy_consensus(self, image):
        """Combine multiple methods to get a consensus estimate"""
        depth_occupancy, depth_map = self.estimate_occupancy_with_depth_map(image)
        segmentation_occupancy, segmentation_map = self.estimate_occupancy_with_segmentation(image)
        edge_occupancy, edge_map = self.estimate_occupancy_with_edge_density(image)
        
        # Compute weighted average (can be adjusted based on reliability of each method)
        consensus_occupancy = (depth_occupancy * 0.3 + 
                              segmentation_occupancy * 0.5 + 
                              edge_occupancy * 0.2)
        
        # Round to nearest 5%
        rounded_occupancy = round(consensus_occupancy / 5) * 5
        
        # Provide a range (±5%)
        lower_bound = max(0, rounded_occupancy - 5)
        upper_bound = min(100, rounded_occupancy + 5)
        
        return (lower_bound, upper_bound), (depth_map, segmentation_map, edge_map)
    
    def calibrate_with_empty_container(self, empty_container_image):
        """Calibrate the system with an image of an empty container"""
        # For a real system, this would establish baseline measurements
        # Here we'll just store the image for reference
        self.empty_container_depth = self.preprocess_image(empty_container_image)
        
    def visualize_results(self, image, results):
        """Visualize the estimation results"""
        occupancy_range, maps = results
        depth_map, segmentation_map, edge_map = maps
        
        # Create a figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Original image
        axes[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Depth map
        axes[0, 1].imshow(depth_map, cmap='jet')
        axes[0, 1].set_title('Depth Map')
        axes[0, 1].axis('off')
        
        # Segmentation map
        axes[1, 0].imshow(segmentation_map, cmap='gray')
        axes[1, 0].set_title('Segmentation Map')
        axes[1, 0].axis('off')
        
        # Edge map
        axes[1, 1].imshow(edge_map, cmap='gray')
        axes[1, 1].set_title('Edge Map')
        axes[1, 1].axis('off')
        
        # Add occupancy information
        fig.suptitle(f'Container Occupancy: {occupancy_range[0]}% - {occupancy_range[1]}%', 
                     fontsize=16, y=0.98)
        
        plt.tight_layout()
        return fig
    
# Example usage
def analyze_container_image(image_path):
    estimator = ContainerOccupancyEstimator()
    image = cv2.imread(image_path)
   
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")
    
        """
    Example usage of the CargoOccupancyAnalyzer
    """
    # Load environment variables from .env file
    load_dotenv()
    
    # Get API key from environment variable
    API_KEY = os.getenv('GEMINI_API_KEY')
    
    if not API_KEY:
        print("Please set your GEMINI_API_KEY in the .env file!")
        print("Create a .env file with: GEMINI_API_KEY=your_actual_api_key")
        print("Get your API key from: https://aistudio.google.com/app/apikey")
        return
    
    # Initialize analyzer
    analyzer = CargoOccupancyAnalyzer(API_KEY)
    
    
    print(f"Analyzing cargo occupancy in: {image_path}")
    occupancy = analyzer.analyze_cargo_occupancy(image_path)
    
    if occupancy == "NO":
       print("upload a valid image")
       return 0
    elif occupancy:
         print(f"Cargo occupancy: {occupancy}")
         if int (occupancy) < 12:
             return (int(occupancy)-5), int(occupancy)
         results = estimator.estimate_occupancy_consensus(image)
         occupancy_range, _ = results
         fig = estimator.visualize_results(image, results)
         plt.show()
         return occupancy_range
    else:
        print("Failed to analyze cargo occupancy")
        return -1
    
    

# Alternative function for direct usage
def estimate_cargo_occupancy(image_path: str, api_key: str) -> Optional[str]:
    """
    Direct function to estimate cargo occupancy
    
    Args:
        image_path (str): Path to truck image
        api_key (str): Gemini API key
        
    Returns:
        str: Percentage occupancy or None
    """
    analyzer = CargoOccupancyAnalyzer(api_key)
    return analyzer.analyze_cargo_occupancy(image_path)

# Command-line interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Estimate container occupancy')
    parser.add_argument('image_path', help='Path to the container image')
    args = parser.parse_args()

    occupancy_range = analyze_container_image(args.image_path)
    print(f"Estimated occupancy: {occupancy_range[0]}% - {occupancy_range[1]}%")