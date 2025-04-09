import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from scipy import ndimage

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
    
    results = estimator.estimate_occupancy_consensus(image)
    occupancy_range, _ = results
    
    fig = estimator.visualize_results(image, results)
    plt.show()
    
    return occupancy_range

# Command-line interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Estimate container occupancy')
    parser.add_argument('image_path', help='Path to the container image')
    args = parser.parse_args()
    
    occupancy_range = analyze_container_image(args.image_path)
    print(f"Estimated occupancy: {occupancy_range[0]}% - {occupancy_range[1]}%")