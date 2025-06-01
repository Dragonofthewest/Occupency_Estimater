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
        self.model = genai.GenerativeModel('gemini-2.0-flash-lite')
    
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

# Example usage
def analyze_container_image(image_path):
    image = cv2.imread(image_path)
   
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")
    
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
         percentage = int(occupancy)  # Convert string to integer
         return [max(0, percentage - 5), min(100, percentage + 0)]  # Return ±5% range

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