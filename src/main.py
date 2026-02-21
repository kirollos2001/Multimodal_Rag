import os
import logging
from PIL import Image
from .llm_utils import extract_search_params
from .search_utils import search_products
from .model_utils import load_siglip_model
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.WARNING, format='%(message)s')
logger = logging.getLogger(__name__)

def main():
    print("Initializing Semantic Search Engine...")
    print("Loading models (this happens once)...")
    
    # Preload models
    load_siglip_model()
    
    print("\n--- Fashion Semantic Search ---")
    print("Enter your query (text) or path to an image.")
    print("Type 'exit' to quit.\n")
    
    while True:
        user_input_text = input("Text Query (or press Enter if image only): ").strip()
        user_input_image = input("Image Path (optional, press Enter to skip): ").strip()
        
        if user_input_text.lower() == 'exit' or (not user_input_text and user_input_image.lower() == 'exit'):
            break
            
        if not user_input_text and not user_input_image:
            print("Please provide at least text or an image.")
            continue
            
        print("\nUsing Gemini to analyze request...")
        
        # 1. Analyze with LLM
        # If user provided just image, we might need a dummy text prompt or let LLM describe it
        valid_image_path = None
        if user_input_image:
            if os.path.exists(user_input_image):
                valid_image_path = user_input_image
            else:
                print(f"Warning: Image path '{user_input_image}' does not exist. Ignoring image.")
        
        search_params = extract_search_params(user_input_text, image_path=valid_image_path)
        
        print(f"DEBUG: Extracted Parameters: {search_params}")
        
        # 2. Perform Search
        # We search primarily using the embedding of the input (Text or Image)
        # Filters are applied from LLM extraction
        
        results = []
        
        # Strategy:
        # If image is provided, use image embedding for search (visual similarity).
        # query_input logic needs to pass PIL Image if image search is desired.
        
        query_input = None
        if valid_image_path:
            try:
                query_input = Image.open(valid_image_path)
            except Exception as e:
                print(f"Error opening query image: {e}")
                query_input = user_input_text
        else:
            # Use extracted keywords (translated, no prices) for embedding
            query_input = search_params.get('keywords', user_input_text)
            print(f"DEBUG: Using query text for embedding: '{query_input}'")
        
        if not query_input:
             print("No valid query input found.")
             continue

        print("\nSearching database...")
        results = search_products(query_input, filters=search_params, top_k=5)
        
        # 3. Display Results
        print(f"\nFound {len(results)} matches:")
        for idx, res in enumerate(results, 1):
            print(f"{idx}. Product: {res.get('folder')} | Score: {res.get('score'):.4f}")
            print(f"   Price: {res.get('price')}")
            if res.get('description'):
                print(f"   Desc: {res.get('description')}")
            if res.get('image'):
                print(f"   Image File: {res.get('image')}")
            print("-" * 30)
            
        print("\nReady for next query.\n")

if __name__ == "__main__":
    main()
