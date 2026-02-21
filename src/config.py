import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-3-flash-preview")
    
    QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
    QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
    QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "products_siglip")
    
    SIGLIP_MODEL_NAME = os.getenv("SIGLIP_MODEL_NAME", "google/siglip-base-patch16-224")

    # Base folder for images (defaulting to the one used in ingestion)
    BASE_IMAGE_FOLDER = r"C:\Users\kirollos\products_images"
