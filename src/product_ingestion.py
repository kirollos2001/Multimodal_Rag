# Requirements:
# pip install torch transformers pillow qdrant-client protobuf sentencepiece python-dotenv

import os
import uuid
import logging
import json
from typing import List, Dict, Tuple, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from .config import Config
from .model_utils import get_image_embedding, get_text_embedding, load_siglip_model

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_json_file(file_path: str) -> Dict:
    """
    Parses the gemini_description.json file.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error reading json file {file_path}: {e}")
        return {}

def process_product_folder(base_folder: str) -> List[PointStruct]:
    """
    Scans the base folder and creates Qdrant points for all products.
    """
    points = []
    
    if not os.path.exists(base_folder):
        logger.error(f"Base folder {base_folder} does not exist.")
        return points

    # Iterate over subdirectories
    try:
        subfolders = [f.path for f in os.scandir(base_folder) if f.is_dir()]
    except Exception as e:
        logger.error(f"Error scanning directory {base_folder}: {e}")
        return points
        
    logger.info(f"Found {len(subfolders)} product folders.")

    # Ensure model is loaded once
    load_siglip_model()

    for folder in subfolders:
        folder_name = os.path.basename(folder)
        
        # 1. Parse JSON File
        info_file = os.path.join(folder, "gemini_description.json")
        product_info = {}
        
        if os.path.exists(info_file):
            product_info = parse_json_file(info_file)
        else:
            logger.warning(f"No gemini_description.json found in {folder_name}")
            
        # Ensure we have a product ID, fallback to folder name if missing
        product_id = product_info.get('ID', folder_name)
        try:
            price = float(product_info.get('Price', 0.0))
        except (ValueError, TypeError):
            price = 0.0
            
        # Combine all attributes except ID and Price for text embedding
        text_attributes = []
        for key, value in product_info.items():
            if key not in ['ID', 'Price']:
                text_attributes.append(f"{key}: {value}")
        
        description = ", ".join(text_attributes)

        # Build base payload including all attributes
        base_payload = {
            "product_id": str(product_id),
            "price": price,
            "source_folder": folder_name
        }
        for key, value in product_info.items():
             base_payload[key] = value

        # 2. Process Text (Combined Attributes) -> Create Vector
        if description:
            text_embedding = get_text_embedding(description)
            if text_embedding:
                point_id = str(uuid.uuid4())
                payload = base_payload.copy()
                payload.update({
                    "type": "text",
                    "text_content": description
                })
                points.append(PointStruct(
                    id=point_id,
                    vector=text_embedding,
                    payload=payload
                ))

        # 3. Process Images -> Create Vectors
        try:
            image_files = [
                f for f in os.listdir(folder) 
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))
            ]
        except Exception as e:
            logger.error(f"Error listing images in {folder}: {e}")
            continue
        
        for img_file in image_files:
            img_path = os.path.join(folder, img_file)
            image_embedding = get_image_embedding(img_path)
            
            if image_embedding:
                point_id = str(uuid.uuid4())
                payload = base_payload.copy()
                payload.update({
                    "type": "image",
                    "image_filename": img_file
                })
                points.append(PointStruct(
                    id=point_id,
                    vector=image_embedding,
                    payload=payload
                ))
    
    return points

def upsert_to_qdrant(points: List[PointStruct]):
    """
    Upserts points to Qdrant. Creates collection if it doesn't exist.
    """
    if not points:
        logger.warning("No points to upsert.")
        return

    client = None
    try:
        logger.info(f"Connecting to Qdrant at {Config.QDRANT_HOST}:{Config.QDRANT_PORT}...")
        client = QdrantClient(host=Config.QDRANT_HOST, port=Config.QDRANT_PORT, timeout=5)
        # Check connection
        client.get_collections()
        logger.info("Connected to Qdrant server.")
    except Exception:
        logger.warning(f"Could not connect to Qdrant. Falling back to local storage at './qdrant_local'.")
        client = QdrantClient(path="./qdrant_local")
        
    try:
        # Check if collection exists
        collection_exists = client.collection_exists(Config.QDRANT_COLLECTION_NAME)

        if not collection_exists:
            logger.info(f"Collection '{Config.QDRANT_COLLECTION_NAME}' not found. Creating...")
            # SigLIP visual/text embedding dimension is 768 for siglip-base-patch16-224
            if points and points[0].vector:
                vector_size = len(points[0].vector)
            else:
                vector_size = 768 # Fallback known dimension for this model
            
            client.create_collection(
                collection_name=Config.QDRANT_COLLECTION_NAME,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )
            logger.info(f"Collection '{Config.QDRANT_COLLECTION_NAME}' created with vector size {vector_size}.")
        
        logger.info(f"Upserting {len(points)} points...")
        
        # Upsert in batches
        BATCH_SIZE = 100
        total_batches = (len(points) + BATCH_SIZE - 1) // BATCH_SIZE
        
        for i in range(0, len(points), BATCH_SIZE):
            batch = points[i:i + BATCH_SIZE]
            client.upsert(
                collection_name=Config.QDRANT_COLLECTION_NAME,
                points=batch
            )
            logger.info(f"Upserted batch {i // BATCH_SIZE + 1}/{total_batches}")
            
        logger.info("Upsert complete.")
        
    except Exception as e:
        logger.error(f"Error interacting with Qdrant: {e}")

def main():
    logger.info("Starting Product Ingestion Process")
    
    try:
        # Process Folders
        # Using configured base image folder or passed argument
        points = process_product_folder(Config.BASE_IMAGE_FOLDER)
        
        logger.info(f"Total points generated: {len(points)}")
        
        # Upsert to Qdrant
        if points:
            upsert_to_qdrant(points)
        else:
            logger.info("No data processed due to errors or empty folders.")
            
    except Exception as e:
        logger.critical(f"Process failed: {e}")

if __name__ == "__main__":
    main()
