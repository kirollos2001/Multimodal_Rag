# Requirements:
# pip install torch transformers pillow qdrant-client protobuf sentencepiece python-dotenv

import os
import uuid
import logging
from typing import List, Dict, Tuple, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from .config import Config
from .model_utils import get_image_embedding, get_text_embedding, load_siglip_model

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_info_file(file_path: str) -> Dict:
    """
    Parses the info.txt file to extract Description, ID, and Price.
    Returns a dictionary with keys: 'description', 'id', 'price'.
    """
    info = {'description': '', 'id': '', 'price': 0.0}
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Simple parsing logic based on "Key: Value"
            if ':' in line:
                key, value = line.split(":", 1)
                key = key.strip().lower()
                value = value.strip()
                
                if key == "description":
                    info['description'] = value
                elif key == "id":
                    info['id'] = value
                elif key == "price":
                    try:
                        # Remove commas and currency symbols if present
                        clean_value = value.replace(',', '').replace('$', '').strip()
                        if clean_value.upper() == "SALE PRICE":
                            info['price'] = 0.0
                            logger.info(f"Price marked as 'SALE PRICE' in {file_path}, defaulting to 0.0")
                        else:
                            info['price'] = float(clean_value)
                    except ValueError:
                        logger.warning(f"Could not parse price '{value}' in {file_path}, defaulting to 0.0")
                        info['price'] = 0.0
                    
        return info
    except Exception as e:
        logger.error(f"Error reading info file {file_path}: {e}")
        return info

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
        
        # 1. Parse Info File
        info_file = os.path.join(folder, "info.txt")
        product_info = {'description': '', 'id': folder_name, 'price': 0.0}
        
        if os.path.exists(info_file):
            product_info = parse_info_file(info_file)
        else:
            logger.warning(f"No info.txt found in {folder_name}")
            
        # Ensure we have a product ID, fallback to folder name if missing
        product_id = product_info.get('id') or folder_name
        price = product_info.get('price', 0.0)
        description = product_info.get('description', '')

        # 2. Process Text (Description) -> Create Vector
        if description:
            text_embedding = get_text_embedding(description)
            if text_embedding:
                point_id = str(uuid.uuid4())
                points.append(PointStruct(
                    id=point_id,
                    vector=text_embedding,
                    payload={
                        "product_id": str(product_id),
                        "price": price,
                        "type": "text",
                        "text_content": description,
                        "source_folder": folder_name
                    }
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
                points.append(PointStruct(
                    id=point_id,
                    vector=image_embedding,
                    payload={
                        "product_id": str(product_id),
                        "price": price,
                        "type": "image",
                        "image_filename": img_file,
                        "source_folder": folder_name
                    }
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
