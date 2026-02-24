from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, Range
from typing import List, Dict, Optional, Union
import logging
import math
from PIL import Image

from .config import Config
from .model_utils import get_text_embedding, get_image_embedding

logger = logging.getLogger(__name__)


def _combine_embeddings(vec_a: List[float], vec_b: List[float]) -> List[float]:
    """
    Combines two embedding vectors by element-wise addition and L2 normalization.
    This fuses visual and textual signals into a single query vector.
    """
    combined = [a + b for a, b in zip(vec_a, vec_b)]
    norm = math.sqrt(sum(x * x for x in combined))
    if norm > 0:
        combined = [x / norm for x in combined]
    return combined


def search_products(
    query: Optional[Union[str, Image.Image]] = None,
    query_image: Optional[Image.Image] = None,
    filters: Optional[Dict] = None,
    top_k: int = 20
) -> List[Dict]:
    """
    Searches for products in Qdrant using semantic embeddings and metadata filters.
    Returns grouped results: each product has image, price, id, description, and score.

    Supports three modes:
      - Text only:  query="black jacket"
      - Image only: query=<PIL Image>  OR  query_image=<PIL Image>
      - Multimodal: query="black jacket", query_image=<PIL Image>
        -> final_vec = normalize(image_vec + text_vec)
    """
    client = QdrantClient(host=Config.QDRANT_HOST, port=Config.QDRANT_PORT)

    # 1. Generate Embedding(s) for Query
    text_embedding = None
    image_embedding = None

    # Handle text query
    if isinstance(query, str):
        text_embedding = get_text_embedding(query)
    # Handle image passed via `query` param (backward compatible)
    elif isinstance(query, Image.Image):
        image_embedding = get_image_embedding(query)

    # Handle image passed via dedicated `query_image` param
    if query_image is not None:
        image_embedding = get_image_embedding(query_image)

    # Combine or select embedding
    if text_embedding and image_embedding:
        # Multimodal: image_vec + text_vec
        embedding = _combine_embeddings(image_embedding, text_embedding)
        logger.info("Using combined image + text embedding for search.")
    elif text_embedding:
        embedding = text_embedding
    elif image_embedding:
        embedding = image_embedding
    else:
        logger.error("Failed to generate embedding for query.")
        return []

    # 2. Construct Qdrant Filters
    must_conditions = []

    if filters:
        if filters.get('price_min') is not None or filters.get('price_max') is not None:
            price_range = Range(
                gte=filters.get('price_min'),
                lte=filters.get('price_max')
            )
            must_conditions.append(FieldCondition(key="price", range=price_range))

    qdrant_filter = Filter(must=must_conditions) if must_conditions else None

    # 3. Execute Search
    try:
        search_result = client.query_points(
            collection_name=Config.QDRANT_COLLECTION_NAME,
            query=embedding,
            query_filter=qdrant_filter,
            limit=top_k * 2,  # Fetch extra to allow grouping
            score_threshold=0.15
        )

        # 4. Group by product (source_folder) and merge data
        product_map = {}  # source_folder -> merged product dict

        for point in search_result.points:
            payload = point.payload
            folder = payload.get('source_folder', '')

            if folder not in product_map:
                product_map[folder] = {
                    'id': payload.get('product_id'),
                    'folder': folder,
                    'price': payload.get('price'),
                    'score': point.score,
                    'description': None,
                    'image': None,
                    'category': payload.get('Category'),
                    'color': payload.get('Color'),
                    'formality': payload.get('Formality'),
                    'gender': payload.get('Gender'),
                    'season': payload.get('Season'),
                    'style': payload.get('Style'),
                    'pattern': payload.get('Pattern'),
                }

            entry = product_map[folder]

            # Keep the best (highest) score for this product
            if point.score > entry['score']:
                entry['score'] = point.score

            # Fill in description and image from whichever point type has it
            if payload.get('Description'):
                entry['description'] = payload['Description']
            if payload.get('type') == 'image' and payload.get('image_filename'):
                entry['image'] = payload['image_filename']

        # 5. For products still missing image or description, look them up
        folders_need_lookup = [
            f for f, p in product_map.items()
            if p['image'] is None or p['description'] is None
        ]

        if folders_need_lookup:
            for folder in folders_need_lookup:
                try:
                    # Fetch siblings from the same product folder
                    siblings = client.scroll(
                        collection_name=Config.QDRANT_COLLECTION_NAME,
                        scroll_filter=Filter(must=[
                            FieldCondition(key="source_folder", match=MatchValue(value=folder))
                        ]),
                        limit=10
                    )
                    for sib in siblings[0]:
                        sp = sib.payload
                        entry = product_map[folder]
                        if sp.get('Description') and not entry['description']:
                            entry['description'] = sp['Description']
                        if sp.get('type') == 'image' and sp.get('image_filename') and not entry['image']:
                            entry['image'] = sp['image_filename']
                except Exception as e:
                    logger.warning(f"Failed to look up siblings for '{folder}': {e}")

        # 6. Sort by score descending and limit to top_k
        results = sorted(product_map.values(), key=lambda x: x['score'], reverse=True)
        return results[:top_k]

    except Exception as e:
        logger.error(f"Search failed: {e}")
        return []

