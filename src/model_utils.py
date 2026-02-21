import torch
from PIL import Image
from transformers import SiglipProcessor, SiglipModel
from typing import List, Optional, Tuple, Union
import logging
import os
from .config import Config

logger = logging.getLogger(__name__)

# Global model cache
_model = None
_processor = None
_device = None

def get_device() -> str:
    global _device
    if _device is None:
        _device = "cuda" if torch.cuda.is_available() else "cpu"
    return _device

def load_siglip_model() -> Tuple[SiglipModel, SiglipProcessor]:
    """
    Loads the SigLIP model and processor. Uses caching to avoid reloading.
    """
    global _model, _processor, _device
    
    if _model is not None and _processor is not None:
        return _model, _processor

    logger.info("Loading SigLIP model and processor...")
    device = get_device()
    try:
        _model = SiglipModel.from_pretrained(Config.SIGLIP_MODEL_NAME).to(device)
        _processor = SiglipProcessor.from_pretrained(Config.SIGLIP_MODEL_NAME)
        _model.eval()
        logger.info("SigLIP model loaded successfully.")
        return _model, _processor
    except Exception as e:
        logger.error(f"Failed to load SigLIP model: {e}")
        raise

def get_image_embedding(image_path_or_pil: Union[str, Image.Image]) -> Optional[List[float]]:
    """
    Generates a normalized embedding for a single image.
    Accepts file path string or PIL Image object.
    """
    model, processor = load_siglip_model()
    device = get_device()
    
    try:
        if isinstance(image_path_or_pil, str):
            image = Image.open(image_path_or_pil).convert("RGB")
        else:
            image = image_path_or_pil.convert("RGB")
            
        inputs = processor(images=image, return_tensors="pt").to(device)
        
        with torch.no_grad():
            output = model.get_image_features(**inputs)
            if isinstance(output, torch.Tensor):
                features = output
            else:
                features = output.pooler_output if hasattr(output, 'pooler_output') else output.last_hidden_state[:, 0, :]
            
            # Normalize
            features = torch.nn.functional.normalize(features, p=2, dim=-1)
            
        return features[0].cpu().to(torch.float32).tolist()
    except Exception as e:
        logger.warning(f"Failed to generate image embedding: {e}")
        return None

def get_text_embedding(text: str) -> Optional[List[float]]:
    """
    Generates a normalized embedding for a text string.
    """
    model, processor = load_siglip_model()
    device = get_device()
    
    try:
        inputs = processor(text=[text], return_tensors="pt", padding="max_length", truncation=True).to(device)
        
        with torch.no_grad():
            output = model.get_text_features(**inputs)
            if isinstance(output, torch.Tensor):
                features = output
            else:
                features = output.pooler_output if hasattr(output, 'pooler_output') else output.last_hidden_state[:, 0, :]
                
            # Normalize
            features = torch.nn.functional.normalize(features, p=2, dim=-1)
            
        return features[0].cpu().to(torch.float32).tolist()
    except Exception as e:
        logger.warning(f"Failed to generate text embedding: {e}")
        return None
