import logging
import os
import shutil
from typing import Optional, List
from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from .config import Config
from .llm_utils import (
    classify_intent,
    extract_search_params,
    generate_chat_response,
    generate_product_response,
)
from .search_utils import search_products

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Nano Banana — AI Fashion Assistant")

# Mount static files
app.mount("/static", StaticFiles(directory="src/static"), name="static")

# Mount product images
if os.path.exists(Config.BASE_IMAGE_FOLDER):
    app.mount("/images", StaticFiles(directory=Config.BASE_IMAGE_FOLDER), name="images")
else:
    logger.warning(f"Image folder not found: {Config.BASE_IMAGE_FOLDER}")

# Templates
templates = Jinja2Templates(directory="src/templates")

# Ensure upload directory exists
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


# ─── Pydantic Models ─────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: Optional[str] = None
    conversation_history: Optional[List[dict]] = None


class ChatResponse(BaseModel):
    reply: str
    intent: str  # "chat" or "search"
    products: Optional[List[dict]] = None
    extracted_params: Optional[dict] = None


# ─── Routes ───────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Serve the main chat UI."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/chat", response_class=JSONResponse)
async def chat_endpoint(
    message: Optional[str] = Form(None),
    image_file: Optional[UploadFile] = File(None),
    conversation_history: Optional[str] = Form(None),
):
    """
    Main chat endpoint. Handles:
    - Pure conversation (greetings, questions, etc.)
    - Product search (text and/or image based)
    Returns a JSON with the AI reply, intent, and optional product results.
    """
    import json

    image_path = None

    # 1. Handle Image Upload
    if image_file and image_file.filename:
        file_location = os.path.join(UPLOAD_DIR, image_file.filename)
        with open(file_location, "wb+") as file_object:
            shutil.copyfileobj(image_file.file, file_object)
        image_path = file_location
        logger.info(f"Image uploaded to: {image_path}")

    user_text = message.strip() if message else ""

    if not user_text and not image_path:
        return JSONResponse(
            status_code=400,
            content={"message": "Please send a message or upload an image."},
        )

    # Parse conversation history
    history = []
    if conversation_history:
        try:
            history = json.loads(conversation_history)
        except Exception:
            history = []

    # 2. Classify Intent
    intent = classify_intent(user_text, image_path)
    logger.info(f"Intent: {intent} | Text: '{user_text}' | Image: {image_path}")

    # 3. Handle according to intent
    if intent == "chat":
        # Pure conversation — no DB search
        reply = generate_chat_response(user_text, conversation_history=history)
        return {
            "reply": reply,
            "intent": "chat",
            "products": None,
            "extracted_params": None,
        }

    else:
        # Search flow
        extracted_params = extract_search_params(user_text, image_path)
        logger.info(f"Extracted Params: {extracted_params}")

        search_query = extracted_params.get("keywords", user_text)
        qdrant_results = []

        if image_path:
            qdrant_results = search_products(
                query=image_path, filters=extracted_params, top_k=10
            )
        elif search_query:
            qdrant_results = search_products(
                query=search_query, filters=extracted_params, top_k=10
            )

        # Generate natural language response about the products
        reply = generate_product_response(
            user_text, qdrant_results, extracted_params, conversation_history=history
        )

        return {
            "reply": reply,
            "intent": "search",
            "products": qdrant_results,
            "extracted_params": extracted_params,
        }


# Keep legacy search endpoint for backward compatibility
@app.post("/search")
async def search_endpoint(
    text_query: Optional[str] = Form(None),
    image_file: Optional[UploadFile] = File(None),
):
    """Legacy search endpoint."""
    image_path = None
    if image_file and image_file.filename:
        file_location = os.path.join(UPLOAD_DIR, image_file.filename)
        with open(file_location, "wb+") as file_object:
            shutil.copyfileobj(image_file.file, file_object)
        image_path = file_location

    user_input = text_query if text_query else ""
    if not user_input and not image_path:
        return JSONResponse(
            status_code=400,
            content={"message": "Please provide text or an image."},
        )

    extracted_params = extract_search_params(user_input, image_path)
    search_query = extracted_params.get("keywords", user_input)
    qdrant_results = []

    if image_path:
        qdrant_results = search_products(
            query=image_path, filters=extracted_params, top_k=20
        )
    elif search_query:
        qdrant_results = search_products(
            query=search_query, filters=extracted_params, top_k=20
        )

    return {"results": qdrant_results, "extracted_params": extracted_params}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
