import google.generativeai as genai
from PIL import Image
import json
import logging
import os
from typing import Dict, Optional, List

from .config import Config

logger = logging.getLogger(__name__)

# Configure Gemini
genai.configure(api_key=Config.GEMINI_API_KEY)

# Load system prompt
SYSTEM_PROMPT_PATH = os.path.join(os.path.dirname(__file__), "system_prompt.txt")
SYSTEM_PROMPT_TEXT = ""
if os.path.exists(SYSTEM_PROMPT_PATH):
    with open(SYSTEM_PROMPT_PATH, "r", encoding="utf-8") as f:
        SYSTEM_PROMPT_TEXT = f.read()


def classify_intent(user_input: str, image_path: Optional[str] = None) -> str:
    """
    Determines if user wants to SEARCH for products or just CHAT.
    Returns 'search' or 'chat'.
    If image is provided, always treat as search.
    """
    if image_path:
        return "search"

    if not user_input or not user_input.strip():
        return "chat"

    try:
        model = genai.GenerativeModel(Config.GEMINI_MODEL_NAME)
        prompt = f"""You are an intent classifier for a men's clothing store assistant.
Determine if the user wants to SEARCH for a product, or just CHAT (greeting, question, general conversation).

Rules:
- If the user mentions any clothing item, style, color, price, or asks to find/show something → "search"
- If the user sends a greeting, asks how you are, says thanks, asks a general question → "chat"
- If the user asks about store policies, return options, sizing → "chat"
- If unsure, default to "chat"

Respond with ONLY one word: "search" or "chat"

User message: "{user_input}"
"""
        response = model.generate_content(prompt)
        intent = response.text.strip().lower().replace('"', '').replace("'", "")
        if "search" in intent:
            return "search"
        return "chat"
    except Exception as e:
        logger.error(f"Intent classification failed: {e}")
        return "chat"


def generate_chat_response(
    user_input: str,
    conversation_history: Optional[List[Dict]] = None
) -> str:
    """
    Generates a conversational response (no product search).
    Uses the system prompt to maintain personality.
    """
    try:
        model = genai.GenerativeModel(
            Config.GEMINI_MODEL_NAME,
            system_instruction=SYSTEM_PROMPT_TEXT if SYSTEM_PROMPT_TEXT else None
        )

        messages = []
        if conversation_history:
            for msg in conversation_history[-10:]:  # Last 10 messages for context
                role = "user" if msg.get("role") == "user" else "model"
                messages.append({"role": role, "parts": [msg.get("content", "")]})

        messages.append({"role": "user", "parts": [user_input]})

        chat = model.start_chat(history=messages[:-1])
        response = chat.send_message(messages[-1]["parts"])

        return response.text

    except Exception as e:
        logger.error(f"Chat response generation failed: {e}")
        return "عذرًا، حصل مشكلة. ممكن تحاول تاني؟"


def generate_product_response(
    user_input: str,
    products: List[Dict],
    extracted_params: Dict,
    conversation_history: Optional[List[Dict]] = None
) -> str:
    """
    Takes search results and generates a natural, human-like response
    presenting the products to the user.
    """
    try:
        model = genai.GenerativeModel(
            Config.GEMINI_MODEL_NAME,
            system_instruction=SYSTEM_PROMPT_TEXT if SYSTEM_PROMPT_TEXT else None
        )

        if not products:
            product_context = "No products were found matching the user's criteria."
        else:
            product_lines = []
            for i, p in enumerate(products[:8], 1):
                line = f"{i}. "
                if p.get('folder'):
                    line += f"Product: {p['folder']}"
                if p.get('price'):
                    line += f" | Price: {p['price']} EGP"
                if p.get('description'):
                    line += f" | Description: {p['description']}"
                if p.get('score'):
                    line += f" | Match: {p['score']:.0%}"
                product_lines.append(line)
            product_context = "\n".join(product_lines)

        prompt = f"""The user asked: "{user_input}"

Here are the products retrieved from the database:
{product_context}

Extracted search parameters: {json.dumps(extracted_params, ensure_ascii=False)}

Present these results to the user in a natural, conversational way.
- Respond in the same language the user used.
- Be brief and friendly, like a real shop assistant.
- Mention key details (name, price) naturally.
- If no results found, be supportive and suggest alternatives.
- Do NOT list products in a robotic numbered list. Be creative.
- Do NOT make up products that aren't in the list above.
- Keep it concise (2-4 sentences max for the intro, then briefly mention top picks).
"""

        messages = []
        if conversation_history:
            for msg in conversation_history[-6:]:
                role = "user" if msg.get("role") == "user" else "model"
                messages.append({"role": role, "parts": [msg.get("content", "")]})

        messages.append({"role": "user", "parts": [prompt]})

        chat = model.start_chat(history=messages[:-1])
        response = chat.send_message(messages[-1]["parts"])

        return response.text

    except Exception as e:
        logger.error(f"Product response generation failed: {e}")
        if products:
            return "لقيتلك شوية خيارات حلوة، بص عليهم! 👇"
        return "مش لاقي حاجة بنفس المواصفات دي دلوقتي. تحب تغير الفلتر؟"


def extract_search_params(user_input: str, image_path: Optional[str] = None) -> Dict:
    """
    Uses Gemini to extract structured search parameters from user input (text/image).
    Returns a dictionary with keys: keywords, category, color, shape, price_min, price_max.
    """
    try:
        model = genai.GenerativeModel(Config.GEMINI_MODEL_NAME)

        prompt = """You are a smart fashion search assistant. 
        Your goal is to extract structured search parameters from the user's input to query a product database.
        
        The user might provide text, an image, or both.
        
        1. **Keywords**: Translate any Arabic text to English. Extract core product terms (e.g., "jacket", "jeans"). **Exclude any numbers, prices, or currency symbols from keywords.**
        2. **Category**: Identify the product category (e.g., "Outerwear", "Pants", "Shirts").
        3. **Color**: Extract standard colors (e.g., "black", "blue", "red").
        4. **Shape/Fit**: Extract fit information (e.g., "oversize", "slim fit", "regular").
        5. **Price**: Extract price constraints if mentioned.
        
        Return the result as a raw JSON object with NO markdown formatting. 
        The keys should be: "keywords" (string), "category" (string or null), "color" (string or null), "shape" (string or null), "price_min" (float or null), "price_max" (float or null).
        
        Example Input: "انا عايز جاكت اسود oversize بسعر اقل من 1500 جنية"
        Example Output: {"keywords": "black oversize jacket", "category": "jacket", "color": "black", "shape": "oversize", "price_min": null, "price_max": 1500.0}
        """

        content = [prompt, f"User Input: {user_input}"]

        if image_path:
            try:
                img = Image.open(image_path)
                content.append(img)
                content.append(
                    "Also analyze this image for visual attributes like color, shape, and category if not explicitly mentioned in text."
                )
            except Exception as e:
                logger.error(f"Failed to load image for LLM: {e}")

        response = model.generate_content(content)

        # Clean response text (remove code blocks if present)
        text_response = response.text.replace("```json", "").replace("```", "").strip()

        return json.loads(text_response)

    except Exception as e:
        logger.error(f"LLM extraction failed: {e}")
        return {
            "keywords": user_input,
            "category": None,
            "color": None,
            "shape": None,
            "price_min": None,
            "price_max": None,
        }
