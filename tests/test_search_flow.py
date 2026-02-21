"""
End-to-end test: انا عايز jacket اسود بسعر اقل من 2000 جنية

Flow:
  1. Generate text embedding using SigLIP
  2. Fetch top-k results from Qdrant (semantic search)
  3. Apply price filter via metadata
  4. Use Gemini LLM to extract params & generate response

Run:  python tests/test_search_flow.py
"""

import sys
import os
import json
import time
import logging

# ─── Setup path ──────────────────────────────────────
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("test")

QUERY = "انا عايز jacket اسود بسعر اقل من 2000 جنية"
PASS = "✅"
FAIL = "❌"


# ═══════════════════════════════════════════════════════
# Step 1: Test Qdrant Connectivity
# ═══════════════════════════════════════════════════════
def test_qdrant_connection():
    print("\n" + "=" * 55)
    print("  STEP 1: Qdrant Connection")
    print("=" * 55)

    from qdrant_client import QdrantClient
    from src.config import Config

    try:
        client = QdrantClient(host=Config.QDRANT_HOST, port=Config.QDRANT_PORT, timeout=5)
        collections = client.get_collections().collections
        col_names = [c.name for c in collections]
        print(f"  Connected to Qdrant at {Config.QDRANT_HOST}:{Config.QDRANT_PORT}")
        print(f"  Collections found: {col_names}")

        assert Config.QDRANT_COLLECTION_NAME in col_names, \
            f"Collection '{Config.QDRANT_COLLECTION_NAME}' not found!"

        info = client.get_collection(Config.QDRANT_COLLECTION_NAME)
        print(f"  Collection '{Config.QDRANT_COLLECTION_NAME}' → {info.points_count} points")
        assert info.points_count > 0, "Collection is empty! No data to search."

        print(f"  {PASS} Qdrant connection OK")
        return True
    except Exception as e:
        print(f"  {FAIL} Qdrant connection failed: {e}")
        return False


# ═══════════════════════════════════════════════════════
# Step 2: Test SigLIP Embedding Generation
# ═══════════════════════════════════════════════════════
def test_embedding_generation():
    print("\n" + "=" * 55)
    print("  STEP 2: SigLIP Text Embedding Generation")
    print("=" * 55)

    from src.model_utils import get_text_embedding

    test_text = "black jacket"
    print(f"  Generating embedding for: '{test_text}'")

    start = time.time()
    embedding = get_text_embedding(test_text)
    elapsed = time.time() - start

    assert embedding is not None, "Embedding is None!"
    assert len(embedding) > 0, "Embedding is empty!"

    print(f"  Embedding dimension: {len(embedding)}")
    print(f"  First 5 values: {embedding[:5]}")
    print(f"  Generated in {elapsed:.2f}s")
    print(f"  {PASS} Embedding generation OK")
    return True


# ═══════════════════════════════════════════════════════
# Step 3: Test Semantic Search (fetch from Qdrant)
# ═══════════════════════════════════════════════════════
def test_semantic_search():
    print("\n" + "=" * 55)
    print("  STEP 3: Semantic Search + Price Filter via Metadata")
    print("=" * 55)

    from src.search_utils import search_products

    search_text = "black jacket"
    filters = {
        "keywords": "black jacket",
        "category": "jacket",
        "color": "black",
        "shape": None,
        "price_min": None,
        "price_max": 2000.0
    }

    print(f"  Query text: '{search_text}'")
    print(f"  Filters: price_max=2000.0")

    start = time.time()
    results = search_products(query=search_text, filters=filters, top_k=5)
    elapsed = time.time() - start

    print(f"  Search completed in {elapsed:.2f}s")
    print(f"  Found {len(results)} results:\n")

    for i, res in enumerate(results, 1):
        price = res.get('price')
        folder = res.get('folder', 'N/A')
        score = res.get('score', 0)
        desc = res.get('description', '')
        image = res.get('image', '')
        print(f"  {i}. Folder: {folder}")
        print(f"     Price: {price} EGP | Score: {score:.4f}")
        if desc:
            print(f"     Desc: {desc[:80]}")
        if image:
            print(f"     Image: {image}")
        print()

        # Verify price filter was applied
        if price is not None:
            assert float(price) <= 2000.0, \
                f"PRICE FILTER FAILED: Product has price {price} > 2000!"

    if len(results) == 0:
        print(f"  ⚠️  No results found. Check if Qdrant has jacket data with price <= 2000.")
    else:
        print(f"  {PASS} Semantic search + price filter OK")

    return results


# ═══════════════════════════════════════════════════════
# Step 4: Test LLM Param Extraction
# ═══════════════════════════════════════════════════════
def test_llm_extraction():
    print("\n" + "=" * 55)
    print("  STEP 4: LLM Parameter Extraction (Gemini)")
    print("=" * 55)

    from src.llm_utils import extract_search_params

    print(f"  Query: '{QUERY}'")

    start = time.time()
    params = extract_search_params(user_input=QUERY)
    elapsed = time.time() - start

    print(f"  Extracted in {elapsed:.2f}s:")
    print(f"  {json.dumps(params, indent=4, ensure_ascii=False)}")

    # Validate extraction
    keywords = params.get("keywords", "")
    price_max = params.get("price_max")

    assert keywords, "Keywords should not be empty!"
    print(f"\n  Keywords: '{keywords}'")

    if price_max is not None:
        assert price_max == 2000 or price_max == 2000.0, \
            f"price_max should be 2000, got {price_max}"
        print(f"  price_max: {price_max} {PASS}")
    else:
        print(f"  ⚠️  price_max was not extracted (got None)")

    print(f"  {PASS} LLM extraction OK")
    return params


# ═══════════════════════════════════════════════════════
# Step 5: Test LLM Response Generation
# ═══════════════════════════════════════════════════════
def test_llm_response(products, params):
    print("\n" + "=" * 55)
    print("  STEP 5: LLM Natural Response Generation")
    print("=" * 55)

    from src.llm_utils import generate_product_response

    start = time.time()
    response = generate_product_response(
        user_input=QUERY,
        products=products,
        extracted_params=params,
    )
    elapsed = time.time() - start

    print(f"  Generated in {elapsed:.2f}s\n")
    print(f"  ┌─ AI Response ─────────────────────────────────")
    for line in response.split("\n"):
        print(f"  │ {line}")
    print(f"  └──────────────────────────────────────────────\n")

    assert response and len(response) > 5, "Response is empty or too short!"
    print(f"  {PASS} LLM response generation OK")


# ═══════════════════════════════════════════════════════
# Main Runner
# ═══════════════════════════════════════════════════════
if __name__ == '__main__':
    print("\n🧪 NANO BANANA — End-to-End Search Flow Test")
    print(f"   Query: \"{QUERY}\"")
    print(f"   Expected flow: Embedding → Qdrant search → Price filter → LLM response\n")

    results_summary = []

    # Step 1: Qdrant
    try:
        ok = test_qdrant_connection()
        results_summary.append(("Qdrant Connection", ok))
        if not ok:
            print("\n⛔ Cannot continue without Qdrant. Is it running?")
            sys.exit(1)
    except Exception as e:
        print(f"\n{FAIL} Step 1 crashed: {e}")
        sys.exit(1)

    # Step 2: Embedding
    try:
        ok = test_embedding_generation()
        results_summary.append(("SigLIP Embedding", ok))
    except Exception as e:
        print(f"\n{FAIL} Step 2 crashed: {e}")
        results_summary.append(("SigLIP Embedding", False))

    # Step 3: Semantic Search
    products = []
    try:
        products = test_semantic_search()
        results_summary.append(("Semantic Search + Filter", len(products) > 0))
    except Exception as e:
        print(f"\n{FAIL} Step 3 crashed: {e}")
        results_summary.append(("Semantic Search + Filter", False))

    # Step 4: LLM Extraction
    params = {}
    try:
        params = test_llm_extraction()
        results_summary.append(("LLM Param Extraction", True))
    except Exception as e:
        print(f"\n{FAIL} Step 4 crashed: {e}")
        results_summary.append(("LLM Param Extraction", False))

    # Step 5: LLM Response
    try:
        test_llm_response(products, params)
        results_summary.append(("LLM Response Generation", True))
    except Exception as e:
        print(f"\n{FAIL} Step 5 crashed: {e}")
        results_summary.append(("LLM Response Generation", False))

    # ─── Summary ───────────────────────────────────────
    print("\n" + "=" * 55)
    print("  TEST SUMMARY")
    print("=" * 55)
    all_passed = True
    for name, passed in results_summary:
        icon = PASS if passed else FAIL
        print(f"  {icon} {name}")
        if not passed:
            all_passed = False

    if all_passed:
        print(f"\n🎉 All tests passed!")
    else:
        print(f"\n⚠️  Some tests failed. Check output above.")
