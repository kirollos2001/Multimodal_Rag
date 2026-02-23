import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.search_utils import search_products

results = search_products(query="black jacket", filters={"price_max": 2000.0}, top_k=5)

print(f"\nFound {len(results)} results:\n")
for i, r in enumerate(results, 1):
    desc = r.get('description', 'N/A')
    if desc and len(desc) > 120:
        desc = desc[:120] + "..."
    print(f"  {i}. Folder: {r['folder']}")
    print(f"     Price: {r['price']} EGP | Score: {r['score']:.4f}")
    print(f"     Desc: {desc}")
    print(f"     Image: {r.get('image', 'N/A')}")
    print()
