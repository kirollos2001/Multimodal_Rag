import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from PIL import Image
from src.search_utils import search_products

# Load the test jacket image (red/maroon hooded jacket)
image_path = os.path.join(os.path.dirname(__file__), "test_jacket.jpg")
query_image = Image.open(image_path)

# Arabic query: "I want a jacket like this but in black color"
query_text = "انا عايز جاكت زي ده لكن لونه اسود"

print(f"Query Text: {query_text}")
print(f"Query Image: {image_path}")
print("=" * 60)

# Multimodal search: image + text combined
# final_query_vec = image_vec + text_vec (normalized)
# The image captures the style (hooded zip-up jacket)
# The text captures the desired modification (black color)
print("\nMultimodal search (Image + Text combined):")
print("  Image: red/maroon hooded jacket (style reference)")
print("  Text:  'black jacket' (desired color)")
print("-" * 60)

results = search_products(
    query="black jacket",
    query_image=query_image,
    top_k=5
)

print(f"\nFound {len(results)} results:\n")
for i, r in enumerate(results, 1):
    desc = r.get('description', 'N/A')
    if desc and len(desc) > 120:
        desc = desc[:120] + "..."
    print(f"  {i}. Folder: {r['folder']}")
    print(f"     Price: {r['price']} EGP | Score: {r['score']:.4f}")
    print(f"     Color: {r.get('color', 'N/A')} | Category: {r.get('category', 'N/A')}")
    print(f"     Desc: {desc}")
    print(f"     Image: {r.get('image', 'N/A')}")
    print()
