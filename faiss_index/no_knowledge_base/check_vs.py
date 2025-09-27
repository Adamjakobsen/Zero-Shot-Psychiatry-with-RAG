import faiss
import pickle

faiss_path = "index.faiss"
pkl_path = "index.pkl"

# Load FAISS index
index = faiss.read_index(faiss_path)
print(f"Number of vectors: {index.ntotal}")

# If you want to see the raw vector(s)
if index.ntotal > 0:
    xb = index.reconstruct_n(0, index.ntotal)
    print("Vectors:")
    print(xb)

# Load pickle metadata
with open(pkl_path, "rb") as f:
    store = pickle.load(f)

print("\nMetadata entries:")
for i, entry in enumerate(store):
    print(f"[{i}] {entry}")
import pickle

with open("index.pkl", "rb") as f:
    store = pickle.load(f)

print(f"Found {len(store)} metadata entries.\n")
for i, entry in enumerate(store):
    print(f"Entry {i}:")
    print(entry)
    print("-" * 40)