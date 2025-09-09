import torch
import os
from sentence_transformers import SentenceTransformer
import qdrant_client
from qdrant_client.http.models import PointStruct, VectorParams, Distance

# Langkah 1: Membaca semua file .txt dalam satu direktori
directory_path = '.'
peraturan_text = ""

for filename in os.listdir(directory_path):
    if filename.endswith('.txt'):
        file_path = os.path.join(directory_path, filename)
        with open(file_path, 'r', encoding='utf-8') as file:
            peraturan_text += file.read() + '\n\n'  # Menambahkan pemisah antar file

# Langkah 2: Inisialisasi Qdrant dan model Sentence Transformers
client = qdrant_client.QdrantClient("http://localhost:6333")
model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2', device='cuda' if torch.cuda.is_available() else 'cpu')

# Fungsi untuk mendapatkan embedding dari teks
def get_embeddings(text):
    return model.encode(text).tolist()  # Mengubah tensor menjadi list

# Membagi teks menjadi paragraf dan mendapatkan embeddingnya
paragraphs = peraturan_text.split('\n\n')
embeddings = [get_embeddings(p) for p in paragraphs]

# Langkah 3: Menyimpan paragraf dan embeddingnya ke Qdrant
try:
    # Recreate collection
    client.recreate_collection(
        collection_name="peraturan_perusahaan",
        vectors_config=VectorParams(size=len(embeddings[0]), distance=Distance.COSINE)
    )
    print("Collection recreated successfully.")
except Exception as e:
    print(f"Error recreating collection: {e}")

# Menyiapkan data untuk di-upsert
points = [
    PointStruct(id=i, vector=embeddings[i], payload={"text": paragraphs[i]})
    for i in range(len(paragraphs))
]

try:
    # Upsert points
    client.upsert(
        collection_name="peraturan_perusahaan",
        wait=True,
        points=points
    )
    print("Data upserted successfully.")
except Exception as e:
    print(f"Error upserting data: {e}")

# Fungsi untuk mencari paragraf yang relevan berdasarkan pertanyaan
def search_peraturan(query):
    try:
        query_vector = get_embeddings(query)
        results = client.search(
            collection_name="peraturan_perusahaan",
            query_vector=query_vector,
            limit=3
        )
        # Mengurutkan hasil berdasarkan skor relevansi
        return [res.payload["text"] for res in results]
    except Exception as e:
        print(f"Error searching for data: {e}")
        return ["Error occurred during search."]