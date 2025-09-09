import torch
import os
from sentence_transformers import SentenceTransformer
import qdrant_client
from qdrant_client.http.models import PointStruct, VectorParams, Distance, Filter

# Inisialisasi Qdrant dan model Sentence Transformers
directory_path = "/OCR/result_ocr"
collection_name = "EBook"
client = qdrant_client.QdrantClient("http://10.12.9.105:6333")
model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2', device='cuda' if torch.cuda.is_available() else 'cpu')

# Fungsi untuk mendapatkan embedding dari teks
def get_embeddings(text):
    return model.encode(text)

# Fungsi untuk membaca teks dari file .txt
def read_text_files(directory_path):
    peraturan_text = ""
    for filename in os.listdir(directory_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(directory_path, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                peraturan_text += file.read() + '\n\n'
    return peraturan_text

# Fungsi untuk memeriksa dan membuat collection jika belum ada
def ensure_collection_exists():
    collections = client.get_collections().collections
    if collection_name not in [c.name for c in collections]:
        client.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=768, distance=Distance.COSINE)  # Sesuaikan ukuran dengan embedding
        )

# Fungsi untuk menghapus semua poin dalam koleksi
def clear_collection():
    client.delete(
        collection_name=collection_name,
        points_selector=Filter(must=[]),  # Filter kosong untuk menghapus semua poin
    )

# Fungsi untuk memperbarui collection Qdrant sesuai file .txt
def update_collection(directory_path):
    ensure_collection_exists()
    
    # Menghapus semua poin di koleksi sebelum memperbarui
    clear_collection()
    
    peraturan_text = read_text_files(directory_path)
    paragraphs = peraturan_text.split('\n\n')
    embeddings = [get_embeddings(p) for p in paragraphs]

    points = [
        PointStruct(id=i, vector=embeddings[i], payload={"text": paragraphs[i]})
        for i in range(len(paragraphs))
    ]

    client.upsert(
        collection_name=collection_name,
        wait=True,
        points=points
    )

# Fungsi untuk mencari paragraf yang relevan berdasarkan pertanyaan
def search_peraturan(query):
    query_vector = get_embeddings(query)
    results = client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=5
    )
    return [res.payload["text"] for res in results]
