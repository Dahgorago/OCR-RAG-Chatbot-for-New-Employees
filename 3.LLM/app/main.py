from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import qdrant_client
import torch
from ollama import Client
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI()

# Inisialisasi model dan klien
model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2', device='cuda' if torch.cuda.is_available() else 'cpu')
client_qdrant = qdrant_client.QdrantClient("http://qdrant_db:6333")
ollama_client = Client(host='http://ollama_api:11434')

class QueryModel(BaseModel):
    prompt: str
        

def get_embeddings(text):
    logger.debug("Generating embeddings for the query.")
    return model.encode(text)

def search_peraturan(query):
    logger.info("Searching for regulations with query: %s", query)
    query_vector = get_embeddings(query)
    try:
        results = client_qdrant.search(
            collection_name="EBook",
            query_vector=query_vector,
            with_vectors=True,
            with_payload=True
        )
        logger.info("Search results found: %d", len(results))
        return [res.payload["text"] for res in results]
    except Exception as e:
        logger.error("Error while searching regulations: %s", str(e))
        raise HTTPException(status_code=500, detail="Error during search operation.")

def generate_response_with_ollama(contexts, query):
    context_text = " ".join(contexts)
    input_text = f"""
    Anda adalah chatbot untuk seseorang bertanya mengenai informasi yang telah disediakan di Database Qdrant.
    Teks informasi di bawah ini adalah hasil pencarian yang relevan dengan pertanyaan yang diajukan, namun mungkin tidak tersusun dengan baik.
    Tugas Anda adalah menyusun ulang informasi ini secara koheren dan menjawab pertanyaan dengan lengkap dan jelas.
    
    {context_text}
    
    Setelah Anda membaca teks informasi tersebut, jawablah pertanyaan berikut dengan lengkap sesuai pertanyaannya:
    Jika teks informasi yang diberikan tidak ada atau kosong, maka Anda jawab "Saya tidak memiliki jawaban berdasarkan informasi yang tersedia", jangan mengarang jawaban dan memunculkan jawaban yang tidak relevan.
    Pertanyaan: {query}
    
    Jika Anda dapat menemukan jawaban berdasarkan teks informasi tersebut, berikan jawaban yang lengkap dengan susunan yang jelas dan koheren, tidak perlu menambahkan jawaban tambahan lainnya.
    Jika Anda tidak dapat menemukan jawaban yang tepat, katakan bahwa "Saya tidak memiliki jawaban berdasarkan informasi yang tersedia", jangan mengarang jawaban dan memunculkan jawaban yang tidak relevan.
    Jawablah sesuai dengan Bahasa yang digunakan di query.
    """

    try:
        logger.info("Generating response from Ollama model.")
        response = ollama_client.chat(
            model="llama3.1",
            messages=[
                {"role": "system", "content": "Jawablah pertanyaan berdasarkan informasi berikut. Jika tidak ada informasi yang relevan, katakan 'Saya tidak memiliki jawaban berdasarkan informasi yang tersedia', jangan mengarang jawaban dan memunculkan jawaban yang tidak relevan."},
                {"role": "user", "content": input_text},
            ],
            options={"seed":40},
            stream=True
        )
        
        output = ""
        for chunk in response:
            output += chunk.get('message', {}).get('content', '')

        logger.info("Response generated successfully.")
        return output

    except Exception as e:
        logger.error("Error while generating response with Ollama: %s", str(e))
        raise HTTPException(status_code=500, detail="Error during response generation.")

@app.get("/")
async def read_root():
    logger.info("Root endpoint accessed.")
    return {"message": "Welcome to LLM API"}

@app.post("/ask")
async def ask_question(query: QueryModel):
    try:
        logger.info("Received query: %s", query.prompt)
        results = search_peraturan(query.prompt)
        
        if not results:
            logger.info("No relevant results found.")
            return {"results": "No relevant results found."}
        
        response = generate_response_with_ollama(results, query.prompt)
        return {"results": response}
    
    except Exception as e:
        logger.error("Error processing query: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e))