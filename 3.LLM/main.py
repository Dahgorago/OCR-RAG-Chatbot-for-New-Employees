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
            collection_name="peraturan_perusahaan",
            query_vector=query_vector,
            limit=3
        )
        logger.info("Search results found: %d", len(results))
        return [res.payload["text"] for res in results]
    except Exception as e:
        logger.error("Error while searching regulations: %s", str(e))
        raise HTTPException(status_code=500, detail="Error during search operation.")

def generate_response_with_ollama(contexts, query):
    context_text = " ".join(contexts)
    input_text = f"""
    Anda adalah chatbot untuk karyawan bertanya mengenai informasi dan peraturan perusahaan.
    Teks informasi dibawah ini tidak dibaca oleh karyawan, tidak perlu menjawab dengan memberitahu BAB maupun subabnya.
    Jika teks informasi yang diberikan tidak ada atau kosong, maka anda jawab "tidak tahu", jangan mengarang jawaban.
    Berikut adalah teks informasi untuk menjawab pertanyaan nantinya:
    
    {context_text}
    
    setelah anda membaca teks informasi tersebut, jawablah pertanyaan berikut dengan jelas sesuai pertanyaannya:
    Jika teks informasi yang diberikan tidak ada atau kosong, maka anda jawab "tidak tahu", jangan mengarang jawaban.
    Pertanyaan: {query}
    
    Jika Anda dapat menemukan jawaban berdasarkan teks informasi tersebut, berikan jawaban yang jelas, tidak perlu menambahkan jawaban tambahan lainnya, dan jangan menambahkan kalimat seperti "Saya memiliki jawaban berdasarkan informasi yang tersedia." 
    Jika Anda tidak dapat menemukan jawaban yang tepat, katakan bahwa "Saya tidak memiliki jawaban berdasarkan informasi yang tersedia" dan "informasi yang saya miliki hanya ..." lalu berikan saran untuk bertanya kepada atasan dan tawarkan bantuan lebih lanjut jika ada pertanyaan lainnya.
    Jawablah dengan bahasa indonesia.
    Huruf kapital dan huruf kecil sama, tidak berbeda.
    """

    try:
        logger.info("Generating response from Ollama model.")
        response = ollama_client.chat(
            model="llama3.1",
            messages=[
                {"role": "system", "content": input_text}
            ],
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