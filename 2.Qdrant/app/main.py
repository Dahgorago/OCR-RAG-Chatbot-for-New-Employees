import logging
from fastapi import FastAPI, HTTPException
from app.qdrant import update_collection, search_peraturan

# Inisialisasi logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

@app.get("/")
async def read_root():
    return {"message": "Berhasil!"}

@app.get("/status_qdrant/")
async def status_qdrant():
    logger.info("Memulai proses pembaruan Qdrant")
    try:
        directory_path = "/OCR/result_ocr"
        update_collection(directory_path)
        logger.info("Collection Qdrant berhasil diperbarui.")
        return {"status": "Qdrant berhasil diperbarui"}
    
    except Exception as e:
        logger.error(f"Terjadi kesalahan saat memperbarui Qdrant: {str(e)}")
        raise HTTPException(status_code=500, detail="Qdrant gagal diperbarui")

@app.get("/search/")
async def search(query: str):
    logger.info(f"Memulai pencarian dengan query: {query}")
    try:
        results = search_peraturan(query)
        
        if not results:
            logger.warning("Tidak ada hasil yang relevan ditemukan.")
            return {"results": "Tidak ada hasil yang relevan ditemukan."}
        
        logger.info("Pencarian di Qdrant berhasil.")
        return {"results": results}
    
    except Exception as e:
        logger.error(f"Terjadi kesalahan saat melakukan pencarian: {str(e)}")
        raise HTTPException(status_code=500, detail="Pencarian gagal dilakukan")
