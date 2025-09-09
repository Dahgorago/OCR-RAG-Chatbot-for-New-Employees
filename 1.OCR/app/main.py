from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from app.ocr import perform_ocr
import os
import uvicorn
import torch
from datetime import datetime, timedelta

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "API berhasil dijalankan"}

@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    file_location = f"temp_{file.filename}"
    output_folder = '/OCR/result_ocr'
    
    # Manipulasi nama file sesuai dengan aturan yang diinginkan
    modified_filename = file.filename.lower()  # Ubah ke huruf kecil
    modified_filename = modified_filename.replace('.pdf', '')  # Hapus ".pdf"
    modified_filename = modified_filename.replace(' ', '_')  # Ubah spasi menjadi underscore
    
    # Tambahkan timestamp pada nama file
    timestamp = (datetime.now() + timedelta(hours=7)).strftime("%Y-%m-%d-%H-%M-%S")  # Format timestamp
    output_file = os.path.join(output_folder, f"hasil_{modified_filename}_{timestamp}.txt")
    
    # Pastikan folder output ada, jika tidak, buat foldernya
    os.makedirs(output_folder, exist_ok=True)
    
    with open(file_location, "wb+") as file_object:
        file_object.write(file.file.read())
    
    try:
        start_time = datetime.now()
        
        perform_ocr(file_location, output_file)
        os.remove(file_location)
        
        end_time = datetime.now()
        elapsed_time = (end_time - start_time).total_seconds()
        reserved_memory = torch.cuda.memory_reserved(0) / (1024 ** 2)  # Konversi ke MB
        return JSONResponse(content={"message": "OCR berhasil dilakukan", "output_file": output_file, 
                                      "waktu": f"OCR berhasil dilakukan dalam {elapsed_time:.2f} detik",
                                      "reserved memory":f"Memory yang digunakan {reserved_memory} MB"})

    except Exception as e:
        os.remove(file_location)
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/list-ocr-results/")
def list_ocr_results():
    output_folder = '/OCR/result_ocr'
    try:
        # Dapatkan semua file .txt dalam folder
        files = [f for f in os.listdir(output_folder) if f.endswith('.txt')]
        return JSONResponse(content={"files": files})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/get-ocr-result/{filename}")
def get_ocr_result(filename: str):
    output_folder = '/OCR/result_ocr'
    file_path = os.path.join(output_folder, filename)
    
    if os.path.exists(file_path):
        with open(file_path, "r") as file:
            content = file.read()
        return JSONResponse(content={"content": content})
    else:
        return JSONResponse(content={"error": "File tidak ditemukan"}, status_code=404)

#if __name__ == "__main__":
    #uvicorn.run(app, host="0.0.0.0", port=8001)
