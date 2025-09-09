# 📖 RAG Chatbot dengan OCR, Qdrant, Ollama, dan Chainlit

Proyek ini adalah implementasi **Retrieval-Augmented Generation (RAG) chatbot** yang dikembangkan selama magang.
Chatbot ini dirancang untuk membantu **karyawan baru** yang baru bergabung dalam tim atau perusahaan.
Mereka bisa menanyakan pertanyaan dasar terlebih dahulu ke chatbot, sebelum menanyakan pertanyaan yang lebih kompleks kepada senior atau atasan.

---

## 🚀 Fitur Utama

* 🔎 **OCR dengan EasyOCR + OpenCV**

  * Sistem ini dapat membaca dokumen PDF berisi peraturan perusahaan.
  * Prosesnya melibatkan:

    1. Konversi setiap halaman PDF menjadi gambar.
    2. Pendeteksian area paragraf menggunakan kontur dan morfologi (dilate/erode).
    3. EasyOCR membaca teks dari area paragraf tersebut.
    4. Teks hasil ekstraksi dibersihkan dari noise sebelum disimpan.

* 📦 **Vector Database dengan Qdrant**

  * Semua teks hasil OCR diubah menjadi **vector embeddings** menggunakan model `paraphrase-multilingual-mpnet-base-v2`.
  * Setiap paragraf disimpan sebagai vektor dalam Qdrant dengan payload berupa teks asli.
  * Saat karyawan bertanya, query akan diubah menjadi embedding, lalu dicocokkan dengan teks yang paling relevan di Qdrant.

* 🧠 **LLM API dengan FastAPI + Ollama**

  * Hasil pencarian dari Qdrant (top-3 paragraf relevan) digunakan sebagai **konteks** untuk LLM.
  * LLM (`llama3.1` melalui Ollama) memproses pertanyaan + konteks dan menghasilkan jawaban dalam Bahasa Indonesia.
  * Jika konteks tidak relevan atau kosong, chatbot akan menjawab *"tidak tahu"* sesuai aturan agar tidak mengarang jawaban.

* 💬 **UI dengan Chainlit**

  * Chainlit menyediakan antarmuka chat sederhana, di mana karyawan bisa langsung berinteraksi dengan chatbot.
  * Saat pesan dikirim, Chainlit meneruskannya ke API FastAPI → Qdrant → Ollama, lalu menampilkan jawaban kembali ke pengguna.

---

## 🛠️ Arsitektur Sistem

```
[PDF Documents] 
     │
     ▼
  OCR (EasyOCR + OpenCV)
     │
     ▼
  Embedding (Sentence-Transformers)
     │
     ▼
  Qdrant (Vector DB) ───► Query Matching
     │
     ▼
  FastAPI (LLM Service dengan Ollama + LLaMA3)
     │
     ▼
  Chainlit (User Interface Chatbot)
```

---

## 📌 Contoh Penggunaan

1. Upload dokumen peraturan perusahaan (PDF).
2. OCR mengekstrak teks peraturan menjadi file `.txt`.
3. Qdrant menyimpan paragraf-paragraf aturan dalam bentuk vektor.
4. Karyawan baru bertanya melalui UI Chainlit.
5. Sistem mencari paragraf terkait, LLM menghasilkan jawaban, dan chatbot memberikan respon yang jelas.

---
