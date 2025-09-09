# 📖 RAG Chatbot with OCR, Qdrant, Ollama, and Chainlit

This project is an implementation of a **Retrieval-Augmented Generation (RAG) chatbot** developed during an internship.
The chatbot is designed to assist **new employees** who have just joined a team or company.
They can ask basic questions directly to the chatbot first, before escalating more complex questions to senior colleagues or supervisors.

---

## 🚀 Key Features

* 🔎 **OCR with EasyOCR + OpenCV**

  * The system processes company regulation documents in PDF format.
  * Workflow:

    1. Each page of the PDF is converted into an image.
    2. Paragraph areas are detected using contour and morphological operations (dilation/erosion).
    3. EasyOCR extracts text from the identified regions.
    4. The extracted text is cleaned and pre-processed before storage.

* 📦 **Vector Database with Qdrant**

  * Extracted text is transformed into **vector embeddings** using the `paraphrase-multilingual-mpnet-base-v2` model.
  * Each paragraph is stored in Qdrant as a vector along with its original text.
  * When an employee submits a query, it is embedded and matched against the most relevant paragraphs in Qdrant.

* 🧠 **LLM API with FastAPI + Ollama**

  * The top-3 relevant paragraphs retrieved from Qdrant are provided as **context** to the LLM.
  * The LLM (`llama3.1` via Ollama) generates responses in **Bahasa Indonesia**, tailored to employee queries.
  * If no relevant context is found, the chatbot responds with *“I don’t know”* to avoid fabricating answers.

* 💬 **User Interface with Chainlit**

  * Chainlit provides a simple, interactive chat interface for employees.
  * Messages from the user are routed to the FastAPI backend → Qdrant → Ollama, and the chatbot’s response is displayed seamlessly.

---

## 🛠️ System Architecture

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
  FastAPI (LLM Service with Ollama + LLaMA3)
     │
     ▼
  Chainlit (Chatbot User Interface)
```

---

## 📌 Example Use Case

1. Upload company regulation documents in PDF format.
2. OCR extracts the text and saves it as `.txt` files.
3. Qdrant stores paragraphs as vectors for semantic retrieval.
4. A new employee submits a question through the Chainlit chat interface.
5. The system searches for relevant paragraphs, the LLM generates an answer, and the chatbot responds clearly.

---
