import chainlit as cl
import requests
import logging
import json

# Setup logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# URL untuk endpoint model LLM Anda
LLM_API_URL = "http://10.12.9.105:8003/ask"

@cl.on_message
async def main(message: cl.Message):
    # Ambil isi pesan dari objek Message
    message_content = message.content
    logger.info("Received message content: %s", message_content)

    # Buat payload untuk mengirimkan pertanyaan ke model LLM
    payload = {"prompt": message_content}
    logger.debug("Payload to LLM: %s", payload)

    try:
        # Kirim permintaan POST ke model LLM
        response = requests.post(LLM_API_URL, json=payload)

        # Periksa apakah permintaan berhasil
        if response.status_code == 200:
            # Ambil respons dari model LLM
            llm_response = response.json().get("results", "Tidak ada jawaban yang tersedia.")
            logger.info("LLM Response: %s", llm_response)
        else:
            llm_response = f"Error: Model LLM mengembalikan status code {response.status_code}"
            logger.error("Failed to get response from LLM. Status code: %d", response.status_code)

    except requests.exceptions.RequestException as e:
        llm_response = f"Error: Tidak dapat menghubungi model LLM. Detail: {str(e)}"
        logger.error("Exception occurred while sending request to LLM: %s", str(e))

    # Log response content
    logger.debug("Sending response to Chainlit: %s", llm_response)

    # Kirim balasan ke Chainlit
    await cl.Message(content=llm_response).send()