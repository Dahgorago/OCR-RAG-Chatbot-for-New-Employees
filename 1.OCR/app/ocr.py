import logging
import time
import re
import easyocr
import cv2
import numpy as np
from pdf2image import convert_from_path
import torch
import psutil

# Konfigurasi logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
reader = easyocr.Reader(['en'], gpu='cuda:0')

def perform_ocr(pdf_path: str, output_path: str) -> None:
    # Function to check if a box (inner) is inside another box (outer)
    def is_inside(inner, outer):
        x1, y1, w1, h1 = inner
        x2, y2, w2, h2 = outer
        return x1 >= x2 and y1 >= y2 and x1 + w1 <= x2 + w2 and y1 + h1 <= y2 + h2

    # Menghapus bounding box yang lebih kecil jika berada di dalam bounding box yang lebih besar
    def remove_nested_boxes(boxes):
        filtered_boxes = []
        for i in range(len(boxes)):
            is_nested = False
            for j in range(len(boxes)):
                if i != j and is_inside(boxes[i], boxes[j]):
                    is_nested = True
                    break
            if not is_nested:
                filtered_boxes.append(boxes[i])
        return filtered_boxes
    
    logging.info("Memulai eksekusi OCR.")
    start_time = time.time()
    
    # Convert PDF to images
    try:
        images = convert_from_path(pdf_path, dpi=300)
        logging.info(f"Konversi PDF ke gambar berhasil, total halaman: {len(images)}")
    except Exception as e:
        logging.error(f"Error saat mengkonversi PDF ke gambar: {e}")
        raise e

    ocr_results_all = []

    for i, img in enumerate(images):
        logging.info(f"Memproses halaman {i+1}")
        try:
            image = np.array(img)
            base_image = image.copy()
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (99, 99), 0)
            thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
            #thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

            kernal = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 41)) # sebelumnya (50,30) (horizontal, vertikal)
            dilate = cv2.dilate(thresh, kernal, iterations=1)
            erosion = cv2.erode(dilate, kernal, iterations=1)

            cnts = cv2.findContours(erosion, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_L1)
            cnts = cnts[0] if len(cnts) == 2 else cnts[1]
            cnts = sorted(cnts, key=lambda x: cv2.boundingRect(x)[1])

            grouped_rois = []
            current_group = []

            for c in cnts:
                x, y, w, h = cv2.boundingRect(c)
                if not (1540 > h > 50 and w > 10):
                    continue
                if current_group and abs(y - current_group[-1][1]) > h / 2:
                    grouped_rois.append(current_group)
                    current_group = []
                if not current_group or abs(current_group[-1][3] - h) <= 100:
                    current_group.append((x, y, w, h))
                else:
                    grouped_rois.append(current_group)
                    current_group = [(x, y, w, h)]

            if current_group:
                grouped_rois.append(current_group)

            merged_rois = []
            for group in grouped_rois:
                if len(group) == 1:
                    merged_rois.append(group[0])
                else:
                    heights = [h for x, y, w, h in group]
                    if max(heights) - min(heights) <= 100:
                        x_min = min([x for x, y, w, h in group])
                        y_min = min([y for x, y, w, h in group])
                        x_max = max([x + w for x, y, w, h in group])
                        y_max = max([y + h for x, y, w, h in group])
                        merged_rois.append((x_min, y_min, x_max - x_min, y_max - y_min))
                    else:
                        merged_rois.extend(group)
            
            filtered_rois = remove_nested_boxes(merged_rois)

            for x, y, w, h in filtered_rois:
                cv2.rectangle(image, (x-5, y-5), (x+w+5, y+h+5), (36, 255, 12), 2)

            for x, y, w, h in filtered_rois:
                roi = base_image[y:y+h, x:x+w]
                ocr_result = reader.readtext(roi)
                ocr_text = ' '.join([item[1] for item in ocr_result])
                if len(ocr_text) > 10 and ocr_text.count('\n\n') < 5 and ocr_text.count('|') <= 1:
                    ocr_text = re.sub(r'\s+', ' ', ocr_text)
                    ocr_text = ocr_text.replace('_', '.')
                    #ocr_text = re.sub(r'(?<!\n)(\d+\.\s+[A-Z])', r'\n\1', ocr_text)
                    #ocr_text = re.sub(r'(\s\d+\)\s[A-Z])', r'\n\n\1', ocr_text)
                    #ocr_text = re.sub(r'(\s[a-zA-Z]\.\s[A-Z])', r'\n\n\1', ocr_text)
                    #ocr_text = re.sub(r'(\s\d+\.\s[A-Z])', r'\n\n\1', ocr_text)
                    ocr_text = re.sub(r'\n{3,}', '', ocr_text)
                    ocr_results_all.append(ocr_text)

            logging.info(f"OCR halaman {i+1} selesai.")

        except Exception as e:
            logging.error(f"Error saat memproses halaman {i+1}: {e}")
            raise e

    end_time = time.time()
    elapsed_time = end_time - start_time
    logging.info(f"Waktu yang dibutuhkan: {elapsed_time:.2f} detik")
    logging.info("Eksekusi OCR selesai.")

    # Simpan hasil OCR ke file teks
    with open(output_path, "w") as f:
        f.write('\n\n'.join(ocr_results_all))