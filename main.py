import os
import cv2
import numpy as np
import pytesseract
from dotenv import load_dotenv
from google.cloud import vision
import google.generativeai as genai
from FotoYakalama import FotoYakalama

pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

load_dotenv()
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
os.environ['GOOGLE_API_KEY'] = GOOGLE_API_KEY
genai.configure(api_key=GOOGLE_API_KEY)


def capture_photo():
    print("Fotoğraf çekmek için 'c' tuşuna basın, çıkmak için 'q' tuşuna basın.")
    foto_yakalama_obj = FotoYakalama()
    return foto_yakalama_obj.foto_yakalama()


def detect_text(image_path):
    client = vision.ImageAnnotatorClient()
    with open(image_path, 'rb') as image_file:
        content = image_file.read()
    
    image = vision.Image(content=content)
    response = client.text_detection(image=image)
    
    texts = response.text_annotations
    text_from_img = ''.join([text.description for text in texts])
    
    return text_from_img


def generate_content(text):
    generation_config = {
        "candidate_count": 1,
        "temperature": 1,
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": 500,
        "response_mime_type": "text/plain"
    }
    
    model_name = "gemini-1.5-flash"
    model = genai.GenerativeModel(model_name=model_name, generation_config=generation_config)
    
    
    prompt = f"You are an expert reader. Please provide detailed information about the text extracted from the image: {text}"
    
  
    response = model.generate_content([prompt])
    return response.text

foto = capture_photo()

image_path = 'path/to/your/image.jpg' 
text_from_img = detect_text(image_path)


print("Tespit Edilen Metin:\n", text_from_img)


ai_response = generate_content(text_from_img)
print("AI answer:\n", ai_response)

cv2.waitKey(0)
cv2.destroyAllWindows()
