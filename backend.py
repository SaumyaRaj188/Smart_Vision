import torch
from PIL import Image
from ultralytics import YOLO
from collections import Counter
import pytesseract
import numpy as np
from torchvision import transforms
import google.generativeai as genai
import os
from dotenv import load_dotenv
load_dotenv()

#Initialize the gemini api
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-1.5-flash')

chat = model.start_chat(
    history=[
        {"role": "user", "parts": "You are a data extration AI, I will provide you with the raw OCR text extracted from a product's images, you need to extract all the meaningful information about the product like name, expiry, sno. etc and return in the format like, \nproduct_name: <name>\n name_property_1: <property_1>\n name_property_2: <property_2> etc. \n\n just return the product details no extra text required"},

        {"role": "model", "parts": "Okay, provide me with the raw OCR text"},
    ]
)

# Initialize the device and load the freshness model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
freshness_model_path = "freshness_detection.pt"  # Replace with your actual model path
freshness_model = torch.load(freshness_model_path)
freshness_model = freshness_model.to(device)
freshness_model.eval()

# Load YOLO model once
yolo_model_path = 'GRID6_detection_new.pt'  # Replace with your YOLO model path
yolo_model = YOLO(yolo_model_path)


def call_api(prompt:str):
    response = chat.send_message(prompt)
    response = response.text
    return response


# Define image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# Custom transformation function
def custom_transform(x, k=0.5):
    return k * (np.arcsin((2 * x - 1) ** 5) * (1 / np.pi) + 0.5) + (1 - k) * x


# Freshness function to process the image and get freshness score
def freshness(image_path):
    image = Image.open(image_path)
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = freshness_model(image_tensor)
        prediction = torch.sigmoid(output)
        sig = prediction.item()
        transformed_output = custom_transform(sig)
    
    return transformed_output * 100


def run_yolo_v8_detection(image_paths):
    detected_classes = []

    # Function to process results, extract class, and save cropped images
    def process_results(results, image_path):
        img = Image.open(image_path)
        for idx, result in enumerate(results):
            for box in result.boxes:
                class_idx = int(box.cls)
                class_name = yolo_model.names[class_idx]
                detected_classes.append(class_name)
                xmin, ymin, xmax, ymax = box.xyxy[0].tolist()
                cropped_img = img.crop((xmin, ymin, xmax, ymax))
                img_name = image_path.split("/")[-1]
                cropped_img_path = f"fresh_ocr/{img_name}"
                cropped_img.save(cropped_img_path)

    for image_path in image_paths:
        results = yolo_model(image_path)
        process_results(results, image_path)

    majority_class = Counter(detected_classes).most_common(1)[0][0]
    return majority_class


def ocr_text(img1, img2, img3, img4):
    image1 = Image.open(img1)
    image2 = Image.open(img2)
    image3 = Image.open(img3)
    image4 = Image.open(img4)
    text1 = pytesseract.image_to_string(image1)
    text2 = pytesseract.image_to_string(image2)
    text3 = pytesseract.image_to_string(image3)
    text4 = pytesseract.image_to_string(image4)
    combined_text = text1 + "\n" + text2 + "\n" + text3 + "\n" + text4

    return call_api(combined_text)


def process_images(class_name, img1, img2, img3, img4):
    if class_name not in ["potato", "onion", "banana", "apple"]:
        return ocr_text(img1, img2, img3, img4)
    else:
        freshness_score1 = freshness(img1)
        freshness_score2 = freshness(img2)
        freshness_score3 = freshness(img3)
        freshness_score4 = freshness(img4)
        avg_freshness = (freshness_score1 + freshness_score2 + freshness_score3 + freshness_score4) / 4
        return f"Average Freshness Score: {avg_freshness:.2f}%"




# Example usage
if __name__ == '__main__':
    image_paths = ['input/front.jpg', 'input/back.jpg', 'input/side1.jpg', 'input/side2.jpg']
    class_name = run_yolo_v8_detection(image_paths)
    
    # Use the cropped images for OCR or freshness checks
    img1 = "fresh_ocr/front.jpg"
    img2 = "fresh_ocr/back.jpg"
    img3 = "fresh_ocr/side1.jpg"
    img4 = "fresh_ocr/side2.jpg"
    result = process_images(class_name, img1, img2, img3, img4)
    
    print(f"Detected majority class: {class_name}")
    print(result)
