# Flipkart GRID 6.0 Robotics Track: Smart Vision

This repository contains two Python scripts, `backend.py` and `gui.py`, that work together to detect, classify, and extract details from product images using YOLO, OCR, and a freshness detection AI model. The GUI allows users to select images and view extracted product details.

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [Modules Used](#modules-used)

## Overview

### `backend.py`
This script handles the core processing, including:
- **YOLO Detection**: Uses the YOLOv8 model to detect objects in provided images and extracts the detected region.
- **OCR Processing**: Uses `pytesseract` to extract text from the images. This raw OCR text data is then processed using an NLP model `google's gemini`
- **Freshness Detection**: A neural network model is used to estimate the freshness score of certain detected classes (like fruits or vegetables).
- **Data Augmentation**: Images are preprocessed using transformations, including resizing, color jittering, and noise addition.

### `gui.py`
This script provides a graphical user interface (GUI) for the project:
- Allows users to select four images (front, back, and sides).
- Saves the selected images and passes them to `backend.py` for processing.
- Displays the detected class and extracted details in the GUI.
- Saves the results in `product_details.txt`.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/product-detail-extraction.git
   cd product-detail-extraction
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   
4. Ensure that pytesseract is correctly installed and set up on your system. Refer to the [Tesseract installation guide](https://github.com/tesseract-ocr/tesseract) if needed.

5. Set up the `.env` file with your Gemini API key:
   ```
   GEMINI_API_KEY=your_api_key_here
   ```

## Usage

1. Run the GUI:
   ```bash
   python gui.py
   ```

2. Use the GUI to select four images (front, back, side1, side2) of a product.

3. The backend will process the images, detect the product class, and extract details. Results will be displayed in the GUI and saved to `product_details.txt`.


## Modules Used

### Python Libraries
- **torch, torchvision**: For loading and using the freshness detection model.
- **ultralytics**: To load the YOLOv8 model and perform object detection.
- **pytesseract**: For extracting text from the detected regions.
- **numpy**: For numerical operations.
- **Pillow (PIL)**: For image manipulation, including loading and saving images.
- **google-generativeai**: To interface with the Gemini API for parsing OCR results.
- **python-dotenv**: For loading environment variables like the Gemini API key.
- **tkinter**: For creating the graphical user interface.

---
