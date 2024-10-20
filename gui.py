import tkinter as tk
from tkinter import filedialog, messagebox
import os
from PIL import Image
import shutil

# Function to select and save images with specified names
def select_images():
    file_paths = filedialog.askopenfilenames(
        title="Select 4 images",
        filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
    )
    
    if len(file_paths) != 4:
        messagebox.showerror("Error", "Please select exactly 4 images.")
        return
    
    # Define names for the images
    image_names = ['front.jpg', 'back.jpg', 'side1.jpg', 'side2.jpg']
    
    # Create input directory if it doesn't exist
    os.makedirs('input', exist_ok=True)
    
    # Save the selected images in the input folder
    for file_path, image_name in zip(file_paths, image_names):
        img = Image.open(file_path)
        img.save(f"input/{image_name}")
    
    messagebox.showinfo("Success", "Images saved successfully!")
    process_and_display()

# Function to run the backend code and display results
def process_and_display():
    from backend import run_yolo_v8_detection, process_images  # Replace with your actual backend script name
    
    # Paths of the saved images
    image_paths = ['input/front.jpg', 'input/back.jpg', 'input/side1.jpg', 'input/side2.jpg']
    class_name = run_yolo_v8_detection(image_paths)
    
    # Get the result based on the class
    result = process_images(class_name, 'front.jpg', 'back.jpg', 'side1.jpg', 'side2.jpg')
    
    # Display the result in the GUI
    result_text.set(f"Class: {class_name}\n {result}")
    
    # Save the result to product_details.txt
    with open("product_details.txt", "w") as f:
        f.write(result)

    messagebox.showinfo("Process Complete", "Processing complete and details saved!")

# Create the Tkinter GUI
root = tk.Tk()
root.title("Product Detail Extraction")

# Create a button to select images
select_button = tk.Button(root, text="Select 4 Images", command=select_images)
select_button.pack(pady=10)

# Display the result
result_text = tk.StringVar()
result_label = tk.Label(root, textvariable=result_text, wraplength=400)
result_label.pack(pady=10)

# Run the Tkinter main loop
root.mainloop()