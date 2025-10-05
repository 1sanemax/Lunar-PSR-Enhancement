import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import torch
from torchvision import transforms
import numpy as np
from GAN_GUI import UNetGenerator

def calculate_psnr(img1, img2):
    """Calculate PSNR between two images"""
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(1.0 / np.sqrt(mse))

# Initialize device and model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
G = UNetGenerator(features=32).to(device)
G.load_state_dict(torch.load("generator_final.pth", map_location=device))
G.eval()

# Transform for the model - MUST match training resolution
transform = transforms.Compose([
    transforms.Resize((600, 600)),
    transforms.ToTensor(),
])

# Global variables to store the selected input and enhanced output images
selected_input_file = None
output_img_tk = None
img_tk = None
enhanced_img_tk = None

def enhance_image(img_path):
    """Process image through GAN model and return enhanced image - EXACTLY as testing.py does"""
    # Load noisy image
    img = Image.open(img_path).convert("L")
    
    noisy_tensor = transform(img).unsqueeze(0).to(device)
    
    # Enhance
    with torch.no_grad():
        enhanced = G(noisy_tensor)
    
    # Convert back to image - preserve full resolution
    inter_img = enhanced.squeeze(0).squeeze(0).cpu().numpy()
    
    # Denormalize from [-1, 1] to [0, 1]
    inter_img = (inter_img + 1) / 2.0
    inter_img = np.clip(inter_img, 0, 1)
    
    # Convert original image to numpy for fair comparison
    noisy_np = np.array(img.resize((600, 600))) / 255.0
    
    # Show difference - THIS IS WHAT testing.py DISPLAYS
    encd = np.abs(inter_img - noisy_np)
    
    # Convert to PIL Image
    enhanced_pil = Image.fromarray((encd * 255).astype(np.uint8))
    
    # Calculate quality metrics
    metrics = {
        'mean': encd.mean(),
        'std': encd.std(),
        'min': encd.min(),
        'max': encd.max(),
        'psnr': calculate_psnr(noisy_np, encd)
        }

        # Convert to PIL Image
    enhanced_pil = Image.fromarray((encd * 255).astype(np.uint8))

    return enhanced_pil, metrics

def insert_image():
    global img_tk, selected_file_label, selected_input_file
    # Specify the folder to restrict selection to
    #folder_path = r"D:\ARGON\TRAINING_DATA\noisy"
    file_path = filedialog.askopenfilename(
        parent=root,
        title="Select an Image",
        filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")]
    )
    if file_path:
        # Display the name of the selected file
        file_name = file_path.split("/")[-1]
        selected_file_label.config(text=f"Selected File: {file_name}")
        # Save the selected file path
        selected_input_file = file_path
        # Load the image for the new window display
        img = Image.open(file_path)
        img.thumbnail((500, 500))
        img_tk = ImageTk.PhotoImage(img) 
        enhance_button.config(state=tk.NORMAL)

def open_new_window():
    global output_img_tk, enhanced_img_tk, img_tk
    
    # Show processing message
    enhance_button.config(text="Processing...", state=tk.DISABLED)
    root.update()
    
    # Process the image through the GAN model
    try:
        enhanced_image, metrics = enhance_image(selected_input_file)
        # Resize both images to the same size (500x500)
        enhanced_image = enhanced_image.resize((500, 500), Image.LANCZOS)
        enhanced_img_tk = ImageTk.PhotoImage(enhanced_image)
        
        # Also resize the input image to match
        input_img = Image.open(selected_input_file).convert("L")
        input_img = input_img.resize((500, 500), Image.LANCZOS)
        img_tk = ImageTk.PhotoImage(input_img)
    except Exception as e:
        print(f"Error enhancing image: {e}")
        import traceback
        traceback.print_exc()
        enhance_button.config(text="Enhance Image ⚙", state=tk.NORMAL)
        return
    
    # Reset button
    enhance_button.config(text="Enhance Image ⚙", state=tk.NORMAL)
    
    root.withdraw()
    new_window = tk.Toplevel()
    new_window.title("Generated Output")
    new_window.attributes("-fullscreen", True)
    new_window.bind("<Escape>", lambda e: new_window.attributes("-fullscreen", False))
    
    # Load the PNG icon
    icon = tk.PhotoImage(file="oss.png")
    new_window.iconphoto(False, icon)
    new_window.configure(bg="#f5f5f5")

    # Header for the new window
    new_header = tk.Frame(new_window, bg="black", padx=20, pady=15)
    new_header.pack(fill=tk.X)

    new_logo_label = tk.Label(new_header, text="mo•n", font=("Arial", 24, "bold"), fg="white", bg="black")
    new_logo_label.pack(side=tk.LEFT)

    new_toggle_container = tk.Frame(new_header, bg="black")
    new_toggle_container.pack(side=tk.RIGHT)

    new_toggle_button = tk.Checkbutton(new_toggle_container, text="ORION SPACE SYSTEMS", bg="black", fg="white", font=("Arial", 12))
    new_toggle_button.pack(side=tk.RIGHT)

    # Main content in the new window
    new_main_frame = tk.Frame(new_window, bg="#f5f5f5", pady=40)
    new_main_frame.pack(fill=tk.BOTH, expand=True)

    # Canvas for placing images
    canvas = tk.Canvas(new_main_frame, width=500, height=500, bg="#f5f5f5")
    canvas.pack(fill=tk.BOTH, expand=True)

    # Display the left image (uploaded image)
    if img_tk:
        img_label_in_new_window = tk.Label(canvas, image=img_tk, bg="#f5f5f5")
        canvas.create_window(220, 10, anchor=tk.NW, window=img_label_in_new_window)

    # Display the right image (enhanced output - the difference image from testing.py)
    if enhanced_img_tk:
        output_img_label = tk.Label(canvas, image=enhanced_img_tk, bg="#f5f5f5")
        canvas.create_window(750, 10, anchor=tk.NW, window=output_img_label)

    # Center the "Close and Return" button below the images - lower and wider
    close_button = tk.Button(new_main_frame, text="Return", bg="#5e5ce6", fg="white", padx=50, pady=20, borderwidth=10, command=lambda: (new_window.destroy(), root.deiconify()))
    canvas.create_window(575, 580, anchor=tk.CENTER, window=close_button)
    
    title_labe = tk.Label(new_main_frame, text="INPUT", font=("Arial", 24), bg="#ffffff", fg="#0a0a0a")
    canvas.create_window(400, -20, anchor=tk.CENTER, window=title_labe)

    title_labe1 = tk.Label(new_main_frame, text="OUTPUT", font=("Arial", 24), bg="#ffffff", fg="#0a0a0a")
    canvas.create_window(950, -20, anchor=tk.CENTER, window=title_labe1)
    
    # Display Quality Assessment Metrics
    metrics_text = f"Quality Assessment:\n"
    metrics_text += f"Mean: {metrics['mean']:.4f}  |  Std: {metrics['std']:.4f}\n"
    metrics_text += f"Min: {metrics['min']:.4f}  |  Max: {metrics['max']:.4f}\n"
    metrics_text += f"PSNR (Noisy vs Enhanced): {metrics['psnr']:.4f} dB"

    metrics_label = tk.Label(new_main_frame, text=metrics_text, font=("Arial", 12), bg="#f5f5f5", fg="#333", justify=tk.LEFT)
    canvas.create_window(1075, 580, anchor=tk.CENTER, window=metrics_label)
    
    # Exit button for the new window - lower and wider
    exit_button = tk.Button(new_main_frame, text="Exit", bg="#ff0f0f", fg="white", padx=50, pady=20, borderwidth=10, command=close_app)
    canvas.create_window(775, 580, anchor=tk.CENTER, window=exit_button)

    # Configure the grid
    new_main_frame.grid_columnconfigure(0, weight=1)
    new_main_frame.grid_columnconfigure(1, weight=1)
    new_main_frame.grid_rowconfigure(0, weight=1)
    new_main_frame.grid_rowconfigure(1, weight=0)

def exit_fullscreen(event=None):
    root.attributes("-fullscreen", False)

def close_app(event=None):
    root.quit()

# Main window
root = tk.Tk()
root.title("Lunar Image Enhancement")
icon = tk.PhotoImage(file="oss.png")
root.iconphoto(False, icon)

# Load the background image
bg_image = Image.open("BG.jpg")
bg_width, bg_height = bg_image.size
bg_image = bg_image.resize((1100, 600), Image.LANCZOS)
bg_image_tk = ImageTk.PhotoImage(bg_image)

# Create a canvas to place the background image
canvas = tk.Canvas(root, width=1100, height=600)
canvas.pack(fill=tk.BOTH, expand=True)
canvas.create_image(0, 0, anchor=tk.NW, image=bg_image_tk)

# Title
title_label0 = tk.Label(root, text="ORION SPACE SYSTEMS", font=("Arial", 24), bg="#0a0a0a", fg="#ffffff")
canvas.create_window(311.5, 176.5, anchor=tk.CENTER, window=title_label0)
title_label = tk.Label(root, text="Lunar Image Enhancement th", font=("Arial", 24), bg="#ffffff", fg="#333")
canvas.create_window(255, 360, anchor=tk.CENTER, window=title_label)
title_label1 = tk.Label(root, text="rough ", font=("Arial", 24), bg="#f8f8f8", fg="#333")
canvas.create_window(511, 360, anchor=tk.CENTER, window=title_label1)
title_label2 = tk.Label(root, text="Driven Photon ", font=("Arial", 24), bg="#e9e9e9", fg="#333")
canvas.create_window(703.7, 360, anchor=tk.CENTER, window=title_label2)
title_label3 = tk.Label(root, text="and GAN-Based Denoising", font=("Arial", 24), bg="#ffffff", fg="#333")
canvas.create_window(235, 402, anchor=tk.CENTER, window=title_label3)
title_label4 = tk.Label(root, text="AI-", font=("Arial", 24), bg="#f2f2f2", fg="#333")
canvas.create_window(576.5, 360, anchor=tk.CENTER, window=title_label4)
title_label6 = tk.Label(root, text="Counting", font=("Arial", 24), bg="#e2e2e2", fg="#333")
canvas.create_window(870, 360, anchor=tk.CENTER, window=title_label6)

# Button container
insert_button = tk.Button(root, text="Insert Image", bg="#5e5ce6", fg="white", padx=33.5, pady=7, command=insert_image, borderwidth=0)
canvas.create_window(149.3, 455.5, anchor=tk.CENTER, window=insert_button)

# Label to display selected file name
selected_file_label = tk.Label(root, text="", bg="#f5f5f5", fg="#333", font=("Arial", 12))
canvas.create_window(150, 495, anchor=tk.CENTER, window=selected_file_label)

enhance_button = tk.Button(root, text="Enhance Image ⚙", bg="#ccc", fg="black", padx=40, pady=7, command=open_new_window, state=tk.DISABLED, borderwidth=0)
canvas.create_window(322, 455.5, anchor=tk.CENTER, window=enhance_button)

root.mainloop()