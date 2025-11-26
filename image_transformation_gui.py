# A GUI component, takes an image, asks for the technique and its intensity to apply, provides the processed image which can be used to test the model
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import os
from pathlib import Path

class ImageProcessorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Processing Technique Selector")
        self.root.geometry("1400x900")
        
        self.original_image = None
        self.processed_image = None
        self.original_path = None
        self.display_size = (400, 400)
        
        # Available techniques with default intensity
        self.techniques = {
            'Gaussian Blur': {'selected': False, 'intensity': 50, 'min': 1, 'max': 100},
            'Median Blur': {'selected': False, 'intensity': 50, 'min': 1, 'max': 100},
            'Sharpen': {'selected': False, 'intensity': 50, 'min': 1, 'max': 100},
            'Edge Detection': {'selected': False, 'intensity': 50, 'min': 1, 'max': 100},
            'Brightness Increase': {'selected': False, 'intensity': 50, 'min': 1, 'max': 100},
            'Brightness Decrease': {'selected': False, 'intensity': 50, 'min': 1, 'max': 100},
            'Contrast Increase': {'selected': False, 'intensity': 50, 'min': 1, 'max': 100},
            'Contrast Decrease': {'selected': False, 'intensity': 50, 'min': 1, 'max': 100},
            'Histogram Equalization': {'selected': False, 'intensity': 50, 'min': 1, 'max': 100},
            'Gamma Correction (Bright)': {'selected': False, 'intensity': 50, 'min': 1, 'max': 100},
            'Gamma Correction (Dark)': {'selected': False, 'intensity': 50, 'min': 1, 'max': 100},
            'Bilateral Filter': {'selected': False, 'intensity': 50, 'min': 1, 'max': 100},
            'Rotation': {'selected': False, 'intensity': 50, 'min': 1, 'max': 100},
            'Gaussian Noise': {'selected': False, 'intensity': 50, 'min': 1, 'max': 100},
            'Salt & Pepper Noise': {'selected': False, 'intensity': 50, 'min': 1, 'max': 100},
            'Box Blur': {'selected': False, 'intensity': 50, 'min': 1, 'max': 100},
            'Motion Blur': {'selected': False, 'intensity': 50, 'min': 1, 'max': 100},
            'Emboss': {'selected': False, 'intensity': 50, 'min': 1, 'max': 100},
            'Sepia': {'selected': False, 'intensity': 50, 'min': 1, 'max': 100},
            'Invert Colors': {'selected': False, 'intensity': 50, 'min': 1, 'max': 100},
        }
        
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the GUI layout"""
        
        # Top control panel
        control_frame = ttk.Frame(self.root, padding="10")
        control_frame.grid(row=0, column=0, columnspan=3, sticky=(tk.W, tk.E))
        
        ttk.Button(control_frame, text="Load Image", command=self.load_image, 
                  width=15).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Apply Techniques", command=self.apply_techniques, 
                  width=15).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Save Processed", command=self.save_processed, 
                  width=15).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Reset All", command=self.reset_all, 
                  width=15).pack(side=tk.LEFT, padx=5)
        
        # Status label
        self.status_label = ttk.Label(control_frame, text="Load an image to begin", 
                                     foreground="blue")
        self.status_label.pack(side=tk.LEFT, padx=20)
        
        # Main content area
        main_frame = ttk.Frame(self.root)
        main_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Left: Technique selection panel with scrollbar
        left_frame = ttk.LabelFrame(main_frame, text="Image Processing Techniques", 
                                    padding="10")
        left_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
        
        # Canvas and scrollbar for techniques
        canvas = tk.Canvas(left_frame, width=450, height=700)
        scrollbar = ttk.Scrollbar(left_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Create technique controls
        self.technique_vars = {}
        self.intensity_vars = {}
        self.intensity_labels = {}
        
        for i, (name, config) in enumerate(self.techniques.items()):
            frame = ttk.Frame(scrollable_frame)
            frame.pack(fill=tk.X, pady=5, padx=5)
            
            # Checkbox
            var = tk.BooleanVar(value=config['selected'])
            self.technique_vars[name] = var
            chk = ttk.Checkbutton(frame, text=name, variable=var, 
                                 command=self.on_technique_change)
            chk.pack(anchor=tk.W)
            
            # Intensity slider frame
            slider_frame = ttk.Frame(frame)
            slider_frame.pack(fill=tk.X, padx=20)
            
            ttk.Label(slider_frame, text="Intensity:").pack(side=tk.LEFT)
            
            intensity_var = tk.IntVar(value=config['intensity'])
            self.intensity_vars[name] = intensity_var
            
            slider = ttk.Scale(slider_frame, from_=config['min'], to=config['max'],
                             variable=intensity_var, orient=tk.HORIZONTAL,
                             command=lambda v, n=name: self.update_intensity_label(n))
            slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
            
            intensity_label = ttk.Label(slider_frame, text=f"{config['intensity']}%", 
                                       width=6)
            intensity_label.pack(side=tk.LEFT)
            self.intensity_labels[name] = intensity_label
        
        # Middle: Original image
        middle_frame = ttk.LabelFrame(main_frame, text="Original Image", padding="10")
        middle_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
        
        self.original_label = ttk.Label(middle_frame, text="No image loaded", 
                                       relief=tk.SUNKEN, anchor=tk.CENTER)
        self.original_label.pack(expand=True, fill=tk.BOTH)
        
        # Right: Processed image
        right_frame = ttk.LabelFrame(main_frame, text="Processed Image", padding="10")
        right_frame.grid(row=0, column=2, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
        
        self.processed_label = ttk.Label(right_frame, text="Apply techniques to see result", 
                                        relief=tk.SUNKEN, anchor=tk.CENTER)
        self.processed_label.pack(expand=True, fill=tk.BOTH)
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.columnconfigure(1, weight=2)
        self.root.columnconfigure(2, weight=2)
        self.root.rowconfigure(1, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=2)
        main_frame.columnconfigure(2, weight=2)
        main_frame.rowconfigure(0, weight=1)
    
    def update_intensity_label(self, technique_name):
        """Update intensity percentage label"""
        intensity = self.intensity_vars[technique_name].get()
        self.intensity_labels[technique_name].config(text=f"{intensity}%")
    
    def on_technique_change(self):
        """Handle technique selection change"""
        selected = [name for name, var in self.technique_vars.items() if var.get()]
        if selected:
            self.status_label.config(text=f"Selected: {len(selected)} technique(s)", 
                                    foreground="green")
        else:
            self.status_label.config(text="No techniques selected", 
                                    foreground="orange")
    
    def load_image(self):
        """Load an image file"""
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"), 
                      ("All files", "*.*")]
        )
        
        if file_path:
            self.original_path = file_path
            self.original_image = cv2.imread(file_path)
            
            if self.original_image is None:
                messagebox.showerror("Error", "Could not load image!")
                return
            
            # Display original image
            self.display_image(self.original_image, self.original_label)
            self.status_label.config(text=f"Loaded: {Path(file_path).name}", 
                                    foreground="green")
            
            # Clear processed image
            self.processed_image = None
            self.processed_label.config(image='', text="Apply techniques to see result")
    
    def display_image(self, cv_image, label):
        """Display OpenCV image in tkinter label"""
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        
        # Resize to fit display
        h, w = rgb_image.shape[:2]
        aspect = w / h
        
        if aspect > 1:
            new_w = self.display_size[0]
            new_h = int(new_w / aspect)
        else:
            new_h = self.display_size[1]
            new_w = int(new_h * aspect)
        
        resized = cv2.resize(rgb_image, (new_w, new_h))
        
        # Convert to PIL and then to ImageTk
        pil_image = Image.fromarray(resized)
        photo = ImageTk.PhotoImage(pil_image)
        
        label.config(image=photo, text='')
        label.image = photo  # Keep a reference
    
    def apply_technique_to_image(self, img, technique_name, intensity):
        """Apply a single technique with specified intensity"""
        # Normalize intensity to 0-1 range
        alpha = intensity / 100.0
        
        if technique_name == 'Gaussian Blur':
            kernel_size = int(3 + (alpha * 18))  # 3 to 21
            if kernel_size % 2 == 0:
                kernel_size += 1
            blurred = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
            return cv2.addWeighted(img, 1-alpha, blurred, alpha, 0)
        
        elif technique_name == 'Median Blur':
            kernel_size = int(3 + (alpha * 18))  # 3 to 21
            if kernel_size % 2 == 0:
                kernel_size += 1
            blurred = cv2.medianBlur(img, kernel_size)
            return cv2.addWeighted(img, 1-alpha, blurred, alpha, 0)
        
        elif technique_name == 'Sharpen':
            strength = alpha * 2  # 0 to 2
            kernel = np.array([[-1,-1,-1], [-1, 9,-1], [-1,-1,-1]]) * strength
            kernel[1,1] = 9
            sharpened = cv2.filter2D(img, -1, kernel)
            return np.clip(sharpened, 0, 255).astype(np.uint8)
        
        elif technique_name == 'Edge Detection':
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            threshold1 = int(50 + alpha * 100)
            threshold2 = int(100 + alpha * 150)
            edges = cv2.Canny(gray, threshold1, threshold2)
            edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            return cv2.addWeighted(img, 1-alpha, edges_colored, alpha, 0)
        
        elif technique_name == 'Brightness Increase':
            beta = int(alpha * 100)  # 0 to 100
            return cv2.convertScaleAbs(img, alpha=1.0, beta=beta)
        
        elif technique_name == 'Brightness Decrease':
            beta = -int(alpha * 100)  # 0 to -100
            return cv2.convertScaleAbs(img, alpha=1.0, beta=beta)
        
        elif technique_name == 'Contrast Increase':
            contrast = 1.0 + (alpha * 2.0)  # 1.0 to 3.0
            return cv2.convertScaleAbs(img, alpha=contrast, beta=0)
        
        elif technique_name == 'Contrast Decrease':
            contrast = 1.0 - (alpha * 0.8)  # 1.0 to 0.2
            return cv2.convertScaleAbs(img, alpha=contrast, beta=0)
        
        elif technique_name == 'Histogram Equalization':
            ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
            ycrcb[:,:,0] = cv2.equalizeHist(ycrcb[:,:,0])
            equalized = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
            return cv2.addWeighted(img, 1-alpha, equalized, alpha, 0)
        
        elif technique_name == 'Gamma Correction (Bright)':
            gamma = 0.5 + (alpha * 1.0)  # 0.5 to 1.5
            inv_gamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255 
                            for i in range(256)]).astype("uint8")
            return cv2.LUT(img, table)
        
        elif technique_name == 'Gamma Correction (Dark)':
            gamma = 1.5 + (alpha * 1.5)  # 1.5 to 3.0
            inv_gamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255 
                            for i in range(256)]).astype("uint8")
            return cv2.LUT(img, table)
        
        elif technique_name == 'Bilateral Filter':
            d = int(5 + alpha * 10)  # 5 to 15
            sigma = 50 + alpha * 100  # 50 to 150
            filtered = cv2.bilateralFilter(img, d, sigma, sigma)
            return cv2.addWeighted(img, 1-alpha, filtered, alpha, 0)
        
        elif technique_name == 'Rotation':
            angle = alpha * 360  # 0 to 360 degrees
            h, w = img.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            return cv2.warpAffine(img, M, (w, h))
        
        elif technique_name == 'Gaussian Noise':
            noise_level = alpha * 50  # 0 to 50
            noise = np.random.normal(0, noise_level, img.shape).astype(np.int16)
            noisy = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            return noisy
        
        elif technique_name == 'Salt & Pepper Noise':
            prob = alpha * 0.05  # 0 to 5%
            noisy = img.copy()
            rnd = np.random.rand(img.shape[0], img.shape[1])
            noisy[rnd < prob/2] = 0
            noisy[rnd > 1 - prob/2] = 255
            return noisy
        
        elif technique_name == 'Box Blur':
            kernel_size = int(3 + (alpha * 28))  # 3 to 31
            if kernel_size % 2 == 0:
                kernel_size += 1
            blurred = cv2.blur(img, (kernel_size, kernel_size))
            return cv2.addWeighted(img, 1-alpha, blurred, alpha, 0)
        
        elif technique_name == 'Motion Blur':
            size = int(5 + alpha * 25)  # 5 to 30
            kernel = np.zeros((size, size))
            kernel[int((size-1)/2), :] = np.ones(size)
            kernel = kernel / size
            blurred = cv2.filter2D(img, -1, kernel)
            return cv2.addWeighted(img, 1-alpha, blurred, alpha, 0)
        
        elif technique_name == 'Emboss':
            kernel = np.array([[-2,-1,0], [-1,1,1], [0,1,2]]) * alpha
            embossed = cv2.filter2D(img, -1, kernel)
            embossed = cv2.convertScaleAbs(embossed)
            return cv2.addWeighted(img, 1-alpha, embossed, alpha, 0)
        
        elif technique_name == 'Sepia':
            kernel = np.array([[0.272, 0.534, 0.131],
                             [0.349, 0.686, 0.168],
                             [0.393, 0.769, 0.189]])
            sepia = cv2.transform(img, kernel)
            sepia = np.clip(sepia, 0, 255).astype(np.uint8)
            return cv2.addWeighted(img, 1-alpha, sepia, alpha, 0)
        
        elif technique_name == 'Invert Colors':
            inverted = cv2.bitwise_not(img)
            return cv2.addWeighted(img, 1-alpha, inverted, alpha, 0)
        
        return img
    
    def apply_techniques(self):
        """Apply selected techniques to the original image"""
        if self.original_image is None:
            messagebox.showwarning("Warning", "Please load an image first!")
            return
        
        # Get selected techniques
        selected = [(name, self.intensity_vars[name].get()) 
                   for name, var in self.technique_vars.items() if var.get()]
        
        if not selected:
            messagebox.showwarning("Warning", "Please select at least one technique!")
            return
        
        # Apply techniques sequentially
        result = self.original_image.copy()
        
        for technique_name, intensity in selected:
            result = self.apply_technique_to_image(result, technique_name, intensity)
        
        self.processed_image = result
        self.display_image(result, self.processed_label)
        
        technique_names = [name for name, _ in selected]
        self.status_label.config(
            text=f"Applied {len(selected)} technique(s): {', '.join(technique_names)}", 
            foreground="green"
        )
    
    def save_processed(self):
        """Save the processed image"""
        if self.processed_image is None:
            messagebox.showwarning("Warning", "No processed image to save!")
            return
        
        if self.original_path is None:
            messagebox.showwarning("Warning", "Original image path not found!")
            return
        
        # Generate output filename
        original_path = Path(self.original_path)
        output_name = f"{original_path.stem}_processed{original_path.suffix}"
        output_path = original_path.parent / output_name
        
        # Ask for save location
        save_path = filedialog.asksaveasfilename(
            title="Save Processed Image",
            initialfile=output_name,
            defaultextension=original_path.suffix,
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp"), 
                      ("All files", "*.*")]
        )
        
        if save_path:
            cv2.imwrite(save_path, self.processed_image)
            messagebox.showinfo("Success", f"Image saved to:\n{save_path}")
            self.status_label.config(text=f"Saved: {Path(save_path).name}", 
                                    foreground="blue")
    
    def reset_all(self):
        """Reset all selections and intensities"""
        for name, var in self.technique_vars.items():
            var.set(False)
            self.intensity_vars[name].set(50)
            self.intensity_labels[name].config(text="50%")
        
        self.processed_image = None
        if hasattr(self.processed_label, 'image'):
            self.processed_label.config(image='', text="Apply techniques to see result")
        
        self.status_label.config(text="All settings reset", foreground="blue")

def main():
    root = tk.Tk()
    app = ImageProcessorGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()