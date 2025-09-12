import cv2
import numpy as np
from paddleocr import PaddleOCR
import re
import time
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
from PIL import Image, ImageTk
import threading
import queue

class CameraOCRApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Camera OCR Recognition System")
        self.root.geometry("1200x800")
        self.root.resizable(True, True)
        
        # Initialize variables
        self.cap = None
        self.is_camera_running = False
        self.frame_queue = queue.Queue()
        self.ocr_results = []
        self.frame_count = 0
        self.start_time = time.time()
        
        # OCR configuration
        self.conf_threshold = 0.5
        self.ocr_frame_interval = 3  # Process OCR every N frames (reduced frequency for better performance)
        self.ocr_reader = None  # Initialize as None
        
        # Create GUI first (before initializing OCR)
        self.create_widgets()
        
        # Initialize PaddleOCR after GUI is created
        self.initialize_ocr()
        
        # Start GUI update loop
        self.update_gui()
    
    def initialize_ocr(self):
        """Initialize PaddleOCR"""
        self.log_message("🔄 Initializing PaddleOCR...")
        
        try:
            # Try local version first (requires pre-downloaded model files)
            self.ocr_reader = PaddleOCR(
                use_textline_orientation=False, 
                lang="en",   
                text_detection_model_name="PP-OCRv5_mobile_det",
                text_recognition_model_name="PP-OCRv5_mobile_rec",
                text_detection_model_dir=r".\PP-OCRv5_mobile_det",
                text_recognition_model_dir=r".\PP-OCRv5_mobile_rec",
            )
            self.log_message("✅ PaddleOCR initialized successfully (local version)")
            return
        except Exception as e:
            self.log_message(f"⚠️ Local OCR initialization failed: {str(e)[:100]}...")
        
        try:
            # Fallback to server version
            self.ocr_reader = PaddleOCR(use_textline_orientation=True, lang='en')
            self.log_message("✅ PaddleOCR initialized successfully (server version)")
            return
        except Exception as e2:
            self.ocr_reader = None
            self.log_message(f"❌ Failed to initialize PaddleOCR: {str(e2)[:100]}...")
            self.log_message("Please check if PaddleOCR is properly installed")
    
    def log_message(self, message):
        """Add message to history log"""
        timestamp = time.strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {message}\n"
        
        # Check if GUI components exist before trying to use them
        if hasattr(self, 'history_text') and self.history_text is not None:
            try:
                self.history_text.insert(tk.END, formatted_message)
                self.history_text.see(tk.END)
                return
            except:
                pass
        
        # If GUI isn't ready or fails, print to console
        print(formatted_message.strip())
    
    def create_widgets(self):
        """Create GUI widgets"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Control panel (left side)
        control_frame = ttk.LabelFrame(main_frame, text="Camera Controls", padding="10")
        control_frame.grid(row=0, column=0, rowspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        
        # Camera control buttons
        self.btn_start = ttk.Button(control_frame, text="Start Camera", command=self.start_camera)
        self.btn_start.pack(fill=tk.X, pady=5)
        
        self.btn_stop = ttk.Button(control_frame, text="Stop Camera", command=self.stop_camera, state=tk.DISABLED)
        self.btn_stop.pack(fill=tk.X, pady=5)
        
        self.btn_screenshot = ttk.Button(control_frame, text="Take Screenshot", command=self.take_screenshot, state=tk.DISABLED)
        self.btn_screenshot.pack(fill=tk.X, pady=5)
        
        # Camera settings
        ttk.Separator(control_frame, orient='horizontal').pack(fill=tk.X, pady=10)
        
        ttk.Label(control_frame, text="Camera Index:").pack(anchor=tk.W)
        self.camera_index_var = tk.StringVar(value="0")
        camera_index_entry = ttk.Entry(control_frame, textvariable=self.camera_index_var, width=10)
        camera_index_entry.pack(fill=tk.X, pady=5)
        
        ttk.Label(control_frame, text="OCR Confidence Threshold:").pack(anchor=tk.W)
        self.conf_threshold_var = tk.DoubleVar(value=0.5)
        conf_scale = ttk.Scale(control_frame, from_=0.1, to=1.0, variable=self.conf_threshold_var, orient=tk.HORIZONTAL)
        conf_scale.pack(fill=tk.X, pady=5)
        
        self.conf_label = ttk.Label(control_frame, text="0.5")
        self.conf_label.pack()
        conf_scale.configure(command=self.update_conf_label)
        
        # Status information
        ttk.Separator(control_frame, orient='horizontal').pack(fill=tk.X, pady=10)
        
        self.status_frame = ttk.LabelFrame(control_frame, text="Status", padding="5")
        self.status_frame.pack(fill=tk.X, pady=5)
        
        self.fps_label = ttk.Label(self.status_frame, text="FPS: 0.00")
        self.fps_label.pack(anchor=tk.W)
        
        self.frame_label = ttk.Label(self.status_frame, text="Frame: 0")
        self.frame_label.pack(anchor=tk.W)
        
        self.camera_status_label = ttk.Label(self.status_frame, text="Camera: Stopped", foreground="red")
        self.camera_status_label.pack(anchor=tk.W)
        
        # OCR status
        self.ocr_status_label = ttk.Label(self.status_frame, text="OCR: Initializing...", foreground="orange")
        self.ocr_status_label.pack(anchor=tk.W)
        
        # Video display (top right)
        video_frame = ttk.LabelFrame(main_frame, text="Camera Feed", padding="10")
        video_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        video_frame.columnconfigure(0, weight=1)
        video_frame.rowconfigure(0, weight=1)
        
        self.video_label = ttk.Label(video_frame, text="Camera feed will appear here", anchor=tk.CENTER)
        self.video_label.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # OCR results display (bottom right)
        results_frame = ttk.LabelFrame(main_frame, text="OCR Recognition Results", padding="10")
        results_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(10, 0))
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(0, weight=1)
        
        # Create notebook for different result views
        self.notebook = ttk.Notebook(results_frame)
        self.notebook.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Real-time results tab
        realtime_frame = ttk.Frame(self.notebook)
        self.notebook.add(realtime_frame, text="Real-time Results")
        
        self.realtime_text = scrolledtext.ScrolledText(realtime_frame, wrap=tk.WORD, height=8)
        self.realtime_text.pack(fill=tk.BOTH, expand=True)
        
        # History tab
        history_frame = ttk.Frame(self.notebook)
        self.notebook.add(history_frame, text="History Log")
        
        self.history_text = scrolledtext.ScrolledText(history_frame, wrap=tk.WORD, height=8)
        self.history_text.pack(fill=tk.BOTH, expand=True)
        
        # Clear buttons
        button_frame = ttk.Frame(results_frame)
        button_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(5, 0))
        
        ttk.Button(button_frame, text="Clear Real-time", command=self.clear_realtime).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Clear History", command=self.clear_history).pack(side=tk.LEFT, padx=5)
        
        # Now that GUI is created, we can safely use log_message
        self.log_message("🖥️ GUI interface initialized successfully")
    
    def update_conf_label(self, value):
        """Update confidence threshold label"""
        self.conf_label.config(text=f"{float(value):.2f}")
        self.conf_threshold = float(value)
    
    def update_ocr_status(self):
        """Update OCR status label"""
        if self.ocr_reader is not None:
            self.ocr_status_label.config(text="OCR: Ready", foreground="green")
        else:
            self.ocr_status_label.config(text="OCR: Failed", foreground="red")
    
    def start_camera(self):
        """Start camera capture"""
        if self.is_camera_running:
            return
        
        try:
            camera_index = int(self.camera_index_var.get())
            self.cap = cv2.VideoCapture(camera_index)
            
            if not self.cap.isOpened():
                raise Exception(f"Cannot open camera with index {camera_index}")
            
            self.is_camera_running = True
            self.frame_count = 0
            self.start_time = time.time()
            
            # Update UI
            self.btn_start.config(state=tk.DISABLED)
            self.btn_stop.config(state=tk.NORMAL)
            self.btn_screenshot.config(state=tk.NORMAL)
            self.camera_status_label.config(text="Camera: Running", foreground="green")
            
            # Start camera thread
            self.camera_thread = threading.Thread(target=self.camera_loop, daemon=True)
            self.camera_thread.start()
            
            self.log_message("✅ Camera started successfully")
            
        except Exception as e:
            messagebox.showerror("Camera Error", f"Failed to start camera: {e}")
            self.log_message(f"❌ Failed to start camera: {e}")
    
    def stop_camera(self):
        """Stop camera capture"""
        if not self.is_camera_running:
            return
        
        self.is_camera_running = False
        
        if self.cap:
            self.cap.release()
            self.cap = None
        
        # Update UI
        self.btn_start.config(state=tk.NORMAL)
        self.btn_stop.config(state=tk.DISABLED)
        self.btn_screenshot.config(state=tk.DISABLED)
        self.camera_status_label.config(text="Camera: Stopped", foreground="red")
        self.video_label.config(image="", text="Camera feed will appear here")
        
        self.log_message("🛑 Camera stopped")
    
    def take_screenshot(self):
        """Take screenshot of current frame"""
        if not self.is_camera_running or self.frame_queue.empty():
            return
        
        try:
            # Get current frame from queue
            current_frame = None
            while not self.frame_queue.empty():
                current_frame = self.frame_queue.get()
            
            if current_frame is not None:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"screenshot_{timestamp}.jpg"
                cv2.imwrite(filename, current_frame)
                self.log_message(f"📷 Screenshot saved: {filename}")
                messagebox.showinfo("Screenshot", f"Screenshot saved as {filename}")
            
        except Exception as e:
            self.log_message(f"❌ Screenshot failed: {e}")
            messagebox.showerror("Screenshot Error", f"Failed to save screenshot: {e}")
    
    def camera_loop(self):
        """Camera capture loop (runs in separate thread)"""
        while self.is_camera_running:
            ret, frame = self.cap.read()
            if not ret:
                self.log_message("❌ Failed to read camera frame")
                break
            
            self.frame_count += 1
            
            # Put frame in queue for GUI update
            if not self.frame_queue.full():
                self.frame_queue.put(frame.copy())
            
            # Process OCR every N frames
            if self.frame_count % self.ocr_frame_interval == 0:
                self.process_ocr(frame)
            
            time.sleep(0.03)  # ~30 FPS
    
    def process_ocr(self, frame):
        """Process OCR on current frame"""
        if self.ocr_reader is None:
            return
        
        try:
            # Edge detection to find largest quadrilateral
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blur, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            max_area = 0
            best_approx = None
            for cnt in contours:
                approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
                if len(approx) == 4:
                    area = cv2.contourArea(approx)
                    if area > 5000:  # Minimum area threshold
                        rect = cv2.boundingRect(approx)
                        aspect_ratio = rect[2] / rect[3]
                        if 0.3 < aspect_ratio < 3.0:  # Aspect ratio constraint
                            if area > max_area:
                                max_area = area
                                best_approx = approx
            
            # Process OCR on detected region
            if best_approx is not None:
                x, y, w, h = cv2.boundingRect(best_approx)
                roi = frame[y:y+h, x:x+w]
                
                if roi.size > 0:
                    results = self.ocr_reader.predict(roi)
                    detected_texts = []
                    
                    for page in results:
                        rec_texts = page.get('rec_texts', [])
                        rec_scores = page.get('rec_scores', [])
                        for text, score in zip(rec_texts, rec_scores):
                            if score >= self.conf_threshold:
                                clean_text = re.sub(r'[^A-Za-z0-9\s]', '', text).strip()
                                if clean_text:
                                    detected_texts.append((clean_text, score))
                    
                    # Update OCR results
                    if detected_texts:
                        self.ocr_results = detected_texts
                        self.update_realtime_results(detected_texts)
        
        except Exception as e:
            self.log_message(f"OCR processing error: {e}")
    
    def update_realtime_results(self, results):
        """Update real-time OCR results display"""
        def update_text():
            try:
                self.realtime_text.delete(1.0, tk.END)
                timestamp = time.strftime("%H:%M:%S")
                self.realtime_text.insert(tk.END, f"Last updated: {timestamp}\n")
                self.realtime_text.insert(tk.END, "-" * 50 + "\n")
                
                for i, (text, confidence) in enumerate(results, 1):
                    result_line = f"{i:2d}. {text:<20} (Confidence: {confidence:.3f})\n"
                    self.realtime_text.insert(tk.END, result_line)
                
                self.realtime_text.see(tk.END)
            except Exception as e:
                print(f"Error updating realtime results: {e}")
        
        # Schedule GUI update in main thread
        self.root.after(0, update_text)
    
    def update_gui(self):
        """Update GUI elements (called periodically)"""
        # Update OCR status if it changed
        self.update_ocr_status()
        
        # Update video display
        if self.is_camera_running and not self.frame_queue.empty():
            try:
                frame = self.frame_queue.get()
                
                # Resize frame for display
                display_frame = cv2.resize(frame, (640, 480))
                display_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                
                # Convert to PIL Image and then to PhotoImage
                pil_image = Image.fromarray(display_frame)
                photo = ImageTk.PhotoImage(image=pil_image)
                
                self.video_label.config(image=photo, text="")
                self.video_label.image = photo  # Keep a reference
            except Exception as e:
                print(f"Error updating video display: {e}")
        
        # Update status information
        if self.is_camera_running:
            elapsed_time = time.time() - self.start_time
            fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
            self.fps_label.config(text=f"FPS: {fps:.2f}")
            self.frame_label.config(text=f"Frame: {self.frame_count}")
        
        # Schedule next update
        self.root.after(33, self.update_gui)  # ~30 FPS
    
    def clear_realtime(self):
        """Clear real-time results"""
        self.realtime_text.delete(1.0, tk.END)
    
    def clear_history(self):
        """Clear history log"""
        self.history_text.delete(1.0, tk.END)
    
    def on_closing(self):
        """Handle application closing"""
        self.stop_camera()
        self.root.destroy()

def main():
    root = tk.Tk()
    app = CameraOCRApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()