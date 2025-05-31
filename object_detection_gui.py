import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import subprocess
import threading
import sys  # To potentially use sys.executable
from PIL import Image as PILImage, ImageTk

# Use Resampling from Pillow 9.0.0+, otherwise fallback
try:
    from PIL.Image import Resampling

    LANCZOS = Resampling.LANCZOS
except ImportError:
    # Fallback for older Pillow versions
    LANCZOS = PILImage.LANCZOS


class YOLOv7GUI:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLOv7 Object Detector")
        # Start with a reasonable default size
        self.root.geometry("1200x700")
        self.root.minsize(800, 500)  # Set a minimum size
        self.root.configure(bg="#f0f0f0")  # Slightly lighter gray background

        # --- Fonts ---
        self.header_font = ("Roboto", 24, "bold")
        self.label_font = ("Roboto", 12)
        self.button_font = ("Roboto", 11)
        self.info_font = ("Roboto", 14)

        # --- Paths ---
        self.input_image_path = None
        # Assume yolov7 is in the same directory as the script or specify absolute path
        self.yolov7_dir = os.path.abspath("yolov7")
        self.weights_path = os.path.join(
            self.yolov7_dir, "yolov7.pt"
        )  # Default weights

        self.setup_styles()
        self.build_widgets()

        self.check_yolov7_setup()  # Check essential files on startup

        # Place this method within the YOLOv7GUI class definition

    def dialogue_msg(self, title, message):
        """Shows a confirmation dialog before resetting the UI."""
        # Ask the user for confirmation
        confirmed = messagebox.askyesno(
            title=title,
            message=message,
        )
        # If the user clicked "Yes" (True), then proceed with the reset
        if confirmed:
            self.reset_ui()  # Call the original reset method
        # else:
        # Optional: Add feedback if user cancels
        # print("Reset cancelled by user.")

    def setup_styles(self):
        style = ttk.Style()
        # Try different themes for potentially better button looks out-of-the-box
        # 'clam' is often good, 'alt' is another option. 'vista' on Windows.
        try:
            # Prefer 'vista' on Windows, 'aqua' on Mac, otherwise 'clam' or 'alt'
            if sys.platform == "win32":
                style.theme_use("clam")
            elif sys.platform == "darwin":
                style.theme_use("aqua")
            else:
                style.theme_use("clam")  # Good fallback
        except tk.TclError:
            print("Chosen theme not available, using default.")
            style.theme_use("default")

        style.configure("TFrame", background="#f0f0f0")
        style.configure("TLabel", background="#f0f0f0", font=self.label_font)
        style.configure(
            "Header.TLabel",
            background="#f0f0f0",
            font=self.header_font,
            foreground="#333333",
        )
        # --- Make the instruction text bigger ---
        self.info_font = (
            "Roboto",
            16,
        )  # Increased size from 14 to 16 (or more if needed)
        style.configure(
            "Info.TLabel",
            background="#f0f0f0",
            font=self.info_font,
            foreground="#555555",
        )

        # --- Standard Button Style (Keep for reference or other buttons) ---
        style.configure("TButton", font=self.button_font, padding=8)
        # Using map allows defining state-specific appearances more easily
        style.map(
            "TButton",
            background=[
                ("active", "#e0e0e0"),
                ("!disabled", "#cccccc"),
            ],  # Slightly darker grey background
            foreground=[("disabled", "#aaaaaa")],
            relief=[("pressed", "sunken"), ("!pressed", "raised")],
        )  # More classic relief

        # --- Accent Button Style (Browse, Detect) ---
        style.configure(
            "Accent.TButton",
            font=self.button_font,
            padding=8,
            foreground="white",
            background="#007ACC",
        )
        style.map(
            "Accent.TButton",
            background=[
                ("active", "#005F99"),
                ("!disabled", "#007ACC"),
                ("disabled", "#cccccc"),
            ],
            foreground=[("disabled", "#888888"), ("!disabled", "white")],
            relief=[("pressed", "sunken"), ("!pressed", "raised")],
        )

        # --- NEW: Reset Button Style (Muted Red) ---
        reset_bg = "#DC3545"  # A common "danger" or muted red color
        reset_active_bg = "#C82333"  # Darker version for active/pressed
        style.configure(
            "Reset.TButton",
            font=self.button_font,
            padding=8,
            foreground="white",
            background=reset_bg,
        )
        style.map(
            "Reset.TButton",
            background=[
                ("active", reset_active_bg),
                ("!disabled", reset_bg),
                ("disabled", "#cccccc"),
            ],
            foreground=[("disabled", "#888888"), ("!disabled", "white")],
            relief=[("pressed", "sunken"), ("!pressed", "raised")],
        )  # Consistent relief

        # --- LabelFrame Style (Card-like) ---
        style.configure(
            "Card.TLabelframe",
            background="white",
            relief="solid",
            borderwidth=1,
            padding=10,
        )
        # Bolder, slightly larger title for the cards
        style.configure(
            "Card.TLabelframe.Label",
            background="white",
            font=("Roboto", 13, "bold"),
            foreground="#444444",
            padding=(0, 0, 0, 5),
        )  # Padding bottom

    def build_widgets(self):
        # --- Header ---
        header_label = ttk.Label(
            self.root, text="YOLOv7 Object Detector", style="Header.TLabel"
        )
        header_label.pack(pady=(20, 10))

        # --- Control Frame ---
        control_frame = ttk.Frame(self.root, padding=(0, 0, 0, 10))
        control_frame.pack()

        self.browse_button = ttk.Button(
            control_frame,
            text="Browse Image",
            command=self.browse_image,
            style="Accent.TButton",
            width=15,
        )
        self.browse_button.grid(row=0, column=0, padx=10, pady=5)

        self.detect_button = ttk.Button(
            control_frame,
            text="Detect Objects",
            command=self.start_detection_thread,
            state=tk.DISABLED,
            style="Accent.TButton",
            width=15,
        )
        self.detect_button.grid(row=0, column=1, padx=10, pady=5)

        # --- Use the new style for the Reset button ---
        self.reset_button = ttk.Button(
            control_frame,
            text="Reset",
            command=self.reset_ui,
            style="Reset.TButton",
            width=15,  # <-- Changed style here
        )
        self.reset_button.grid(row=0, column=2, padx=10, pady=5)

        # --- Info Label ---
        # Apply the style configure in setup_styles
        self.info_label = ttk.Label(
            self.root, text="Select an image to begin detection.", style="Info.TLabel"
        )
        self.info_label.pack(pady=(0, 10))

        # --- Display Frame (for images) ---
        # Make the display frame expandable
        display_frame = ttk.Frame(self.root, padding=(20, 0, 20, 20))
        display_frame.pack(fill=tk.BOTH, expand=True)
        display_frame.columnconfigure(0, weight=1)
        display_frame.columnconfigure(1, weight=1)
        display_frame.rowconfigure(0, weight=1)

        # --- Left Panel (Original Image) ---
        self.left_panel = ttk.LabelFrame(
            display_frame, text="Original Image", style="Card.TLabelframe"
        )
        self.left_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        # Use a Frame inside LabelFrame for better padding control of the label
        left_content_frame = ttk.Frame(self.left_panel, style="Card.TFrame")
        left_content_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.input_image_label = ttk.Label(
            left_content_frame, anchor="center", background="white"
        )
        self.input_image_label.pack(fill=tk.BOTH, expand=True)
        self.input_image_label.bind(
            "<Configure>", lambda e: self.update_label_image(self.input_image_label)
        )

        # --- Right Panel (Detection Result) ---
        self.right_panel = ttk.LabelFrame(
            display_frame, text="Detection Result", style="Card.TLabelframe"
        )
        self.right_panel.grid(row=0, column=1, sticky="nsew", padx=(10, 0))
        # Use a Frame inside LabelFrame for better padding control of the label
        right_content_frame = ttk.Frame(self.right_panel, style="Card.TFrame")
        right_content_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.result_image_label = ttk.Label(
            right_content_frame, anchor="center", background="white"
        )
        self.result_image_label.pack(fill=tk.BOTH, expand=True)
        self.result_image_label.bind(
            "<Configure>", lambda e: self.update_label_image(self.result_image_label)
        )

    def check_yolov7_setup(self):
        """Checks if essential YOLOv7 files/dirs exist on startup."""
        if not os.path.isdir(self.yolov7_dir):
            messagebox.showwarning(
                "Setup Warning",
                f"YOLOv7 directory not found at: {self.yolov7_dir}\n"
                f"Please ensure the 'yolov7' folder is in the same directory as this script, "
                f"or modify the 'self.yolov7_dir' path in the code.\n"
                "Detection will likely fail.",
            )
            return False  # Indicate setup issue

        detect_script = os.path.join(self.yolov7_dir, "detect.py")
        if not os.path.isfile(detect_script):
            messagebox.showwarning(
                "Setup Warning",
                f"YOLOv7 'detect.py' script not found in: {self.yolov7_dir}\n"
                "Detection will not work.",
            )
            return False

        if not os.path.isfile(self.weights_path):
            messagebox.showwarning(
                "Setup Warning",
                f"Default weights file '{os.path.basename(self.weights_path)}' not found in: {self.yolov7_dir}\n"
                "Detection requires weights. You might need to download them (e.g., yolov7.pt).",
            )
            # Don't return False here, maybe user has other weights, but warn them.

        return True  # Setup seems okay

    def browse_image(self):
        file_path = filedialog.askopenfilename(
            title="Select an Image for Object Detection",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.webp *.tif")],
        )
        if file_path:
            self.reset_ui(
                clear_input=False
            )  # Clear previous results but keep buttons enabled if needed
            self.input_image_path = file_path
            short_name = os.path.basename(file_path)
            self.info_label.config(text=f"Selected: {short_name}")
            self.display_flexible_image(file_path, self.input_image_label)
            self.detect_button.config(state=tk.NORMAL)
            # Clear previous result image
            self.result_image_label.config(image="")
            self.result_image_label.image = None
            self.result_image_label.original_image = None

    def display_flexible_image(self, image_path, label_widget):
        """Loads an image and prepares it for flexible display."""
        try:
            # Clear previous image reference if any
            label_widget.image = None
            label_widget.original_image = None

            orig_image = PILImage.open(image_path)
            # Store the original PIL image on the widget
            label_widget.original_image = orig_image
            # Trigger the first update explicitly
            self.update_label_image(label_widget)
        except FileNotFoundError:
            messagebox.showerror("Error", f"Image file not found: {image_path}")
            self.reset_ui()
        except Exception as e:
            messagebox.showerror(
                "Image Loading Error", f"Could not load image: {image_path}\nError: {e}"
            )
            self.reset_ui()

    def update_label_image(self, label_widget):
        """Callback function to resize and display the image in the label."""
        orig_image = getattr(label_widget, "original_image", None)
        if orig_image is None:
            # No original image loaded for this label, do nothing
            return

        # Get the current size of the label widget's container frame
        container = label_widget.master
        width = container.winfo_width()
        height = container.winfo_height()

        # Avoid excessive calculations or potential errors if widget size is tiny
        if width < 20 or height < 20:
            # print(f"Widget size too small ({width}x{height}), deferring update.")
            # Optionally schedule another update soon:
            # label_widget.after(100, lambda: self.update_label_image(label_widget))
            return

        # Create a copy to avoid modifying the original
        img_copy = orig_image.copy()

        # Calculate aspect ratio and resize using thumbnail (preserves ratio)
        img_copy.thumbnail((width - 10, height - 10), LANCZOS)  # Subtract padding

        # Convert to PhotoImage
        try:
            photo = ImageTk.PhotoImage(img_copy)

            # Update the label's image
            label_widget.config(image=photo)
            # Keep a reference to prevent garbage collection!
            label_widget.image = photo
        except Exception as e:
            print(f"Error creating PhotoImage: {e}")

    def start_detection_thread(self):
        if not self.input_image_path or not os.path.exists(self.input_image_path):
            messagebox.showerror("Error", "No valid image selected or file not found.")
            return

        # Check YOLOv7 setup again before running
        if not self.check_yolov7_setup():
            return  # Stop if setup check fails

        # Disable buttons during processing
        self.detect_button.config(state=tk.DISABLED)
        self.browse_button.config(state=tk.DISABLED)
        self.reset_button.config(state=tk.DISABLED)
        self.info_label.config(text="Detecting objects, please wait...")
        self.root.update_idletasks()  # Ensure UI updates before starting thread

        # Clear previous result display before starting
        self.result_image_label.config(image="")
        self.result_image_label.image = None
        self.result_image_label.original_image = None

        # Run detection in a separate thread
        thread = threading.Thread(
            target=self.detect_objects, daemon=True
        )  # Daemon allows exit even if thread runs
        thread.start()

    def detect_objects(self):
        """Handles the YOLOv7 detection process."""
        output_path = None  # Initialize output path
        try:
            image_name = os.path.basename(self.input_image_path)
            # Define project/name for output, helps in finding results
            project_dir = os.path.join(self.yolov7_dir, "runs", "detect")
            experiment_name = "GUI_exp"  # Consistent experiment name

            # --- Prepare Command ---
            # Use sys.executable to try using the same python interpreter as the GUI
            # This might be more robust if dependencies are managed in one environment.
            # If yolov7 needs a specific different python, change "sys.executable" back to "python" or the specific path.
            python_executable = sys.executable
            command = [
                python_executable,
                "detect.py",
                "--weights",
                self.weights_path,
                # "--img", "640", # Use default or let detect.py choose
                "--conf-thres",
                "0.25",  # Confidence threshold
                "--iou-thres",
                "0.45",  # IoU threshold for NMS
                "--source",
                self.input_image_path,  # Source image path
                "--project",
                project_dir,  # Base directory for runs
                "--name",
                experiment_name,  # Specific folder name for this run
                "--exist-ok",  # Overwrite previous results in the same folder name
                # "--device", "cpu", # Uncomment to force CPU if needed
                "--save-txt",  # Optionally save labels to txt files
                "--no-trace",  # Avoids potential issues with traced models if not needed
            ]

            print(f"Running command: {' '.join(command)}")
            print(f"Working directory: {self.yolov7_dir}")

            # --- Execute Command ---
            result = subprocess.run(
                command,
                cwd=self.yolov7_dir,
                capture_output=True,
                text=True,
                check=False,
            )  # check=False to handle errors manually

            # --- Process Output ---
            print("\n--- YOLOv7 detect.py stdout ---")
            print(result.stdout)
            if result.stderr:
                print("\n--- YOLOv7 detect.py stderr ---")
                print(result.stderr)

            if result.returncode != 0:
                error_message = f"Object detection failed (detect.py exited with code {result.returncode}).\n\n"
                error_message += "Check console/terminal output for details.\n"
                if result.stderr:
                    # Show last few lines of stderr
                    error_message += f"\nError details:\n{' '.join(result.stderr.splitlines()[-5:])}"  # Show last 5 lines
                self.root.after(
                    0, messagebox.showerror, "Detection Error", error_message
                )  # Schedule messagebox on main thread
                return  # Stop processing

            # --- Find Output Image ---
            # Output is expected in project_dir / experiment_name / image_name
            output_path = os.path.join(project_dir, experiment_name, image_name)

            if os.path.exists(output_path):
                # Schedule GUI update on the main thread using self.root.after
                self.root.after(0, self.update_gui_post_detection, output_path, True)
            else:
                # Try finding the latest run if exist-ok wasn't used or failed
                print(
                    f"Warning: Expected output not found at {output_path}. Searching latest run..."
                )
                latest_run_path = self.find_latest_run_output(project_dir, image_name)
                if latest_run_path and os.path.exists(latest_run_path):
                    self.root.after(
                        0, self.update_gui_post_detection, latest_run_path, True
                    )
                else:
                    err_msg = f"Output image could not be found.\nExpected: {output_path}\nAlso checked latest runs in: {project_dir}"
                    self.root.after(0, messagebox.showerror, "Error", err_msg)
                    # Still update GUI state, but without result image
                    self.root.after(0, self.update_gui_post_detection, None, False)

        except FileNotFoundError as fnf_error:
            # Specific error for missing python or detect.py
            error_message = f"File not found during detection: {fnf_error}.\n"
            error_message += f"Ensure '{python_executable}' is runnable and 'detect.py' exists in {self.yolov7_dir}."
            self.root.after(0, messagebox.showerror, "Execution Error", error_message)
            self.root.after(
                0, self.update_gui_post_detection, None, False
            )  # Update UI state

        except Exception as e:
            error_message = f"An unexpected error occurred during detection:\n{type(e).__name__}: {e}"
            print(f"Full exception details: {e}")  # Log full error
            self.root.after(0, messagebox.showerror, "Execution Error", error_message)
            self.root.after(
                0, self.update_gui_post_detection, None, False
            )  # Update UI state

    def find_latest_run_output(self, project_dir, image_name):
        """Finds the output image in the most recently modified run directory."""
        try:
            sub_dirs = [
                os.path.join(project_dir, d)
                for d in os.listdir(project_dir)
                if os.path.isdir(os.path.join(project_dir, d))
            ]
            if not sub_dirs:
                return None
            # Sort by modification time, newest first
            latest_dir = max(sub_dirs, key=os.path.getmtime)
            potential_output = os.path.join(latest_dir, image_name)
            return potential_output if os.path.exists(potential_output) else None
        except Exception as e:
            print(f"Error finding latest run directory: {e}")
            return None

    def update_gui_post_detection(self, output_path, success):
        """Updates the GUI elements after detection attempt (runs on main thread)."""
        if success and output_path:
            self.info_label.config(
                text=f"Detection complete. Result: {os.path.basename(output_path)}"
            )
            self.display_flexible_image(output_path, self.result_image_label)
            # Optionally show a success message
            # messagebox.showinfo("Success", "Object detection completed successfully!")
        else:
            # Update info label even on failure
            self.info_label.config(text="Detection failed or output not found.")
            # Clear the result panel
            self.result_image_label.config(image="")
            self.result_image_label.image = None
            self.result_image_label.original_image = None

        # Re-enable buttons regardless of success/failure
        self.detect_button.config(
            state=tk.NORMAL if self.input_image_path else tk.DISABLED
        )
        self.browse_button.config(state=tk.NORMAL)
        self.reset_button.config(state=tk.NORMAL)

    def     reset_ui(self, clear_input=True):
        """Resets the UI to its initial state."""
        if clear_input:
            self.input_image_path = None
            self.info_label.config(text="Select an image to begin detection.")
            self.input_image_label.config(image="")
            self.input_image_label.image = None  # Clear reference
            self.input_image_label.original_image = (
                None  # Clear stored original PIL image
            )
            self.detect_button.config(state=tk.DISABLED)

        # Always clear the result panel on reset
        self.result_image_label.config(image="")
        self.result_image_label.image = None  # Clear reference
        self.result_image_label.original_image = None  # Clear stored original PIL image

        # Ensure buttons are in correct state
        self.browse_button.config(state=tk.NORMAL)
        self.reset_button.config(state=tk.NORMAL)
        # Re-enable detect only if input image still exists (when clear_input is False)
        if not clear_input and self.input_image_path:
            self.detect_button.config(state=tk.NORMAL)


if __name__ == "__main__":
    root = tk.Tk()
    app = YOLOv7GUI(root)
    root.mainloop()