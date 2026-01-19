import os
import sys
import threading
import time
import urllib.request
import webbrowser
import subprocess
from typing import Tuple, Optional, Callable
import tkinter as tk

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms
from PIL import Image, ImageTk, ImageOps, ImageDraw
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib
import customtkinter as ctk

# -----------------------------------------------------------------------------
# REQUIREMENTS.TXT CONTENT
# -----------------------------------------------------------------------------
# torch>=2.0.0
# torchvision>=0.15.0
# facenet-pytorch>=2.5.3
# customtkinter>=5.2.0
# pillow>=10.0.0
# matplotlib>=3.7.0
# -----------------------------------------------------------------------------

# Try importing facenet-pytorch
try:
    from facenet_pytorch import MTCNN, InceptionResnetV1
except ImportError:
    print("Critical Error: 'facenet-pytorch' module not found.")
    sys.exit(1)

# -----------------------------------------------------------------------------
# CONSTANTS & THEME
# -----------------------------------------------------------------------------
THEME = {
    "bg": "#050505",       # Deep Matte Black
    "panel": "#101010",    # Dark Gunmetal
    "cyan": "#00f0ff",     # Neon Cyan
    "red": "#ff2a2a",      # Alert Red
    "text": "#eeeeee",     # Off-white
    "text_dim": "#808080", # Gray
}

# -----------------------------------------------------------------------------
# CLOAK ENGINE (THE BRAIN)
# -----------------------------------------------------------------------------
class CloakEngine:
    """
    Core engine for Project Maske.
    Implements PGD (Projected Gradient Descent) using pure PyTorch operations.
    """
    def __init__(self):
        self.device = self._get_device()
        print(f"[Engine] Initializing on device: {self.device}")
        
        # Load Models
        self.resnet = InceptionResnetV1(pretrained='vggface2').to(self.device).eval()
        self.mtcnn = MTCNN(keep_all=False, select_largest=True, device=self.device)

    def _get_device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')

    def protect(self, image_path: str, epsilon: float = 0.05, steps: int = 50, alpha: float = 0.005, progress_callback: Callable[[float, str], None] = None) -> Tuple[Optional[Image.Image], Optional[Image.Image], dict]:
        """
        Applies adversarial perturbation and pastes the result back onto the original image.
        """
        start_time = time.time()
        try:
            if progress_callback: progress_callback(0.1, "Loading Image & AI Models...")
            
            # 1. Load Image
            img_pil = Image.open(image_path).convert('RGB')
            w, h = img_pil.size
            
            # 2. Detect & Crop
            if progress_callback: progress_callback(0.2, "Detecting Face...")
            
            # Get boxes first to know where to paste back
            boxes, _ = self.mtcnn.detect(img_pil)
            
            if boxes is None:
                return None, None, {"status": "error", "msg": "No face detected"}
                
            # Select largest face (matching self.mtcnn.select_largest=True logic)
            # Box format: [x1, y1, x2, y2]
            largest_box_idx = 0
            max_area = 0
            for i, box in enumerate(boxes):
                area = (box[2] - box[0]) * (box[3] - box[1])
                if area > max_area:
                    max_area = area
                    largest_box_idx = i
            
            target_box = boxes[largest_box_idx]
            target_box = [int(b) for b in target_box]
            
            # Get the aligned face tensor for recognition model
            x_face = self.mtcnn(img_pil)
            
            if x_face is None:
                 return None, None, {"status": "error", "msg": "Face extraction failed"}

            x_face = x_face.unsqueeze(0).to(self.device) # [1, 3, 160, 160]
            
            # 3. Original Embedding
            if progress_callback: progress_callback(0.3, "Extracting Biometrics...")
            with torch.no_grad():
                original_embedding = self.resnet(x_face).detach()

            # 4. Target Embedding (Shifted Centroid)
            noise = torch.randn_like(original_embedding).to(self.device)
            noise = F.normalize(noise, p=2, dim=1)
            target_embedding = original_embedding + (noise * 5.0)
            target_embedding = F.normalize(target_embedding, p=2, dim=1)

            # 5. PGD Loop
            delta = torch.zeros_like(x_face).to(self.device)
            delta.requires_grad = True
            optimizer = torch.optim.SGD([delta], lr=alpha)

            for step_i in range(steps):
                if progress_callback and step_i % 5 == 0:
                    prog = 0.3 + (0.6 * (step_i / steps))
                    progress_callback(prog, f"Injecting Noise... Iteration {step_i}/{steps}")

                adv_x = x_face + delta
                adv_embedding = self.resnet(adv_x)
                
                cos_sim = F.cosine_similarity(adv_embedding, target_embedding)
                loss = -cos_sim.mean()
                
                optimizer.zero_grad()
                loss.backward()
                
                grad_sign = delta.grad.sign()
                delta.data = delta.data - alpha * grad_sign
                delta.data = torch.clamp(delta.data, -epsilon, epsilon)
                delta.grad.zero_()

            # 6. Finalize & Paste Back
            if progress_callback: progress_callback(0.95, "Reconstructing Image...")
            
            with torch.no_grad():
                final_adv_x = x_face + delta
                final_embedding = self.resnet(final_adv_x)
                
                # Metrics
                final_sim = F.cosine_similarity(final_embedding, original_embedding).item()
                id_shift = (1.0 - final_sim) * 100 
                l2_dist = torch.norm(final_adv_x - x_face).item()
                
                # Heatmap
                diff = torch.abs(final_adv_x - x_face).squeeze(0).mean(dim=0).cpu()
                diff_norm = (diff - diff.min()) / (diff.max() - diff.min() + 1e-8)
                
                try:
                    cmap = matplotlib.colormaps['inferno'] if hasattr(matplotlib, 'colormaps') else cm.get_cmap('inferno')
                except:
                    cmap = cm.get_cmap('inferno')
                    
                heatmap_np = cmap(diff_norm.numpy())
                
                # Create Full Size Heatmap Overlay
                box_w = target_box[2] - target_box[0]
                box_h = target_box[3] - target_box[1]
                
                heatmap_face = Image.fromarray((heatmap_np * 255).astype('uint8')).resize((box_w, box_h), Image.BILINEAR)
                
                full_heatmap = Image.new('RGBA', img_pil.size, (0,0,0,0))
                full_heatmap.paste(heatmap_face, (target_box[0], target_box[1]))
                
                full_heatmap_composite = img_pil.copy().convert("RGBA")
                full_heatmap_composite = Image.alpha_composite(full_heatmap_composite, full_heatmap).convert("RGB")

                # Reconstruct Protected Face
                out_tensor = final_adv_x.squeeze(0).cpu()
                out_tensor = (out_tensor - out_tensor.min()) / (out_tensor.max() - out_tensor.min())
                out_permute = out_tensor.permute(1, 2, 0)
                protected_face_pil = Image.fromarray((out_permute.numpy() * 255).astype("uint8"))
                protected_face_pil = protected_face_pil.resize((box_w, box_h), Image.LANCZOS)
                
                # Paste back
                full_protected_img = img_pil.copy()
                full_protected_img.paste(protected_face_pil, (target_box[0], target_box[1]))

            elapsed = (time.time() - start_time) * 1000 # ms
            
            return full_protected_img, full_heatmap_composite, {
                "l2": l2_dist,
                "id_shift": id_shift,
                "time": elapsed,
                "device": str(self.device)
            }

        except Exception as e:
            print(f"[Engine] Error: {e}")
            import traceback; traceback.print_exc()
            return None, None, {"status": "error", "msg": str(e)}

# -----------------------------------------------------------------------------
# ZOOMABLE IMAGE CANVAS CLASS
# -----------------------------------------------------------------------------
class ZoomableImageCanvas(ctk.CTkFrame):
    def __init__(self, master, bg_color=None, sync_callback=None, **kwargs):
        super().__init__(master, **kwargs)
        self.sync_callback = sync_callback
        
        # Internal Canvas
        self.canvas = ctk.CTkCanvas(self, bg=bg_color if bg_color else THEME["bg"], highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        
        # State
        self.pil_image = None
        self.ctk_image_ref = None # Keep reference
        self.zoom_scale = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self.is_dragging = False
        self.last_mouse_x = 0
        self.last_mouse_y = 0
        
        # Bindings
        self.canvas.bind("<ButtonPress-1>", self._on_drag_start)
        self.canvas.bind("<B1-Motion>", self._on_drag_move)
        self.canvas.bind("<ButtonRelease-1>", self._on_drag_end)
        self.canvas.bind("<MouseWheel>", self._on_zoom) # Windows
        # self.canvas.bind("<Button-4>", self._on_zoom) # Linux Scroll Up
        # self.canvas.bind("<Button-5>", self._on_zoom) # Linux Scroll Down

    def set_image(self, pil_image):
        if not pil_image:
            self.canvas.delete("all")
            self.pil_image = None
            return
            
        self.pil_image = pil_image
        self.reset_view()

    def reset_view(self):
        self.zoom_scale = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self._redraw()
        
    def _redraw(self):
        if not self.pil_image: return
        
        # Calculate new size
        orig_w, orig_h = self.pil_image.size
        new_w = int(orig_w * self.zoom_scale)
        new_h = int(orig_h * self.zoom_scale)
        
        # Optimization: Don't resize if tiny difference
        # But we need resizing for zoom
        # Use simple resizing for speed during drag/zoom? No, quality is fast enough usually.
        
        # Center logic
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        
        # If image is smaller than canvas, center it inherently ??
        # Or Just place at center + pan
        
        center_x = canvas_w // 2 + self.pan_x
        center_y = canvas_h // 2 + self.pan_y
        
        # Create temp resized image
        # For huge images, this might be slow. Optimization: Crop visible area then resize.
        # For this tool (face images usually < 4k), resize whole is okay.
        
        resized = self.pil_image.resize((new_w, new_h), Image.BILINEAR)
        self.ctk_image_ref = ImageTk.PhotoImage(resized)
        
        self.canvas.delete("img")
        self.canvas.create_image(center_x, center_y, image=self.ctk_image_ref, tags="img")

    def _on_drag_start(self, event):
        self.is_dragging = True
        self.last_mouse_x = event.x
        self.last_mouse_y = event.y

    def _on_drag_move(self, event):
        if not self.is_dragging or not self.pil_image: return
        
        dx = event.x - self.last_mouse_x
        dy = event.y - self.last_mouse_y
        
        self.pan_x += dx
        self.pan_y += dy
        
        self.last_mouse_x = event.x
        self.last_mouse_y = event.y
        
        self._redraw()
        if self.sync_callback:
            self.sync_callback(self.zoom_scale, self.pan_x, self.pan_y)

    def _on_drag_end(self, event):
        self.is_dragging = False

    def _on_zoom(self, event):
        if not self.pil_image: return
        
        # Windows: event.delta is usually 120
        if event.delta > 0:
            factor = 1.1
        else:
            factor = 0.9
            
        new_scale = self.zoom_scale * factor
        
        # Limits
        if new_scale < 0.1: new_scale = 0.1
        if new_scale > 10.0: new_scale = 10.0
        
        self.zoom_scale = new_scale
        
        self._redraw()
        if self.sync_callback:
            self.sync_callback(self.zoom_scale, self.pan_x, self.pan_y)

    def set_view_params(self, scale, px, py):
        """ Used by sync mechanism to update this view from another """
        if self.zoom_scale == scale and self.pan_x == px and self.pan_y == py:
            return
            
        self.zoom_scale = scale
        self.pan_x = px
        self.pan_y = py
        self._redraw()


# -----------------------------------------------------------------------------
# MASKE APP v3 (UX OVERHAUL)
# -----------------------------------------------------------------------------
class MaskeApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        # Theme Setup
        ctk.set_appearance_mode("Dark")
        ctk.set_default_color_theme("dark-blue")
        self.configure(fg_color=THEME["bg"])
        
        # Window Config
        self.title("MASKE // CYBERSECURITY TOOL v1")
        self.geometry("1200x800")
        
        # State
        self.engine = None
        self.original_image_path = None
        self.protected_image = None
        self.heatmap_image = None
        self.is_heatmap_active = False
        self.is_processing = False
        self.is_fullscreen = False
        
        # Init Layout
        self._init_layout()
        
        # Bindings
        self.bind("<F11>", self._toggle_fullscreen)
        self.bind("<Escape>", lambda e: self._toggle_fullscreen(e, force_exit=True))
        
        # Async Engine Load
        self._load_engine()

    def _init_layout(self):
        # Grid: 1 Row, 2 Columns (Sidebar, Content)
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

        # --- Sidebar ---
        self.sidebar = ctk.CTkFrame(self, width=280, corner_radius=0, fg_color=THEME["panel"])
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.sidebar.grid_propagate(False)
        self._build_sidebar()

        # --- Main Content Area ---
        self.content_area = ctk.CTkFrame(self, fg_color="transparent")
        self.content_area.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)
        self._build_content()

    def _build_sidebar(self):
        # Reset Button (Top)
        self.btn_reset = ctk.CTkButton(
            self.sidebar,
            text="⟳ NEW TARGET",
            font=("Roboto Bold", 12),
            height=30,
            fg_color="#222",
            hover_color="#333", 
            command=self._reset_app
        )
        self.btn_reset.pack(pady=(20, 10), padx=20, fill="x")

        # Logo
        lbl_title = ctk.CTkLabel(self.sidebar, text="MASKE", font=("Roboto Medium", 36), text_color=THEME["cyan"])
        lbl_title.pack(pady=(10, 5))
        
        lbl_ver = ctk.CTkLabel(self.sidebar, text="v1.0 // DEEP CLOAK", font=("Roboto Mono", 12), text_color=THEME["text_dim"])
        lbl_ver.pack(pady=(0, 40))

        # Controls Group
        self._create_separator(self.sidebar, "CONTROL DECK")
        
        self.slider_eps = self._create_control("Intensity (Epsilon)", 0.01, 0.2, 0.05)
        self.slider_steps = self._create_control("Iterations", 10, 200, 50, resolution=1)
        
        self.sw_heatmap = ctk.CTkSwitch(
            self.sidebar, 
            text="NOISE HEATMAP", 
            command=self._toggle_heatmap, 
            progress_color=THEME["cyan"],
            font=("Roboto Bold", 12),
            text_color=THEME["text"]
        )
        self.sw_heatmap.pack(pady=20, padx=25, anchor="w")

        # Engage Button
        self.btn_engage = ctk.CTkButton(
            self.sidebar,
            text="SYSTEM READY",
            font=("Roboto Bold", 16),
            height=50,
            fg_color="#333",
            state="disabled",
            command=self._start_process
        )
        self.btn_engage.pack(pady=(40, 10), padx=20, fill="x")

        # Export Button
        self.btn_export = ctk.CTkButton(
            self.sidebar,
            text="EXPORT DATA",
            font=("Roboto Bold", 14),
            height=40,
            fg_color="transparent",
            border_width=2,
            border_color=THEME["cyan"],
            text_color=THEME["cyan"],
            state="disabled",
            command=self._export_result
        )
        self.btn_export.pack(pady=10, padx=20, fill="x")

        # Signature
        lbl_sig = ctk.CTkLabel(self.sidebar, text="Developed by MacallanTheRoot", font=("Roboto", 10), text_color="gray", cursor="hand2")
        lbl_sig.pack(pady=(5, 0))
        lbl_sig.bind("<Button-1>", lambda e: webbrowser.open("https://github.com/MacallanTheRoot"))
        
        # Status Log
        self.lbl_status = ctk.CTkLabel(self.sidebar, text="Initializing...", font=("Roboto Mono", 10), text_color="gray")
        self.lbl_status.pack(side="bottom", pady=20)

    def _create_control(self, text, vmin, vmax, vdef, resolution=0.01):
        frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        frame.pack(fill="x", padx=20, pady=10)
        
        lbl = ctk.CTkLabel(frame, text=text, font=("Roboto", 12), text_color="white")
        lbl.pack(anchor="w")
        
        slider = ctk.CTkSlider(
            frame, 
            from_=vmin, 
            to=vmax, 
            number_of_steps=(vmax-vmin)/resolution,
            progress_color=THEME["cyan"], 
            button_color="white",
            button_hover_color=THEME["cyan"]
        )
        slider.set(vdef)
        slider.pack(pady=5, fill="x")
        return slider

    def _create_separator(self, parent, text):
        f = ctk.CTkFrame(parent, fg_color="transparent")
        f.pack(fill="x", padx=20, pady=5)
        l = ctk.CTkLabel(f, text=text, font=("Roboto Bold", 11), text_color=THEME["cyan"])
        l.pack(anchor="w")
        ctk.CTkFrame(f, height=2, fg_color="#333").pack(fill="x", pady=(2, 10))

    def _build_content(self):
        # 1. VISUALIZER (Top Section)
        self.vis_frame = ctk.CTkFrame(self.content_area, fg_color="transparent")
        self.vis_frame.pack(fill="both", expand=True, pady=(0, 20))
        
        self.vis_frame.grid_columnconfigure(0, weight=1)
        self.vis_frame.grid_columnconfigure(1, weight=1)
        self.vis_frame.grid_rowconfigure(0, weight=1)

        # Panel Left (Original)
        self.panel_left_frame = self._create_panel_frame(self.vis_frame, 0, "UNSECURED", THEME["red"])
        self.canvas_left = ZoomableImageCanvas(self.panel_left_frame, sync_callback=self._sync_view_from_left)
        self.canvas_left.pack(fill="both", expand=True, padx=2, pady=2) # Inside frame
        
        # Panel Right (Result)
        self.panel_right_frame = self._create_panel_frame(self.vis_frame, 1, "SECURED", THEME["cyan"])
        self.canvas_right = ZoomableImageCanvas(self.panel_right_frame, sync_callback=self._sync_view_from_right)
        self.canvas_right.pack(fill="both", expand=True, padx=2, pady=2)
        
        # Re-add badges on top of canvases? Canvases are inside frames.
        # Let's put badges in the frame, place() works relative to parent (frame).
        # But pack() of canvas might cover it? 
        # Z-order: place children appear on top if added later.
        self._add_badge(self.panel_left_frame, "UNSECURED", THEME["red"])
        self._add_badge(self.panel_right_frame, "SECURED", THEME["cyan"])

        # Drop Zone Overlay
        self._build_drop_zone()

        # 2. METRICS (Bottom Section)
        self.metrics_frame = ctk.CTkFrame(self.content_area, height=180, fg_color=THEME["panel"])
        self.metrics_frame.pack(fill="x", side="bottom")
        
        self.progress_bar = ctk.CTkProgressBar(self.metrics_frame, progress_color=THEME["cyan"], height=4)
        self.progress_bar.set(0)
        self.progress_bar.pack(fill="x", side="top")
        
        self.cards_frame = ctk.CTkFrame(self.metrics_frame, fg_color="transparent")
        self.cards_frame.pack(fill="both", expand=True, padx=20, pady=20)
        self.cards_frame.grid_columnconfigure((0,1,2), weight=1)
        
        self.card_l2 = self._create_metric_card(self.cards_frame, 0, "L2 Distance", "0.00")
        self.card_id = self._create_metric_card(self.cards_frame, 1, "Identity Shift", "0%")
        self.card_time = self._create_metric_card(self.cards_frame, 2, "Process Time", "0ms")

    def _build_drop_zone(self):
        self.drop_zone = ctk.CTkFrame(self.vis_frame, fg_color="#0f1218", corner_radius=15, border_width=2, border_color="#333")
        self.drop_zone.place(relx=0.5, rely=0.5, anchor="center", relwidth=0.6, relheight=0.4)
        
        lbl_title = ctk.CTkLabel(self.drop_zone, text="DROP TARGET ACQUIRED", font=("Roboto Medium", 18), text_color="white")
        lbl_title.pack(pady=(40, 5))
        
        lbl_sub = ctk.CTkLabel(self.drop_zone, text="Click to Initialize Source File", font=("Roboto", 12), text_color="gray")
        lbl_sub.pack()
        
        # Bindings
        for w in [self.drop_zone, lbl_title, lbl_sub]:
            w.bind("<Button-1>", lambda e: self._browse_file())
            w.configure(cursor="hand2")

    def _create_panel_frame(self, parent, col, status, color):
        f = ctk.CTkFrame(parent, fg_color="#080808", corner_radius=10, border_width=1, border_color="#222")
        f.grid(row=0, column=col, sticky="nsew", padx=10)
        return f
        
    def _add_badge(self, parent, text, color):
        badge = ctk.CTkLabel(parent, text=f" {text} ", fg_color=color, text_color="black", font=("Roboto Bold", 10), corner_radius=4)
        badge.place(x=10, y=10)

    def _create_metric_card(self, parent, col, title, val):
        f = ctk.CTkFrame(parent, fg_color="#181818", corner_radius=8)
        f.grid(row=0, column=col, sticky="ew", padx=10)
        ctk.CTkLabel(f, text=title, font=("Roboto", 11), text_color="gray").pack(pady=(15,0))
        lbl_val = ctk.CTkLabel(f, text=val, font=("Roboto Medium", 24), text_color="white")
        lbl_val.pack(pady=(0,15))
        return lbl_val

    # --- Feature: Reset ---
    def _reset_app(self):
        # Clear Data
        self.original_image_path = None
        self.protected_image = None
        self.heatmap_image = None
        
        # Clear Canvases
        self.canvas_left.set_image(None)
        self.canvas_right.set_image(None)
        
        # Reset Metrics
        self.progress_bar.set(0)
        self.card_l2.configure(text="0.00")
        self.card_id.configure(text="0%")
        self.card_time.configure(text="0ms")
        self.lbl_status.configure(text="System Washed. Ready for New Target.")
        
        # Disable Buttons
        self.btn_engage.configure(state="disabled", text="SYSTEM READY")
        self.btn_export.configure(state="disabled")
        
        # Show Drop Zone
        self.drop_zone.place(relx=0.5, rely=0.5, anchor="center", relwidth=0.6, relheight=0.4)

    # --- Feature: Sync Zoom/Pan ---
    def _sync_view_from_left(self, s, x, y):
        # Apply to right
        self.canvas_right.set_view_params(s, x, y)
        
    def _sync_view_from_right(self, s, x, y):
        self.canvas_left.set_view_params(s, x, y)

    # --- Feature: Fullscreen ---
    def _toggle_fullscreen(self, event=None, force_exit=False):
        if force_exit:
            self.is_fullscreen = False
        else:
            self.is_fullscreen = not self.is_fullscreen
        self.attributes("-fullscreen", self.is_fullscreen)

    # --- Standard Logic ---
    def _load_engine(self):
        def _load():
            try:
                self.engine = CloakEngine()
                self.after(0, lambda: self._on_engine_ready())
            except Exception as e:
                print(e)
        threading.Thread(target=_load, daemon=True).start()

    def _on_engine_ready(self):
        self.lbl_status.configure(text="System Online. Waiting for Input.")
        
    def _browse_file(self):
        path = ctk.filedialog.askopenfilename(filetypes=[("Images", "*.jpg;*.png;*.jpeg")])
        if path:
            self.original_image_path = path
            self._display_source(path)
            self.drop_zone.place_forget()
            self.btn_engage.configure(state="normal")
            self.lbl_status.configure(text=f"Target Loaded: {os.path.basename(path)}")
            
    def _display_source(self, path):
        img = Image.open(path)
        self.canvas_left.set_image(img)
        self.canvas_right.set_image(None) # Clear previous result

    def _start_process(self):
        if self.is_processing or not self.original_image_path: return
        self.is_processing = True
        self.btn_engage.configure(state="disabled", text="PROCESSING...")
        self.progress_bar.set(0)
        
        eps = self.slider_eps.get()
        steps = int(self.slider_steps.get())
        
        threading.Thread(target=self._run_engine, args=(eps, steps), daemon=True).start()

    def _run_engine(self, eps, steps):
        def cb(prog, msg):
            self.after(0, lambda: self._update_progress(prog, msg))
        protected, heatmap, metrics = self.engine.protect(self.original_image_path, epsilon=eps, steps=steps, progress_callback=cb)
        self.after(0, lambda: self._on_complete(protected, heatmap, metrics))

    def _update_progress(self, prog, msg):
        self.progress_bar.set(prog)
        self.lbl_status.configure(text=msg)

    def _on_complete(self, protected, heatmap, metrics):
        self.is_processing = False
        self.btn_engage.configure(state="normal", text="ENGAGE CLOAK")
        
        if protected is None:
            self.lbl_status.configure(text=f"Error: {metrics.get('msg')}")
            return
        
        self.protected_image = protected
        self.heatmap_image = heatmap
        self.progress_bar.set(1.0)
        self.lbl_status.configure(text="Cloaking Successful.")
        self.btn_export.configure(state="normal")
        
        self._update_right_panel()
        
        self.card_l2.configure(text=f"{metrics['l2']:.2f}")
        self.card_id.configure(text=f"{metrics['id_shift']:.1f}%")
        self.card_time.configure(text=f"{metrics['time']:.0f}ms")

    def _toggle_heatmap(self):
        self.is_heatmap_active = bool(self.sw_heatmap.get())
        if self.protected_image:
            self._update_right_panel()

    def _update_right_panel(self):
        target_img = self.heatmap_image if self.is_heatmap_active else self.protected_image
        if target_img:
            # ZoomableCanvas handles full res logic internally in set_image
            self.canvas_right.set_image(target_img)

    def _export_result(self):
        if not self.protected_image: return
        dir_name = os.path.dirname(self.original_image_path)
        base = os.path.splitext(os.path.basename(self.original_image_path))[0]
        out_path = os.path.join(dir_name, f"{base}_masked.png")
        self.protected_image.save(out_path)
        self.lbl_status.configure(text=f"Exported to: {os.path.basename(out_path)}")
        
        # Cross-Platform Open Folder
        try:
            if sys.platform == "win32":
                os.startfile(dir_name)
            elif sys.platform == "darwin":
                subprocess.call(["open", dir_name])
            else: # linux variants
                subprocess.call(["xdg-open", dir_name])
        except Exception as e:
            print(f"Error opening folder: {e}")

if __name__ == "__main__":
    app = MaskeApp()
    app.mainloop()
