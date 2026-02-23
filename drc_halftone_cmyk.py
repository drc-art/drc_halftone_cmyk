# drc_halftone_cmyk.py — CMYK halftone tool
# This build:
#   • ICON_PATH resolved relative to script dir (cross-platform; .icns on macOS)
#   • Verifies icon load and reports result in the status bar
#   • Solid red artboard border (no dashes)
#   • Rounded buttons/combos (#363436), app background #414042, small black slider handles
#   • Fast slider interactivity + grayscale K-only mode + invert, shapes, sizes, etc.

import os, sys, math, time, json, base64
from dataclasses import dataclass, replace, field
from typing import Optional, Tuple, List, Dict, TYPE_CHECKING
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count

if TYPE_CHECKING:
    from PIL.Image import Image as PILImage
    import numpy.typing as npt
else:
    PILImage = "PILImage"
    npt = None

# PySide6 GUI imports
from PySide6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                               QWidget, QLabel, QPushButton, QSlider, QComboBox, 
                               QCheckBox, QGroupBox, QDial, QFileDialog, QMessageBox, 
                               QGridLayout, QSizePolicy, QMenu, QToolBox, QScrollArea,
                               QLineEdit, QColorDialog)
from PySide6.QtCore import Qt, QTimer, QSize, QPoint, QRect
from PySide6.QtGui import (QIcon, QPixmap, QImage, QPainter, QPen, QColor, 
                           QMouseEvent, QPainterPath, QAction, QBrush, QFont)
from PySide6.QtWidgets import QFrame

# Image processing imports
try:
    from PIL import Image, ImageDraw, ImageChops
    from PIL.Image import Transpose
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    Image = None
    ImageDraw = None
    ImageChops = None
    Transpose = None

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None

# Optional imports for enhanced functionality
try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False

try:
    import cupy as cp  # type: ignore[import-not-found]
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False
    cp = None

try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    def njit(func):
        return func  # No-op decorator fallback

try:
    from PIL.ImageCms import createProfile, profileToProfile
    HAS_CMS = True
except ImportError:
    HAS_CMS = False

try:
    import svgwrite
    HAS_SVGWRITE = True
except ImportError:
    HAS_SVGWRITE = False

try:
    from svgpathtools import parse_path, Path, Line, Arc, CubicBezier, QuadraticBezier, svg2paths2
    HAS_SVGPATHTOOLS = True
    SVGPath = Path
except ImportError:
    HAS_SVGPATHTOOLS = False
    SVGPath = None
    svg2paths2 = None

try:
    from reportlab.pdfgen import canvas as rl_canvas
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.utils import ImageReader
    from reportlab.lib.units import inch as RL_INCH
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False
    rl_canvas = None
    ImageReader = None
    # Don't reassign RL_INCH, just let it remain undefined for the except block

# ---------- icon path (resolved relative to script dir, cross-platform) ----------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if sys.platform == "darwin":
    # Prefer .icns on macOS, fall back to .ico / .png
    _ICON_CANDIDATES = ["cmyk_logo.icns", "cmyk_logo.png", "cmyk_logo.ico"]
else:
    _ICON_CANDIDATES = ["cmyk_logo.ico", "cmyk_logo.png"]
ICON_PATH = ""
for _c in _ICON_CANDIDATES:
    _p = os.path.join(_SCRIPT_DIR, _c)
    if os.path.isfile(_p):
        ICON_PATH = _p
        break
if not ICON_PATH:
    # Default even if file doesn't exist yet (keeps downstream code unchanged)
    ICON_PATH = os.path.join(_SCRIPT_DIR, _ICON_CANDIDATES[0])

# ---------- constants ----------
MAX_LONG_SIDE = 7680
DOC_DPI = 240

# High-quality performance optimization constants
# Grid budgets extremely aggressive for impatient customers
GRID_BUDGET_PREVIEW = 25_000      # Increased for better resolution
GRID_BUDGET_IDLE = 80_000         # Increased budget for better idle quality  
GRID_BUDGET_INTERACTIVE = 5_000   # Increased for better interaction quality

# Interactive preview optimization for smooth interactions
SIZE_QUANT_INTERACT = 1           # Maximum detail preservation during interaction
SIZE_QUANT_IDLE = 1               # No quality compromise during idle
PREVIEW_THROTTLE_MS = 100         # 10fps for idle - much more efficient
INTERACTIVE_THROTTLE_MS = 33      # 30fps during interaction - smooth but efficient

# Performance optimization flags
USE_SMART_CACHING = True          # Enable intelligent caching for repeated operations
USE_PARALLEL_PROCESSING = True    # Enable multi-core processing
ADAPTIVE_QUALITY = True           # Dynamically adjust quality based on interaction state

# ---------- theme ----------
QSS = """
QWidget { background:#414042; color:#fff; font-size:10pt; }
QGroupBox { border:1px solid #444; margin-top:8px; padding-top:10px; }
QGroupBox::title { left:8px; color:#fff; text-transform: lowercase; }

QPushButton {
  background:#363436; color:#fff; border:1px solid #444;
  padding:6px 10px; letter-spacing:0.5px; text-transform: lowercase;
  border-radius:6px;
}
QPushButton:hover { background:#4a484a; }

QComboBox {
  background:#363436; border:1px solid #444; padding:2px 6px; color:#fff;
  text-transform: lowercase; border-radius:6px;
}
QComboBox QAbstractItemView {
  background:#363436; color:#fff; selection-background-color:#4a484a;
  border:1px solid #444; border-radius:6px;
}
QLabel { text-transform: lowercase; }

/* Preview widget - border/bevel now on the scroll area wrapper */
#PreviewArea {
  background: #6d6e71;
  border: none;
}

/* flat white bar sliders + smaller black handles */
QSlider::groove:horizontal { height:2px; margin:0px; background:#fff; border:none; }
QSlider::handle:horizontal { width:10px; height:10px; margin:-5px 0; border:2px solid #fff; background:#000; border-radius:5px; }

/* colored CMYK dials */
QDial#dialC { background:#00bcd4; border:2px solid #000; border-radius:21px; }
QDial#dialM { background:#ff4081; border:2px solid #000; border-radius:21px; }
QDial#dialY { background:#ffea00; border:2px solid #000; border-radius:21px; }
QDial#dialK { background:#111;    border:2px solid #fff; border-radius:21px; }

/* QToolBox (Accordion) */
QToolBox {
  border: 1px solid #444;
  border-radius: 6px;
  background: #414042;
  padding-top: 0px;
  color: rgb(255,255,255);
}

QToolBox::tab {
  background: #363436;
  color: rgb(255,255,255);
  border: 1px solid #444;
  border-bottom: none;
  height: 28px;
  padding: 0 14px;
  font-size: 9pt;
  font-weight: 600;
  border-top-left-radius: 6px;
  border-top-right-radius: 6px;
  margin: 4px 6px 2px 6px;
  qproperty-iconSize: 16px 16px;
}

QToolBox::tab:hover {
  background: #4a4a4a;
  color: rgb(255,255,255);
}

QToolBox::tab:selected {
  background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #cc5500, stop:1 #993d00);
  color: rgb(255,255,255);
  font-weight: 700;
  margin-bottom: 2px;
}

/* moiré tile */
#MoireBox { background:#fff; border:1px solid #666; border-radius:6px; }
"""

# ---------- utils ----------
def resource_path(rel: str) -> str:
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        base = sys._MEIPASS  # type: ignore
    else:
        base = os.path.abspath(os.path.dirname(__file__))
    return os.path.join(base, rel)

def _load_app_icon() -> Optional[QIcon]:
    # Try explicit ICON_PATH first, then fallbacks.
    candidates = [ICON_PATH,
                  resource_path("cmyk_logo.ico"),
                  resource_path("cmyk_logo.png"),
                  resource_path("halftone_cmyk.ico"),
                  resource_path("halftone_cmyk_logo.ico"),
                  resource_path("halftone_cmyk.png"),
                  resource_path("halftone_cmyk_logo.png")]
    for pth in candidates:
        try:
            if pth and os.path.isfile(pth):
                ic = QIcon(pth)
                if not ic.isNull():
                    return ic
        except Exception:
            pass
    return None

def pil_to_qpixmap(pil_img) -> QPixmap:
    if pil_img.mode != "RGBA":
        pil_img = pil_img.convert("RGBA")
    w, h = pil_img.size
    data = pil_img.tobytes("raw", "RGBA")
    qimg = QImage(data, w, h, QImage.Format.Format_RGBA8888)
    return QPixmap.fromImage(qimg)

def load_image_capped(path: str):
    if not Image:
        raise ImportError("PIL not available")
    im = Image.open(path)
    
    # Handle transparency: composite onto white background
    if im.mode == 'RGBA':
        white_bg = Image.new('RGBA', im.size, (255, 255, 255, 255))
        im = Image.alpha_composite(white_bg, im)
    elif im.mode == 'LA':
        im_rgba = im.convert('RGBA')
        white_bg = Image.new('RGBA', im_rgba.size, (255, 255, 255, 255))
        im = Image.alpha_composite(white_bg, im_rgba)
    elif im.mode == 'P':
        if 'transparency' in im.info:
            im = im.convert('RGBA')
            white_bg = Image.new('RGBA', im.size, (255, 255, 255, 255))
            im = Image.alpha_composite(white_bg, im)
        else:
            im = im.convert('RGBA')
    else:
        im = im.convert('RGBA')

    w, h = im.size
    long_side = max(w, h)
    if long_side > MAX_LONG_SIDE:
        s = MAX_LONG_SIDE / long_side
        # Use faster NEAREST resampling for speed
        im = im.resize((int(w*s), int(h*s)), Image.Resampling.NEAREST)
    return im

def find_cmyk_icc() -> Optional[str]:
    env = os.environ.get("CMYK_ICC")
    if env and os.path.isfile(env): return env
    here = os.path.abspath(os.path.dirname(sys.argv[0] if getattr(sys, 'frozen', False) else __file__))
    for name in ("cmyk.icc","USWebCoatedSWOP.icc","CoatedFOGRA39.icc","GRACoL2006_Coated1v2.icc"):
        p = os.path.join(here, name)
        if os.path.isfile(p): return p
    if sys.platform.startswith("win"):
        search = [r"C:\Windows\System32\spool\drivers\color"]
    else:
        search = ["/Library/ColorSync/Profiles","/System/Library/ColorSync/Profiles"]
    for folder in search:
        try:
            for fn in os.listdir(folder):
                if fn.lower().endswith((".icc",".icm")) and any(k in fn.lower() for k in ("swop","gracol","fogra","cmyk")):
                    return os.path.join(folder, fn)
        except Exception:
            pass
    return None

def rgb_to_cmyk_saturated(img_rgba):
    """Convert RGB to CMYK with better saturation preservation.
    
    PIL's default conversion desaturates colors significantly.
    This uses a UCR/GCR approach that preserves more saturation.
    """
    if not np:
        return img_rgba.convert("CMYK")
    
    # Convert to RGB if needed, get numpy array
    img_rgb = img_rgba.convert("RGB")
    rgb = np.asarray(img_rgb, dtype=np.float32) / 255.0
    
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    
    # Calculate CMY (complementary colors)
    c = 1.0 - r
    m = 1.0 - g
    y = 1.0 - b
    
    # Calculate K (black) using UCR (Under Color Removal)
    # Use minimum of CMY as the black component
    k = np.minimum(np.minimum(c, m), y)
    
    # Apply GCR (Gray Component Replacement) with reduced black generation
    # This preserves more saturation in the CMY channels
    # gcr_factor controls how much black replaces CMY (0.0 = no black, 1.0 = full UCR)
    gcr_factor = 0.5  # Balanced - preserves saturation while still using black
    k_adjusted = k * gcr_factor
    
    # Remove the black component from CMY
    # Avoid division by zero
    divisor = np.where(k < 1.0, 1.0 - k, 1.0)
    c = np.clip((c - k_adjusted) / divisor, 0.0, 1.0)
    m = np.clip((m - k_adjusted) / divisor, 0.0, 1.0)
    y = np.clip((y - k_adjusted) / divisor, 0.0, 1.0)
    k = k_adjusted
    
    # Stack and convert to uint8
    cmyk = np.stack([c, m, y, k], axis=-1)
    cmyk_uint8 = (cmyk * 255.0).astype(np.uint8)
    
    if Image is None:
        raise ImportError("PIL not available")
    return Image.fromarray(cmyk_uint8, mode="CMYK")

def rgba_to_cmyk_with_icc(img_rgba, icc_path_opt: Optional[str] = None):
    """Convert RGBA to CMYK using ICC profile if available, otherwise use saturated conversion."""
    if not HAS_CMS:
        return rgb_to_cmyk_saturated(img_rgba)
    try:
        from PIL import ImageCms
        srgb = ImageCms.createProfile("sRGB")
        icc_path = icc_path_opt or find_cmyk_icc()
        if icc_path and os.path.isfile(icc_path):
            cmyk = ImageCms.getOpenProfile(icc_path)
            xform = ImageCms.buildTransformFromOpenProfiles(srgb, cmyk, "RGBA", "CMYK", renderingIntent=ImageCms.Intent.PERCEPTUAL)
            transformed_img = ImageCms.applyTransform(img_rgba, xform)
            if transformed_img is not None:
                return transformed_img
        # Fall back to saturated conversion (better than PIL's naive conversion)
        return rgb_to_cmyk_saturated(img_rgba)
    except Exception:
        return rgb_to_cmyk_saturated(img_rgba)

# ---------- data ----------
@dataclass
class Params:
    mode: str
    cell: float
    elem: float
    stroke: float
    contrast_pct: float
    brightness_pct: float
    smoothing_pct: float
    preview_channel: str
    ang_c: float
    ang_m: float
    ang_y: float
    ang_k: float
    full_composite_preview: bool
    regs_on: bool
    reg_size_px: float
    reg_offset_px: float
    grayscale_mode: bool
    invert_gray: bool
    mirror_image: bool
    dot_gap_pct: float = 0.0  # percent increase in spacing between dot centers (does not shrink dots)
    # Dithering parameters
    dither_enabled: bool = False
    dither_method: str = "bayer 4x4"
    dither_amount: float = 50.0
    dither_threshold: float = 50.0
    # Dot scaling parameters
    scaling_centers: list = field(default_factory=list)  # List of (x, y, radius, scale_amount, falloff) tuples
    # Channel opacity (for PDF/TIFF export only)
    opacity_c: float = 100.0
    opacity_m: float = 100.0
    opacity_y: float = 100.0
    opacity_k: float = 100.0
    # Custom channel colors (RGBA tuples)
    color_c: tuple = (0, 255, 255, 255)
    color_m: tuple = (255, 0, 255, 255)
    color_y: tuple = (255, 255, 0, 255)
    color_k: tuple = (0, 0, 0, 255)

@dataclass
class Transform:
    scale_pct: int
    offset_x_px: int
    offset_y_px: int
    orientation: str

# ---------- preview widget ----------
class PreviewArea(QLabel):
    def __init__(self, main_window, on_wheel, on_drag_delta, get_art_size_px):
        super().__init__()
        self.main_window = main_window
        self.on_wheel = on_wheel
        self.on_drag_delta = on_drag_delta
        self.get_art_size_px = get_art_size_px
        self.setObjectName("PreviewArea")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._dragging = False
        self._last_pos: Optional[QPoint] = None
        self._pixmap: Optional[QPixmap] = None
        self.zoom_scale: float = 1.0
        # Dot scaling support
        self.scaling_centers: list = []  # List of (x, y, radius, scale_amount, falloff) tuples
        self._mouse_pos: Optional[QPoint] = None  # Track mouse position for radius circle
        # Set default cursor and track mouse for cursor updates
        self.setMouseTracking(True)
        self._update_cursor()

    def setPixmap(self, pm: QPixmap | QImage):
        if isinstance(pm, QImage):
            self._pixmap = QPixmap.fromImage(pm)
        else:
            self._pixmap = pm
        self._update_minimum_size()
        self.update()

    def set_zoom(self, z: float):
        self.zoom_scale = max(0.1, min(4.0, float(z)))
        self._update_minimum_size()
        self.update()

    def _update_minimum_size(self):
        """Resize widget to zoomed content size so QScrollArea scrollbars appear."""
        if self._pixmap and not self._pixmap.isNull():
            z = self.zoom_scale
            w = int(self._pixmap.width() * z)
            h = int(self._pixmap.height() * z)
            self.setFixedSize(max(w, 1), max(h, 1))
        else:
            # No content yet — don't constrain; let scroll area handle it
            self.setMinimumSize(0, 0)
            self.setMaximumSize(16777215, 16777215)  # QWIDGETSIZE_MAX

    def _update_cursor(self):
        """Update cursor based on current tool state"""
        if hasattr(self.main_window, 'dot_scale_enabled') and self.main_window.dot_scale_enabled.isChecked():
            # When dot scaling is enabled, show crosshair cursor
            self.setCursor(Qt.CursorShape.CrossCursor)
        else:
            # Default to crosshair cursor for precision work
            self.setCursor(Qt.CursorShape.CrossCursor)

    def enterEvent(self, event):
        """Called when mouse enters the widget"""
        self._update_cursor()
        super().enterEvent(event)

    def leaveEvent(self, event):
        """Called when mouse leaves the widget"""
        self._mouse_pos = None
        self.update()  # Refresh to hide radius circle
        super().leaveEvent(event)

    def wheelEvent(self, e):
        self.on_wheel(+1 if e.angleDelta().y() > 0 else -1)

    def mousePressEvent(self, e: QMouseEvent):
        if e.button() == Qt.MouseButton.LeftButton:
            # Check if dot scaling tool is enabled
            if hasattr(self.main_window, 'dot_scale_enabled') and self.main_window.dot_scale_enabled.isChecked():
                # Convert widget coordinates to artboard coordinates
                artboard_pos = self._widget_to_artboard_coords(e.position().toPoint())
                if artboard_pos:
                    # Add new scaling center
                    radius = float(self.main_window.scale_radius_s.value())
                    amount = float(self.main_window.scale_amount_s.value()) / 100.0
                    falloff = self.main_window.scale_falloff.currentText()
                    
                    self.scaling_centers.append((artboard_pos[0], artboard_pos[1], radius, amount, falloff))
                    self.main_window.clear_scaling_btn.setEnabled(True)
                    self.main_window._mark_halftone_dirty()
                    self.main_window.update_preview(force=True)
                return
            
            # Normal pan behavior
            self._dragging = True
            self._last_pos = e.position().toPoint()
            # Change cursor to grabbing hand during drag
            self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def mouseMoveEvent(self, e: QMouseEvent):
        # Always track mouse position for radius circle
        self._mouse_pos = e.position().toPoint()
        
        if self._dragging and self._last_pos is not None:
            cur = e.position().toPoint()
            dx = cur.x() - self._last_pos.x()
            dy = cur.y() - self._last_pos.y()
            self._last_pos = cur
            self.on_drag_delta(dx, dy)
        else:
            # Update display to show radius circle if scaling tool is enabled
            if hasattr(self.main_window, 'dot_scale_enabled') and self.main_window.dot_scale_enabled.isChecked():
                self.update()

    def mouseReleaseEvent(self, e: QMouseEvent):
        if e.button() == Qt.MouseButton.LeftButton:
            self._dragging = False
            self._last_pos = None
            # Restore appropriate cursor
            self._update_cursor()

    def _widget_to_artboard_coords(self, widget_pos: QPoint) -> Optional[Tuple[float, float]]:
        """Convert widget click position to artboard pixel coordinates"""
        if not self._pixmap:
            return None
            
        # Account for zoom and centering
        z = self.zoom_scale
        pm_w, pm_h = self._pixmap.width(), self._pixmap.height()
        draw_w, draw_h = int(pm_w * z), int(pm_h * z)
        label_w, label_h = self.width(), self.height()
        ox = (label_w - draw_w) // 2
        oy = (label_h - draw_h) // 2
        
        # Convert to artboard coordinates
        artboard_x = (widget_pos.x() - ox) / z
        artboard_y = (widget_pos.y() - oy) / z
        
        if 0 <= artboard_x < pm_w and 0 <= artboard_y < pm_h:
            return (artboard_x, artboard_y)
        return None

    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        path = QPainterPath()
        path.addRoundedRect(self.rect().adjusted(0, 0, -1, -1), 8, 8)
        p.setClipPath(path)
        if self._pixmap is None or self._pixmap.isNull():
            p.end(); return

        pm_w, pm_h = self._pixmap.width(), self._pixmap.height()
        z = self.zoom_scale
        draw_w, draw_h = int(pm_w * z), int(pm_h * z)
        label_w, label_h = self.width(), self.height()
        ox = (label_w - draw_w) // 2
        oy = (label_h - draw_h) // 2

        target = QRect(ox, oy, draw_w, draw_h)
        p.drawPixmap(target, self._pixmap)

        # Draw the artboard border to exactly match the pixmap being displayed.
        # This keeps the red border aligned during reduced-resolution previews.
        p.setPen(QPen(QColor(200, 40, 40, 255), 2, Qt.PenStyle.SolidLine))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRect(target)
        
        # Draw scaling area indicators if scaling is enabled
        if self.main_window.dot_scale_enabled.isChecked() and self.scaling_centers:
            transform = self._get_transform()
            if transform:
                # Draw scaling areas
                for x, y, radius, scale_amount, falloff in self.scaling_centers:
                    # Convert artboard coordinates to widget coordinates
                    widget_x = int(ox + x * z * transform.m11())
                    widget_y = int(oy + y * z * transform.m22())
                    widget_radius = int(radius * z * min(transform.m11(), transform.m22()))
                    
                    # Draw the scaling area circle
                    p.setPen(QPen(QColor(100, 200, 255, 150), 2, Qt.PenStyle.DashLine))
                    p.setBrush(Qt.BrushStyle.NoBrush)
                    circle_rect = QRect(
                        widget_x - widget_radius, 
                        widget_y - widget_radius,
                        widget_radius * 2, 
                        widget_radius * 2
                    )
                    p.drawEllipse(circle_rect)
                    
                    # Draw the center point
                    p.setPen(QPen(QColor(100, 200, 255, 200), 3, Qt.PenStyle.SolidLine))
                    p.setBrush(QBrush(QColor(100, 200, 255, 100)))
                    center_size = 6
                    center_rect = QRect(
                        widget_x - center_size // 2,
                        widget_y - center_size // 2,
                        center_size,
                        center_size
                    )
                    p.drawEllipse(center_rect)
                    
                    # Draw scale amount text near the center
                    p.setPen(QPen(QColor(255, 255, 255, 200), 1, Qt.PenStyle.SolidLine))
                    p.setFont(QFont("Arial", 10, QFont.Weight.Bold))
                    scale_text = f"{scale_amount:.0f}%"
                    text_rect = QRect(widget_x + 10, widget_y - 15, 50, 20)
                    p.drawText(text_rect, Qt.AlignmentFlag.AlignLeft, scale_text)
        
        # Draw radius circle at mouse position when scaling tool is enabled
        if (self.main_window.dot_scale_enabled.isChecked() and 
            self._mouse_pos is not None and 
            hasattr(self.main_window, 'scale_radius_s')):
            transform = self._get_transform()
            if transform:
                # Get current radius setting
                radius = self.main_window.scale_radius_s.value()
                widget_radius = int(radius * z * min(transform.m11(), transform.m22()))
                
                # Draw radius circle at mouse position
                p.setPen(QPen(QColor(255, 100, 100, 120), 2, Qt.PenStyle.SolidLine))
                p.setBrush(Qt.BrushStyle.NoBrush)
                circle_rect = QRect(
                    self._mouse_pos.x() - widget_radius,
                    self._mouse_pos.y() - widget_radius,
                    widget_radius * 2,
                    widget_radius * 2
                )
                p.drawEllipse(circle_rect)
        
        p.end()

    def _get_transform(self):
        return self.main_window._get_transform()

# ---------- main ----------
class Main(QMainWindow):
    def __init__(self):
        super().__init__()

        self.ART_SIZES = {
            "Letter (8.5×11)": (8.5, 11.0),
            "A4": (8.27, 11.69),
            "8×10 in" : (8.0, 10.0),
            "9×12 in" : (9.0, 12.0),
            "11×15 in": (11.0, 15.0),
            "11×17 in": (11.0, 17.0),
            "13×19 in": (13.0, 19.0),
            "15×22 in": (15.0, 22.0),
        }

        self.setWindowTitle("drc_halftone_cmyk")
        self.setMinimumSize(QSize(1220, 880))

        # icon (absolute path first) - enhanced for Windows taskbar
        ic = _load_app_icon()
        if ic:
            self.setWindowIcon(ic)
        # We'll also announce success/failure after status bar is created.

        self.cmyk_icc_path: Optional[str] = find_cmyk_icc()
        self.img_full_rgba = None
        self._prev_pix = None

        self.shape_paths = None
        self.shape_bbox = None
        self.rotate_shape = True
        self.mirror_direction: str = "horizontal"  # "horizontal" or "vertical"

        self.reg_paths = None
        self.reg_bbox = None

        # Performance-optimized cache system
        self._art_cache = {}
        self._cmyk_cache = {}
        self._arr_cache = {}
        self._grid_cache = {}
        self._sprite_cache = {}
        self._rotated_sprite_cache = {}
        self._mask_cache = {}

        self._last_render_base_rgba = None
        self._halftone_dirty = True
        self._threads = max(2, min(8, (cpu_count() or 4)))

        self.live_preview = True
        self.interacting = False
        self._updating = False  # Prevent redundant updates
        self._interaction_start_time = None  # Track interaction timing for optimizations
        self._interaction_scale = 1.0  # Scale factor for interaction performance
        self.preview_timer = QTimer(self)
        self.preview_timer.setSingleShot(True)
        self.preview_timer.timeout.connect(lambda: self.update_preview(force=False))
        
        self._offset_x = 0
        self._offset_y = 0

        # --- UI ---
        open_btn = QPushButton("file"); open_btn.clicked.connect(self.on_file_menu)
        open_btn.setToolTip("File operations: Open image, Save/Load projects")
        export_svg_btn = QPushButton(".svg"); export_svg_btn.clicked.connect(self.on_export_svg); export_svg_btn.setEnabled(False)
        export_png_btn = QPushButton(".png"); export_png_btn.clicked.connect(self.on_export_png); export_png_btn.setEnabled(False)
        export_pdf_btn = QPushButton(".pdf"); export_pdf_btn.clicked.connect(self.on_export_pdf); export_pdf_btn.setEnabled(False)
        export_tiff_btn = QPushButton(".tiff"); export_tiff_btn.clicked.connect(self.on_export_tiff_cmyk); export_tiff_btn.setEnabled(False)
        self.export_svg_btn = export_svg_btn; self.export_png_btn = export_png_btn
        self.export_pdf_btn = export_pdf_btn; self.export_tiff_btn = export_tiff_btn

        self.mode = QComboBox()
        self.mode.addItems(["dot","circle","circle outline","square","diamond","triangle","cross","lines"])
        self.mode.setToolTip("Halftone dot shape\n• Dot: Classic round halftone dots\n• Circle: Perfect circles\n• Square: Modern geometric look\n• Lines: Linear screen printing style")
        self.mode.currentTextChanged.connect(lambda _t: (self._mark_halftone_dirty(), self._on_control_changed(), self._update_moire_tile()))
        self.preview_chan = QComboBox(); self.preview_chan.addItems(["composite","c","m","y","k"])
        self.preview_chan.setToolTip("Preview channel\n• Composite: Full CMYK result\n• C/M/Y/K: Individual color plates")
        self.preview_chan.currentTextChanged.connect(lambda _t: (self._mark_halftone_dirty(), self._on_control_changed()))
        self.full_comp = QCheckBox("full composite preview"); self.full_comp.setChecked(True)
        self.full_comp.setToolTip("Show complete CMYK composite\n• Checked: Realistic print simulation\n• Unchecked: Faster individual plate preview")
        self.full_comp.stateChanged.connect(lambda _v:(self._mark_halftone_dirty(), self._on_control_changed()))

        # >>> Label updates here <<<
        self.gray_chk = QCheckBox("halftone (b/w)"); self.gray_chk.setChecked(False)
        self.gray_chk.stateChanged.connect(self._on_gray_toggled)
        self.mirror_chk = QCheckBox("mirror image"); self.mirror_chk.setChecked(False)
        self.mirror_chk.setToolTip("Mirror (flip) the image horizontally\nGreat for woodblock prints")
        self.mirror_chk.stateChanged.connect(self._on_mirror_toggled)
        self.invert_chk = QCheckBox("invert"); self.invert_chk.setChecked(False)
        self.invert_chk.setToolTip("Invert image tones\n• Unchecked: Normal (dark = more dots)\n• Checked: Inverted (dark = fewer dots)")
        self.invert_chk.stateChanged.connect(lambda _v: (self._mark_halftone_dirty(), self._on_control_changed(), self._update_moire_tile()))

        self.art_group = QGroupBox("")
        art = QVBoxLayout()
        orow = QHBoxLayout()
        orow.addWidget(QLabel("orientation"))
        self.orientation = QComboBox(); self.orientation.addItems(["portrait","landscape"])
        self.orientation.currentTextChanged.connect(self._on_artboard_changed)
        self.center_btn = QPushButton("center"); self.center_btn.clicked.connect(self.center_offsets)
        self.fit_image_btn = QPushButton("fit image"); self.fit_image_btn.clicked.connect(self.fit_image_to_artboard)
        orow.addWidget(self.orientation)
        orow.addWidget(QLabel("size"))
        self.art_size = QComboBox()
        for name in self.ART_SIZES.keys(): self.art_size.addItem(name)
        self.art_size.setCurrentText("11×15 in")
        self.art_size.currentTextChanged.connect(self._on_artboard_changed)
        orow.addWidget(self.art_size); orow.addStretch(1); orow.addWidget(self.center_btn); orow.addWidget(self.fit_image_btn)
        art.addLayout(orow)
        srow = QHBoxLayout()
        srow.addWidget(QLabel("scale"))
        self.scale_s = QSlider(Qt.Orientation.Horizontal); self.scale_s.setRange(10, 400); self.scale_s.setValue(100)
        self.scale_val = QLabel("100%")
        def _on_scale(v):
            self.scale_val.setText(f"{v}%"); self._mark_halftone_dirty(); self._on_control_changed()
        self.scale_s.valueChanged.connect(_on_scale)
        self.scale_s.sliderPressed.connect(self._on_slider_pressed); self.scale_s.sliderReleased.connect(self._on_slider_released)
        srow.addWidget(self.scale_s,1); srow.addWidget(self.scale_val)
        art.addLayout(srow)
        self.art_group.setLayout(art)

        angles_group = QGroupBox("cmyk angles")
        ag = QVBoxLayout()
        preset_row = QHBoxLayout()
        preset_row.addWidget(QLabel("presets"))
        self.preset_combo = QComboBox()
        self.PRESETS = {
            "preset 1 (c15 m75 y0 k45)"  : (15, 75, 0, 45),
            "preset 2 (c105 m75 y90 k15)": (105, 75, 90, 15),
            "preset 3 (c15 m45 y0 k75)"  : (15, 45, 0, 75),
            "preset 4 (c165 m45 y90 k105)": (165, 45, 90, 105),
        }
        for name in self.PRESETS.keys(): self.preset_combo.addItem(name)
        self.preset_combo.setToolTip("CMYK angle presets\n• Preset 1: Traditional offset printing\n• Preset 2: Alternative moiré reduction\n• Preset 3: High contrast layout\n• Preset 4: Experimental angles")
        self.preset_combo.currentTextChanged.connect(self._apply_preset)
        preset_row.addWidget(self.preset_combo,1)
        ag.addLayout(preset_row)
        grid = QGridLayout(); grid.setHorizontalSpacing(18); grid.setVerticalSpacing(6)
        def make_dial(obj_name:str, init:int):
            dial = QDial(); dial.setRange(0,180); dial.setValue(init)
            dial.setNotchesVisible(True); dial.setFixedSize(32,32)
            dial.setObjectName(obj_name)
            lab = QLabel(f"{init}°"); lab.setAlignment(Qt.AlignmentFlag.AlignCenter)
            def on_change(v, lab=lab): lab.setText(f"{v}°"); self._mark_halftone_dirty(); self._on_control_changed(); self._update_moire_tile()
            dial.valueChanged.connect(on_change)
            dial.sliderPressed.connect(self._on_slider_pressed); dial.sliderReleased.connect(self._on_slider_released)
            return dial, lab
        self.ang_c, lc = make_dial("dialC", 15)
        self.ang_m, lm = make_dial("dialM", 75)
        self.ang_y, ly = make_dial("dialY", 0)
        self.ang_k, lk = make_dial("dialK", 45)
        grid.addWidget(self.ang_c,0,0); grid.addWidget(self.ang_m,0,1)
        grid.addWidget(lc,1,0);       grid.addWidget(lm,1,1)
        grid.addWidget(self.ang_y,2,0); grid.addWidget(self.ang_k,2,1)
        grid.addWidget(ly,3,0);         grid.addWidget(lk,3,1)
        self.moire_lbl = QLabel(); self.moire_lbl.setObjectName("MoireBox")
        self.moire_lbl.setFixedSize(100, 100)
        grid.addWidget(self.moire_lbl, 0, 2, 4, 1)
        ag.addLayout(grid)
        angles_group.setLayout(ag)

        def row(label,minv,maxv,init,fmt, on_change_cb=None):
            h=QHBoxLayout(); h.addWidget(QLabel(label))
            s=QSlider(Qt.Orientation.Horizontal); s.setRange(minv,maxv); s.setValue(init)
            
            # Editable value input instead of label
            v=QLineEdit(fmt(init)); v.setFixedWidth(80)
            v.setAlignment(Qt.AlignmentFlag.AlignRight)
            
            # Parse the formatted value back to slider value
            def parse_value(text):
                try:
                    # Remove common suffixes and parse
                    clean = text.replace('px','').replace('%','').replace('in','').replace('°','').strip()
                    val = float(clean)
                    # Reverse the format transformation
                    if 'px' in fmt(init) or 'in' in fmt(init):
                        val = val * 100  # cell size, stroke use /100
                    return int(round(val))
                except:
                    return s.value()
            
            def on_slider_change(x, lab=v):
                lab.blockSignals(True)
                lab.setText(fmt(x))
                lab.blockSignals(False)
                if on_change_cb: on_change_cb()
                else:
                    self._mark_halftone_dirty(); self._on_control_changed(); self._update_moire_tile()
            
            def on_edit_finished():
                val = parse_value(v.text())
                val = max(minv, min(maxv, val))  # Clamp to range
                s.blockSignals(True)
                s.setValue(val)
                s.blockSignals(False)
                v.setText(fmt(val))
                if on_change_cb: on_change_cb()
                else:
                    self._mark_halftone_dirty(); self._on_control_changed(); self._update_moire_tile()
            
            s.valueChanged.connect(on_slider_change)
            v.editingFinished.connect(on_edit_finished)
            s.sliderPressed.connect(self._on_slider_pressed); s.sliderReleased.connect(self._on_slider_released)
            h.addWidget(s,1); h.addWidget(v); return h,s,v

        # size / stroke sliders
        cell_row, self.cell_s, self.cell_lbl = row("cell size", 25, 4000, 1600, lambda x:f"{x/100:.2f}px")
        self.cell_s.setToolTip("Halftone cell size (dot spacing)\n• Smaller: Higher resolution, more dots, finer detail\n• Larger: Lower resolution, fewer dots, bolder look\n• 16px typical for 150 LPI printing")
        str_row,  self.str_s,  self.str_lbl  = row("stroke width", 25, 1000, 100, lambda x:f"{x/100:.2f}px")
        self.str_s.setToolTip("Outline thickness for outlined shapes\n• Only affects 'circle outline' mode\n• Thicker outlines = bolder appearance")

        # registration helpers
        def px_to_in(px): return px / DOC_DPI
        def fmt_px_in(v): return f"{px_to_in(v):.2f}in"
        reg_size_row,   self.reg_size_s,   self.reg_size_lbl   = row(
            "reg size", 20, 600, 120, lambda v: fmt_px_in(v),
            on_change_cb=lambda: self._on_regs_changed()
        )
        reg_offset_row, self.reg_offset_s, self.reg_offset_lbl = row(
            "reg offset", 10, 1000, int(1.0*DOC_DPI), lambda v: fmt_px_in(v),
            on_change_cb=lambda: self._on_regs_changed()
        )
        reg_row1 = QHBoxLayout()
        self.regs_chk = QCheckBox("add reg marks"); self.regs_chk.setChecked(True)
        self.regs_chk.stateChanged.connect(lambda _v: self._on_regs_changed())
        self.load_reg_btn = QPushButton("load reg mark…"); self.load_reg_btn.clicked.connect(self.on_load_reg)
        if not HAS_SVGPATHTOOLS:
            self.load_reg_btn.setEnabled(False)
            self.load_reg_btn.setToolTip("Install 'svgpathtools' to load a custom reg mark SVG.")
        reg_row1.addWidget(self.regs_chk); reg_row1.addWidget(self.load_reg_btn); reg_row1.addStretch(1)

        # Dot scaling widgets (kept for compatibility but not shown in UI)
        self.dot_scale_enabled = QCheckBox("enable dot scaling")
        self.dot_scale_enabled.setChecked(False)  # Disabled by default
        self.scale_amount_s = QSlider(Qt.Orientation.Horizontal)
        self.scale_amount_s.setRange(100, 1000)
        self.scale_amount_s.setValue(150)
        self.scale_radius_s = QSlider(Qt.Orientation.Horizontal)
        self.scale_radius_s.setRange(20, 200)
        self.scale_radius_s.setValue(60)
        self.scale_falloff = QComboBox()
        self.scale_falloff.addItems(["linear", "smooth", "sharp"])
        self.scale_falloff.setCurrentText("smooth")
        self.clear_scaling_btn = QPushButton("clear all scaling")
        self.clear_scaling_btn.clicked.connect(self._clear_scaling_areas)
        self.clear_scaling_btn.setEnabled(False)
        
        # ===== FLATTENED UI SYSTEM =====
        
        # Create scroll area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setContentsMargins(0, 0, 0, 0)
        
        # --- SECTION 1: CMYK + Halftone + Registration ---
        cmyk_halftone_layout = QVBoxLayout()
        
        # Mode and preview controls
        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("dot settings")); mode_row.addWidget(self.mode)
        mode_row.addWidget(QLabel("preview")); mode_row.addWidget(self.preview_chan)
        mode_row.addWidget(self.full_comp); mode_row.addStretch(1)
        cmyk_halftone_layout.addLayout(mode_row)
        
        # Group halftone (b/w), invert, and mirror checkboxes together
        gray_row = QHBoxLayout()
        gray_row.addWidget(self.gray_chk); gray_row.addWidget(self.invert_chk); gray_row.addWidget(self.mirror_chk); gray_row.addStretch(1)
        cmyk_halftone_layout.addLayout(gray_row)
        
        cmyk_halftone_layout.addSpacing(8)
        
        # Initialize channel color values (used by color swatches below)
        self.color_c = (0, 255, 255, 255)
        self.color_m = (255, 0, 255, 255)
        self.color_y = (255, 255, 0, 255)
        self.color_k = (0, 0, 0, 255)
        
        # Pre-declare opacity and swatch attributes (populated in loop below)
        self.opacity_c: QComboBox = QComboBox()
        self.opacity_m: QComboBox = QComboBox()
        self.opacity_y: QComboBox = QComboBox()
        self.opacity_k: QComboBox = QComboBox()
        self.swatch_c: QPushButton = QPushButton()
        self.swatch_m: QPushButton = QPushButton()
        self.swatch_y: QPushButton = QPushButton()
        self.swatch_k: QPushButton = QPushButton()
        
        cmyk_halftone_layout.addWidget(angles_group)
        cmyk_halftone_layout.addSpacing(6)
        
        # Channel opacity & color pickers (between angles and halftone)
        opacity_values = [f"{i}%" for i in range(100, -1, -5)]
        
        chan_grid = QGridLayout()
        chan_grid.setSpacing(4)
        
        for col, (ch, ch_upper) in enumerate([('c','C'), ('m','M'), ('y','Y'), ('k','K')]):
            lbl = QLabel(ch_upper)
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            chan_grid.addWidget(lbl, 0, col)
            
            combo = QComboBox()
            combo.addItems(opacity_values)
            combo.setCurrentText("100%")
            combo.setFixedWidth(70)
            combo.currentTextChanged.connect(lambda _, c=ch: (self._mark_halftone_dirty(), self._on_control_changed(), self._update_moire_tile()))
            setattr(self, f'opacity_{ch}', combo)
            chan_grid.addWidget(combo, 1, col)
            
            swatch = QPushButton()
            swatch.setFixedSize(70, 26)
            swatch.setCursor(Qt.CursorShape.PointingHandCursor)
            swatch.setToolTip(f"Click to choose {ch_upper} channel color")
            self._update_swatch_style(swatch, getattr(self, f'color_{ch}'))
            swatch.clicked.connect(lambda checked=False, c=ch: self._pick_channel_color(c))
            setattr(self, f'swatch_{ch}', swatch)
            chan_grid.addWidget(swatch, 2, col)
        
        chan_grid.setColumnStretch(4, 1)
        cmyk_halftone_layout.addLayout(chan_grid)
        
        cmyk_halftone_layout.addSpacing(8)
        cmyk_halftone_layout.addWidget(QLabel("halftone"))
        
        # Shape controls at top of halftone section
        shp_row = QHBoxLayout()
        self.load_shape_btn = QPushButton("load dot shape…"); self.load_shape_btn.clicked.connect(self.on_load_shape)
        self.shape_rot_chk = QCheckBox("rotate shape with grid"); self.shape_rot_chk.setChecked(True)
        self.shape_rot_chk.setToolTip("Rotate dot shape to match halftone angle\n• Checked: Shapes align with screen angle\n• Unchecked: Shapes stay upright\n• Important for diamond/triangle shapes")
        self.shape_rot_chk.stateChanged.connect(lambda _v:(setattr(self,'rotate_shape', self.shape_rot_chk.isChecked()), self._clear_sprite_caches(), self._mark_halftone_dirty(), self._on_control_changed(), self._update_moire_tile()))
        if not HAS_SVGPATHTOOLS:
            self.load_shape_btn.setEnabled(False)
            self.load_shape_btn.setToolTip("Install 'svgpathtools' to use custom shapes.")
        else:
            self.load_shape_btn.setToolTip("Load custom SVG shape for halftone dots\nSupports any SVG path for unique effects")
        shp_row.addWidget(self.load_shape_btn); shp_row.addWidget(self.shape_rot_chk); shp_row.addStretch(1)
        cmyk_halftone_layout.addLayout(shp_row)
        
        # Add all halftone sliders
        cmyk_halftone_layout.addLayout(cell_row); cmyk_halftone_layout.addLayout(str_row)
        
        # Add registration controls
        cmyk_halftone_layout.addSpacing(8)
        cmyk_halftone_layout.addWidget(QLabel("registration"))
        cmyk_halftone_layout.addLayout(reg_row1); cmyk_halftone_layout.addLayout(reg_size_row); cmyk_halftone_layout.addLayout(reg_offset_row)
        
        # Wrap in GroupBox
        group1 = QGroupBox("cmyk + halftone + registration")
        group1.setLayout(cmyk_halftone_layout)
        scroll_layout.addWidget(group1)
        
        scroll_layout.addStretch(1) # Push everything up
        
        scroll_area.setWidget(scroll_content)

        # Create the left panel
        self.info = QLabel(""); self.info.setStyleSheet("color:#bbb;")
        left = QVBoxLayout()
        
        # File/Export buttons at the very top
        topBtns = QHBoxLayout()
        topBtns.addWidget(open_btn); topBtns.addWidget(export_svg_btn); topBtns.addWidget(export_png_btn); topBtns.addWidget(export_pdf_btn); topBtns.addWidget(export_tiff_btn)
        left.addLayout(topBtns)
        
        left.addWidget(self.art_group)  # Artboard controls
        left.addWidget(scroll_area, 1)   # Scroll area takes remaining space
        left.addWidget(self.info, 0)     # Info at bottom
        left_w = QWidget(); left_w.setLayout(left); left_w.setMinimumWidth(560); left_w.setMaximumWidth(600)

        self.prev_lbl = PreviewArea(self, self.on_wheel_zoom, self.on_drag_delta, self._artboard_size_px_from_transform)
        
        # Wrap preview in a scroll area for scrollbars when zoomed in
        self.preview_scroll = QScrollArea()
        self.preview_scroll.setWidget(self.prev_lbl)
        self.preview_scroll.setWidgetResizable(False)
        self.preview_scroll.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_scroll.setStyleSheet(
            "QScrollArea { background: #6d6e71; border: 6px solid #555; border-radius: 14px; "
            "border-top: 6px solid #2e2f31; border-left: 6px solid #2e2f31; "
            "border-right: 6px solid #2e2f31; border-bottom: 6px solid #2e2f31; }"
        )
        
        ztop = QHBoxLayout()
        self.zoom_minus = QPushButton("–"); self.zoom_minus.setFixedWidth(28)
        self.zoom_plus  = QPushButton("+"); self.zoom_plus.setFixedWidth(28)
        self.zoom_fit   = QPushButton("fit")
        self.zoom_100   = QPushButton("100%")
        ztop.addWidget(QLabel("zoom")); ztop.addWidget(self.zoom_minus)
        self.zoom = QSlider(Qt.Orientation.Horizontal); self.zoom.setRange(10, 400); self.zoom.setValue(100)
        ztop.addWidget(self.zoom,1); ztop.addWidget(self.zoom_plus); ztop.addWidget(self.zoom_fit); ztop.addWidget(self.zoom_100)
        def _zoom_apply(): self.prev_lbl.set_zoom(self.zoom.value()/100.0)
        self.zoom.valueChanged.connect(lambda _v: _zoom_apply())
        self.zoom.sliderPressed.connect(self._on_slider_pressed); self.zoom.sliderReleased.connect(self._on_slider_released)
        self.zoom_minus.clicked.connect(lambda: self.zoom.setValue(max(self.zoom.minimum(), self.zoom.value()-10)))
        self.zoom_plus.clicked.connect(lambda: self.zoom.setValue(min(self.zoom.maximum(), self.zoom.value()+10)))
        self.zoom_100.clicked.connect(lambda: self.zoom.setValue(100))
        self.zoom_fit.clicked.connect(self.fit_view_to_window)

        right = QVBoxLayout()
        right.addWidget(self.art_group)
        right.addWidget(self.preview_scroll, 1)
        right.addLayout(ztop)
        right_w = QWidget(); right_w.setLayout(right)

        # Create Instagram button
        instagram_btn = QPushButton("@drc_art")
        instagram_btn.setToolTip("Visit @drc_art on Instagram")
        instagram_btn.setStyleSheet("""
            QPushButton {
                background-color: #363436;
                color: white;
                border: 1px solid #555;
                border-radius: 15px;
                padding: 8px 15px;
                font-weight: bold;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #454447;
                border-color: #666;
            }
            QPushButton:pressed {
                background-color: #2a2729;
                border-color: #777;
            }
        """)
        instagram_btn.clicked.connect(self.open_instagram)
        instagram_btn.setMaximumWidth(100)

        # Create bottom layout with Instagram button on the left
        bottom_layout = QHBoxLayout()
        bottom_layout.addWidget(instagram_btn)
        bottom_layout.addStretch()  # Push button to the left

        # Main layout with bottom section
        main_content = QHBoxLayout()
        main_content.addWidget(left_w, 0)
        main_content.addWidget(right_w, 1)
        
        main_layout = QVBoxLayout()
        main_layout.addLayout(main_content, 1)
        main_layout.addLayout(bottom_layout)
        main_layout.setContentsMargins(5, 5, 5, 5)
        
        root = QWidget()
        root.setLayout(main_layout)
        self.setCentralWidget(root)

        self.statusBar().showMessage("ready")
        # Report icon load success/failure
        if os.path.isfile(ICON_PATH):
            if _load_app_icon() is not None:
                self.statusBar().showMessage(f"icon loaded: {ICON_PATH}")
            else:
                self.statusBar().showMessage(f"icon found but failed to load (null): {ICON_PATH}")
        else:
            self.statusBar().showMessage(f"icon not found at: {ICON_PATH}")

        # Show GPU acceleration status
        gpu_status = []
        if HAS_CUPY:
            gpu_status.append("GPU")
        if HAS_OPENCV:
            gpu_status.append("OpenCV")
        if HAS_NUMBA:
            gpu_status.append("Numba")

        if gpu_status:
            accel_msg = f"acceleration: {', '.join(gpu_status)}"
            QTimer.singleShot(2000, lambda: self.statusBar().showMessage(accel_msg))

        self.setStyleSheet(QSS)
        self.prev_lbl.set_zoom(1.0)
        self._refresh_gray_ui()
        self._update_titles()
        self._update_moire_tile()
        
        # Keep initial zoom at 100% - no auto-fit
        # QTimer.singleShot(100, self.fit_view_to_window)  # Removed to keep zoom at 100%

    # ----- helpers / titles / artboard -----
    def _update_titles(self):
        w_in, h_in = self._get_doc_size_in(self.orientation.currentText())
        self.setWindowTitle(f"drc_halftone_cmyk — {int(w_in)}×{int(h_in)} @240 dpi • cmyk plates + grayscale K")
        self.art_group.setTitle(f"artboard ({int(w_in)}×{int(h_in)} @ {DOC_DPI} dpi) • place / scale • orientation")

    def _on_artboard_changed(self, *_):
        self._art_cache.clear(); self._cmyk_cache.clear(); self._arr_cache.clear()
        self._mark_halftone_dirty(); self._update_titles()
        self._on_control_changed()

    def _clear_sprite_caches(self):
        self._rotated_sprite_cache.clear(); self._mask_cache.clear()

    def _refresh_gray_ui(self):
        gray = self.gray_chk.isChecked()
        for w in (self.ang_c, self.ang_m, self.ang_y, self.preset_combo, self.full_comp, self.preview_chan):
            w.setEnabled(not gray)

    def _shape_id(self) -> str:
        m = self.mode.currentText()
        if m == "dot": return "circle"
        if m == "custom" and self.shape_paths is not None: return "custom"
        return m

    def _make_base_sprite(self, shape: str, size:int=96):
        key = f"{shape}-{size}"
        if key in self._sprite_cache: return self._sprite_cache[key]
        if not Image or not ImageDraw:
            return None
        try:
            img = Image.new("L", (size, size), 0); d = ImageDraw.Draw(img)
            if shape in ("circle","dot"): d.ellipse((0,0,size-1,size-1), fill=255)
            elif shape=="square": d.rectangle((0,0,size-1,size-1), fill=255)
            elif shape=="triangle": d.polygon([(size/2,0),(size-1,size-1),(0,size-1)], fill=255)
            elif shape=="cross":
                t = max(1, int(size*0.28)); c = size//2
                d.rectangle((c-t//2,0,c+t//2,size-1), fill=255)
                d.rectangle((0,c-t//2,size-1,c+t//2), fill=255)
            elif shape=="diamond":
                d.polygon([(size/2,0),(size-1,size/2),(size/2,size-1),(0,size/2)], fill=255)
            elif shape=="circle outline":
                pass  # ring built per-size using stroke as thickness
            elif shape=="custom" and self.shape_paths is not None and self.shape_bbox is not None:
                img = self._rasterize_svg_paths(self.shape_paths, self.shape_bbox, out_size=size)
            else:
                d.ellipse((0,0,size-1,size-1), fill=255)
            self._sprite_cache[key] = img; return img
        except:
            return None

    def _get_rotated_mask(self, shape: str, angle_deg: int, d: int, p):
        if not HAS_PIL or not Image:
            # Return a dummy mask if PIL is not available
            return None
            
        rot_key = (shape, 96, angle_deg if self.rotate_shape else -999)
        base_rot = self._rotated_sprite_cache.get(rot_key)
        if base_rot is None:
            if shape != "circle outline":
                base = self._make_base_sprite(shape, 96)
                if base and hasattr(base, 'rotate') and self.rotate_shape:
                    try:
                        if hasattr(Image, 'Resampling'):
                            base_rot = base.rotate(angle_deg, resample=Image.Resampling.BICUBIC, expand=True)
                        else:
                            # Use the integer constant directly for older PIL versions
                            base_rot = base.rotate(angle_deg, expand=True)
                    except:
                        base_rot = base
                else:
                    base_rot = base
            else:
                if Image:
                    base_rot = Image.new("L", (96,96), 0)
                else:
                    base_rot = None
            self._rotated_sprite_cache[rot_key] = base_rot

        ring_t = 0
        if shape == "circle outline" and p is not None:
            ring_t = max(1, int(round(p.stroke)))

        mask_key = (shape, angle_deg if self.rotate_shape else -999, d, ring_t)
        mask = self._mask_cache.get(mask_key)
        if mask is None:
            if shape == "circle outline":
                if not Image or not ImageDraw:
                    mask = None
                else:
                    try:
                        sz = max(1, d)
                        m = Image.new("L", (sz, sz), 0)
                        dr = ImageDraw.Draw(m)
                        dr.ellipse((0,0,sz-1,sz-1), fill=255)
                        inner = max(0, sz - 2*ring_t)
                        if inner > 0:
                            pad = (sz - inner)//2
                            dr.ellipse((pad, pad, pad+inner-1, pad+inner-1), fill=0)
                        mask = m
                    except:
                        mask = None
            else:
                if base_rot and hasattr(base_rot, 'resize'):
                    try:
                        if hasattr(Image, 'Resampling'):
                            mask = base_rot.resize((d, d), Image.Resampling.LANCZOS)
                        else:
                            mask = base_rot.resize((d, d), 1)  # 1 = LANCZOS
                    except:
                        mask = None
                else:
                    mask = None
            self._mask_cache[mask_key] = mask
        return mask

    def _rasterize_svg_paths(self, paths, bbox, out_size:int=96):
        if not Image or not ImageDraw:
            return None
        minx, miny, maxx, maxy = bbox
        w = max(1.0, maxx - minx); h = max(1.0, maxy - miny)
        scale = min(out_size / w, out_size / h); ox = -minx; oy = -miny
        try:
            img = Image.new("L", (out_size, out_size), 0); drw = ImageDraw.Draw(img)
            for path in paths:
                N = 600; pts = [path.point(i/(N-1)) for i in range(N)]
                xy = [((float(p.real)+ox)*scale, (float(p.imag)+oy)*scale) for p in pts]
                if len(xy) >= 3: drw.polygon(xy, fill=255)
            return img
        except:
            return None

    def _resize_array(self, arr, scale: float):
        if scale == 1.0 or not np or not Image: 
            return arr
        try:
            h, w = arr.shape; nh, nw = max(1,int(h*scale)), max(1,int(w*scale))
            if hasattr(Image, 'Resampling'):
                img = Image.fromarray((np.clip(arr,0,1)*255).astype(np.uint8), mode="L").resize((nw,nh), Image.Resampling.BOX)
            else:
                img = Image.fromarray((np.clip(arr,0,1)*255).astype(np.uint8), mode="L").resize((nw,nh), 3)  # 3 = BOX
            return (np.asarray(img, dtype=np.float32) / 255.0)
        except:
            return arr

    def _get_doc_size_in(self, orientation: Optional[str]=None) -> Tuple[float,float]:
        if orientation is None: orientation = self.orientation.currentText()
        w_in, h_in = self.ART_SIZES.get(self.art_size.currentText(), (11.0, 15.0))
        return (w_in, h_in) if orientation=="portrait" else (h_in, w_in)

    def _get_transform(self) -> Transform:
        return Transform(int(self.scale_s.value()), int(self._offset_x), int(self._offset_y), self.orientation.currentText())

    def _artboard_size_px_from_transform(self, tr: Transform) -> Tuple[int,int]:
        w_in, h_in = self._get_doc_size_in(tr.orientation)
        return (int(round(w_in*DOC_DPI)), int(round(h_in*DOC_DPI)))

    def _get_mirrored_rgba(self):
        """Get the RGBA image with mirror transformation applied if enabled."""
        if self.img_full_rgba is None:
            return None
        
        src_rgba = self.img_full_rgba
        if self.mirror_chk.isChecked() and Image and hasattr(Image, 'Transpose'):
            if self.mirror_direction == "horizontal":
                src_rgba = src_rgba.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            elif self.mirror_direction == "vertical":
                src_rgba = src_rgba.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        
        return src_rgba

    def _compose_on_artboard(self, src_rgba, tr):
        if not Image:
            return None
        W,H = self._artboard_size_px_from_transform(tr)
        canvas = Image.new("RGBA", (W,H), (255,255,255,255))
        if src_rgba is None: 
            return canvas
        
        s = max(0.01, tr.scale_pct / 100.0)
        sw, sh = src_rgba.size
        tw, th = max(1, int(sw*s)), max(1, int(sh*s))
        try:
            img = src_rgba.resize((tw, th), Image.Resampling.LANCZOS)
            ox = (W - img.width)//2 + tr.offset_x_px
            oy = (H - img.height)//2 + tr.offset_y_px
            canvas.alpha_composite(img, dest=(ox, oy))
        except Exception as e:
            print(f"Error in _compose_on_artboard: {e}")
        return canvas

    def _compose_on_artboard_scaled_preview(self, tr, scale: float):
        if not Image:
            return None
        # Always return a full-size artboard so the preview never looks like a thumbnail.
        W, H = self._artboard_size_px_from_transform(tr)
        canvas = Image.new("RGBA", (W, H), (255, 255, 255, 255))
        if self.img_full_rgba is None:
            return canvas

        # Source with mirror applied (if any)
        src_rgba = self._get_mirrored_rgba()
        if src_rgba is None:
            return canvas

        # Compute full-resolution placement geometry
        img_scale = max(0.01, tr.scale_pct / 100.0)
        sw, sh = src_rgba.size
        tw_full, th_full = max(1, int(sw * img_scale)), max(1, int(sh * img_scale))

        # Render content at reduced resolution, then upscale back to full placement size
        try:
            # Ensure scale is within safe bounds
            scale = max(0.02, min(scale, 1.0))  # Prevent extreme values
            
            tw_small, th_small = max(1, int(tw_full * scale)), max(1, int(th_full * scale))
            # Ultra-fast resampling for extreme scales - prioritize speed over quality
            if scale < 0.08:
                # For very low scales (interaction), use fastest possible method
                resample_method = Image.Resampling.NEAREST
            elif scale < 0.2:
                # For low scales, still prioritize speed
                resample_method = Image.Resampling.NEAREST  # Changed from BILINEAR for speed
            else:
                # For reasonable scales, use good quality
                resample_method = Image.Resampling.BILINEAR  # Changed from LANCZOS for speed
                
            down = src_rgba.resize((tw_small, th_small), resample_method)
            # Always use NEAREST for upscaling during interaction for maximum speed
            img = down.resize((tw_full, th_full), Image.Resampling.NEAREST)

            # Offsets are NOT scaled so the artboard frame stays fixed
            ox = (W - tw_full) // 2 + tr.offset_x_px
            oy = (H - th_full) // 2 + tr.offset_y_px

            # Clamp composition inside the canvas
            ox = max(-img.width + 1, min(W, ox))
            oy = max(-img.height + 1, min(H, oy))

            canvas.alpha_composite(img, dest=(ox, oy))
        except Exception as e:
            print(f"Error in scaled preview: {e}")

        return canvas

    def _art_key(self, tr: Transform) -> Tuple:
        return (id(self.img_full_rgba), self.art_size.currentText(), tr.orientation, tr.scale_pct, tr.offset_x_px, tr.offset_y_px, 
                self.mirror_chk.isChecked(), self.mirror_direction if self.mirror_chk.isChecked() else "")

    def _artboard_rgba_cached(self, tr):
        if self.img_full_rgba is None:
            W,H = self._artboard_size_px_from_transform(tr)
            if not Image:
                return None
            return Image.new("RGBA", (W,H), (255,255,255,255))
        key = self._art_key(tr)
        art = self._art_cache.get(key)
        if art is None:
            art = self._compose_on_artboard(self._get_mirrored_rgba(), tr)
            self._art_cache[key] = art
        return art

    def _artboard_rgba_cached_with_invert(self, tr, invert_enabled=False):
        """Get cached artboard with invert awareness"""
        if self.img_full_rgba is None:
            W,H = self._artboard_size_px_from_transform(tr)
            if not Image:
                return None
            return Image.new("RGBA", (W,H), (255,255,255,255))
        
        # Create a key that includes invert state
        base_key = self._art_key(tr)
        key = f"{base_key}_inv_{invert_enabled}"
        
        art = self._art_cache.get(key)
        if art is None:
            art = self._final_artboard_rgba_with_invert(tr, invert_enabled)
            self._art_cache[key] = art
        return art

    def _base_cmyk_cached(self, art_key, art_rgba):
        im = self._cmyk_cache.get(art_key)
        if im is None:
            im = rgba_to_cmyk_with_icc(art_rgba, self.cmyk_icc_path)
            self._cmyk_cache[art_key] = im
        return im

    def _gamma_from_contrast(self, pct: float) -> float:
        pct = max(1.0, float(pct))
        return 100.0 / pct

    def _calculate_radial_scale(self, dot_pos: Tuple[float, float], center_pos: Tuple[float, float], 
                               radius: float, max_scale: float, falloff_curve: str) -> float:
        """Calculate scale factor based on distance from center point"""
        distance = math.sqrt((dot_pos[0] - center_pos[0])**2 + (dot_pos[1] - center_pos[1])**2)
        
        if distance > radius:
            return 1.0  # No scaling outside radius
        
        # Normalize distance (0 at center, 1 at edge)
        normalized_distance = distance / radius
        
        # Apply falloff curve
        if falloff_curve == "linear":
            scale_factor = 1.0 - normalized_distance
        elif falloff_curve == "smooth":
            scale_factor = 1.0 - (normalized_distance ** 2)
        elif falloff_curve == "sharp":
            scale_factor = 1.0 - (normalized_distance ** 0.5)
        else:
            scale_factor = 1.0 - normalized_distance  # Default to linear
        
        # Apply scaling: 1.0 = no change, max_scale = maximum scaling
        result = 1.0 + (max_scale - 1.0) * scale_factor
        
        # Debug: Print first few scaling calculations
        if hasattr(self, '_debug_count'):
            self._debug_count += 1
        else:
            self._debug_count = 1
        
        if self._debug_count <= 5:
            print(f"Scaling debug: dist={distance:.1f}, radius={radius}, max_scale={max_scale:.2f}, result={result:.2f}")
        
        return result

    def _fast_array_processing(self, arr, operation: str, **kwargs):
        """High-performance array processing with GPU acceleration when available"""
        try:
            if HAS_CUPY and cp and arr.size > 100_000:  # Use GPU for large arrays
                gpu_arr = cp.asarray(arr)
                if operation == "invert":
                    threshold = kwargs.get('threshold', 0.015)
                    mask = gpu_arr > threshold
                    result = cp.where(mask, 1.0 - gpu_arr, gpu_arr)
                elif operation == "gamma":
                    gamma = kwargs.get('gamma', 1.0)
                    result = cp.clip(gpu_arr, 0.0, 1.0) ** gamma
                elif operation == "threshold":
                    threshold = kwargs.get('threshold', 0.5)
                    result = (gpu_arr > threshold).astype(cp.float32)
                elif operation == "threshold_map":
                    threshold_map = kwargs.get('threshold_map')
                    if threshold_map is not None:
                        gpu_threshold_map = cp.asarray(threshold_map)
                        result = (gpu_arr > gpu_threshold_map).astype(cp.float32)
                    else:
                        result = gpu_arr
                elif operation == "multiply":
                    factor = kwargs.get('factor', 1.0)
                    result = gpu_arr * factor
                elif operation == "clip":
                    min_val = kwargs.get('min_val', 0.0)
                    max_val = kwargs.get('max_val', 1.0)
                    result = cp.clip(gpu_arr, min_val, max_val)
                else:
                    result = gpu_arr
                return cp.asnumpy(result)
            else:
                # CPU processing with NumPy optimizations
                if not np:
                    return arr  # Fallback if numpy not available
                if operation == "invert":
                    threshold = kwargs.get('threshold', 0.015)
                    mask = arr > threshold
                    return np.where(mask, 1.0 - arr, arr)
                elif operation == "gamma":
                    gamma = kwargs.get('gamma', 1.0)
                    return np.clip(arr, 0.0, 1.0) ** gamma
                elif operation == "threshold":
                    threshold = kwargs.get('threshold', 0.5)
                    if not np:
                        return arr
                    return (arr > threshold).astype(np.float32)
                elif operation == "threshold_map":
                    threshold_map = kwargs.get('threshold_map')
                    if threshold_map is not None:
                        if not np:
                            return arr
                        return (arr > threshold_map).astype(np.float32)
                    else:
                        return arr
                elif operation == "multiply":
                    factor = kwargs.get('factor', 1.0)
                    return arr * factor
                elif operation == "clip":
                    min_val = kwargs.get('min_val', 0.0)
                    max_val = kwargs.get('max_val', 1.0)
                    if not np:
                        return arr
                    return np.clip(arr, min_val, max_val)
                else:
                    return arr
        except Exception as e:
            # Fallback to CPU processing if GPU fails
            if not np:
                return arr
            if operation == "invert":
                threshold = kwargs.get('threshold', 0.015)
                mask = arr > threshold
                return np.where(mask, 1.0 - arr, arr)
            elif operation == "gamma":
                gamma = kwargs.get('gamma', 1.0)
                return np.clip(arr, 0.0, 1.0) ** gamma
            elif operation == "multiply":
                factor = kwargs.get('factor', 1.0)
                return arr * factor
            elif operation == "clip":
                min_val = kwargs.get('min_val', 0.0)
                max_val = kwargs.get('max_val', 1.0)
                return np.clip(arr, min_val, max_val)
            else:
                return arr

    def _fast_resize(self, img, new_size, method='lanczos'):
        """Fast image resizing with OpenCV backend when available"""
        try:
            cv2_module = sys.modules.get('cv2', None)
            if HAS_OPENCV and np and cv2_module and hasattr(cv2_module, 'cvtColor') and img.size[0] * img.size[1] > 50_000:  # Use OpenCV for larger images
                # Convert PIL to OpenCV format
                cv_img = cv2_module.cvtColor(np.array(img), cv2_module.COLOR_RGBA2BGRA)
                # Use OpenCV's optimized resize
                if method == 'lanczos':
                    cv_resized = cv2_module.resize(cv_img, new_size, interpolation=cv2_module.INTER_LANCZOS4)
                elif method == 'nearest':
                    cv_resized = cv2_module.resize(cv_img, new_size, interpolation=cv2_module.INTER_NEAREST)
                else:
                    cv_resized = cv2_module.resize(cv_img, new_size, interpolation=cv2_module.INTER_CUBIC)
                # Convert back to PIL
                pil_array = cv2_module.cvtColor(cv_resized, cv2_module.COLOR_BGRA2RGBA)
                if not Image:
                    return img
                return Image.fromarray(pil_array, 'RGBA')
            else:
                # Fallback to PIL
                if not Image:
                    return img
                if method == 'lanczos':
                    if hasattr(Image, 'Resampling'):
                        return img.resize(new_size, Image.Resampling.LANCZOS)
                    else:
                        return img.resize(new_size, 1)  # 1 = LANCZOS
                elif method == 'nearest':
                    if hasattr(Image, 'Resampling'):
                        return img.resize(new_size, Image.Resampling.NEAREST)
                    else:
                        return img.resize(new_size, 0)  # 0 = NEAREST
                else:
                    if hasattr(Image, 'Resampling'):
                        return img.resize(new_size, Image.Resampling.LANCZOS)
                    else:
                        return img.resize(new_size, 1)  # 1 = LANCZOS
        except Exception:
            # Fallback to PIL on any error
            if not Image:
                return img
            if method == 'lanczos':
                if hasattr(Image, 'Resampling'):
                    return img.resize(new_size, Image.Resampling.LANCZOS)
                else:
                    return img.resize(new_size, 1)  # 1 = LANCZOS
            elif method == 'nearest':
                if hasattr(Image, 'Resampling'):
                    return img.resize(new_size, Image.Resampling.NEAREST)
                else:
                    return img.resize(new_size, 0)  # 0 = NEAREST
            else:
                if hasattr(Image, 'Resampling'):
                    return img.resize(new_size, Image.Resampling.LANCZOS)
                else:
                    return img.resize(new_size, 1)  # 1 = LANCZOS

    def _arr_gray_cached(self, tr, invert: bool, contrast_pct: float, brightness_pct: float = 100.0):
        key = (self._art_key(tr), 'grayK', bool(invert), int(contrast_pct), int(brightness_pct))
        arr = self._arr_cache.get(key)
        if arr is None:
            W,H = self._artboard_size_px_from_transform(tr)
            if not np:
                return None
            cov_full = np.zeros((H, W), dtype=np.float32)
            if self.img_full_rgba is not None:
                # Apply mirror transformation if checkbox is checked
                src_rgba = self.img_full_rgba
                if self.mirror_chk.isChecked() and Image and hasattr(Image, 'Transpose'):
                    if self.mirror_direction == "horizontal":
                        src_rgba = src_rgba.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                    elif self.mirror_direction == "vertical":
                        src_rgba = src_rgba.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
                
                s = max(0.01, tr.scale_pct / 100.0)
                sw, sh = src_rgba.size
                tw, th = max(1, int(sw*s)), max(1, int(sh*s))
                ox = (W - tw)//2 + tr.offset_x_px
                oy = (H - th)//2 + tr.offset_y_px
                resized = self._fast_resize(src_rgba, (tw, th), 'lanczos')
                L = resized.convert("L"); A = resized.getchannel("A")
                aL = np.asarray(L, dtype=np.float32)/255.0
                aA = np.asarray(A, dtype=np.float32)/255.0
                
                # Normal coverage: dark pixels = high coverage (ink)
                # cov = 1 - lightness means black (0) -> 1.0 coverage, white (1) -> 0 coverage
                cov_small = 1.0 - aL
                
                if invert:
                    # When inverting, we want to flip the coverage BUT
                    # respect white backgrounds and transparency:
                    # - White/near-white pixels (aL >= 0.98) should stay as no coverage (background)
                    # - Transparent pixels (aA <= 0.01) should stay as no coverage
                    # - Only actual content gets inverted
                    is_not_white = aL < 0.98  # Not white/near-white
                    has_alpha = aA > 0.01     # Has meaningful opacity
                    content_mask = is_not_white & has_alpha
                    
                    # Invert coverage only for content pixels
                    # Inverted coverage = lightness (bright content becomes high coverage)
                    cov_small = np.where(content_mask, aL, cov_small)
                
                # Apply brightness adjustment
                brightness_factor = brightness_pct / 100.0
                cov_small = np.clip(cov_small * brightness_factor, 0.0, 1.0)
                
                # Apply contrast adjustment
                gamma = self._gamma_from_contrast(contrast_pct)
                cov_small = self._fast_array_processing(cov_small, "gamma", gamma=gamma)
                cov_small = self._fast_array_processing(cov_small, "multiply", factor=aA)
                l = max(0, ox); t = max(0, oy)
                r = min(W, ox+tw); b = min(H, oy+th)
                if r>l and b>t:
                    ys0, xs0 = t-oy, l-ox
                    cov_full[t:b, l:r] = cov_small[ys0:ys0+(b-t), xs0:xs0+(r-l)]
            self._arr_cache[key] = cov_full; arr = cov_full
        return arr

    def _arr_cmyk_cached(self, art_key, art_rgba, channel: str, contrast_pct: float, brightness_pct: float = 100.0):
        key = (art_key, 'cmyk', channel, int(contrast_pct), int(brightness_pct))
        arr = self._arr_cache.get(key)
        if arr is None:
            base_cmyk = self._base_cmyk_cached(art_key, art_rgba)
            c,m,y,k = base_cmyk.split()
            chan = {"c":c,"m":m,"y":y,"k":k}[channel]
            if not np:
                return None
            a = np.asarray(chan, dtype=np.float32) / 255.0
            
            # Apply brightness adjustment
            brightness_factor = brightness_pct / 100.0
            a = np.clip(a * brightness_factor, 0.0, 1.0)
            
            # Apply contrast adjustment
            gamma = self._gamma_from_contrast(contrast_pct)
            a = self._fast_array_processing(a, "gamma", gamma=gamma)
            self._arr_cache[key] = a; arr = a
        return arr

    def fit_image_to_artboard(self):
        if self.img_full_rgba is None: return
        tr = self._get_transform()
        W, H = self._artboard_size_px_from_transform(tr)
        sw, sh = self.img_full_rgba.size
        if sw==0 or sh==0: return
        s = max(0.01, min(W/sw, H/sh)) * 100.0
        self.scale_s.blockSignals(True); self.scale_s.setValue(int(round(s))); self.scale_s.blockSignals(False)
        self.center_offsets(); self.update_preview(force=True)

    def fit_view_to_window(self):
        if not self._prev_pix: return
        pm_w, pm_h = self._prev_pix.width(), self._prev_pix.height()
        # Use the scroll area viewport size for fitting
        vp = self.preview_scroll.viewport()
        vw, vh = max(1, vp.width()), max(1, vp.height())
        z = int(max(10, min(400, math.floor(min(vw/pm_w, vh/pm_h) * 100))))
        self.zoom.blockSignals(True); self.zoom.setValue(z); self.zoom.blockSignals(False)
        self.prev_lbl.set_zoom(z/100.0)

    def on_drag_delta(self, dx_widget, dy_widget):
        # Convert widget-space drag to artboard pixels for consistent panning
        z = max(0.01, float(self.prev_lbl.zoom_scale))
        # Keep panning independent of the temporary interaction preview scale
        scale_factor = z
        dx_art = int(round(dx_widget / scale_factor))
        dy_art = int(round(dy_widget / scale_factor))
        self._offset_x += dx_art
        self._offset_y += dy_art
        self._mark_halftone_dirty()
        self.preview_timer.start(0)  # Instant updates - no delay at all!

    def center_offsets(self):
        self._offset_x = 0; self._offset_y = 0
        self._mark_halftone_dirty(); self.update_preview(force=True)

    def _update_swatch_style(self, btn, rgba):
        """Update a color swatch button's background to reflect the given RGBA tuple."""
        r, g, b = rgba[0], rgba[1], rgba[2]
        # Choose border color for contrast
        lum = 0.299 * r + 0.587 * g + 0.114 * b
        border = '#aaa' if lum < 80 else '#555'
        btn.setStyleSheet(f"background-color: rgb({r},{g},{b}); border: 2px solid {border}; border-radius: 3px;")

    def _pick_channel_color(self, channel):
        """Open a QColorDialog for the given CMYK channel and update the swatch."""
        current = getattr(self, f'color_{channel}')
        initial = QColor(current[0], current[1], current[2], current[3] if len(current) > 3 else 255)
        color = QColorDialog.getColor(initial, self, f"Choose {channel.upper()} channel color", QColorDialog.ColorDialogOption.ShowAlphaChannel)
        if not color.isValid():
            return
        rgba = (color.red(), color.green(), color.blue(), color.alpha())
        setattr(self, f'color_{channel}', rgba)
        swatch = getattr(self, f'swatch_{channel}')
        self._update_swatch_style(swatch, rgba)
        # Update the CMYK angle dial background to match
        dial = getattr(self, f'ang_{channel}', None)
        if dial:
            r, g, b = rgba[0], rgba[1], rgba[2]
            lum = 0.299 * r + 0.587 * g + 0.114 * b
            border_col = '#fff' if lum < 80 else '#000'
            dial.setStyleSheet(f"background: rgb({r},{g},{b}); border: 2px solid {border_col}; border-radius: 21px;")
        self._mark_halftone_dirty()
        self._on_control_changed()
        self._update_moire_tile()

    def _rgba_to_hex(self, rgba):
        """Convert an RGBA tuple to a hex color string like '#00ffff'."""
        return f"#{rgba[0]:02x}{rgba[1]:02x}{rgba[2]:02x}"

    def _mark_halftone_dirty(self):
        self._halftone_dirty = True

    def params(self) -> Params:
        cell = self.cell_s.value()/100.0
        elem = 1.0  # Fixed at 100%
        stroke = self.str_s.value()/100.0
        contrast = 100.0  # Fixed at 100%
        brightness = 100.0  # Fixed at 100%
        smoothing = 30.0
        regs_on = self.regs_chk.isChecked()
        reg_size_px = float(self.reg_size_s.value())
        reg_offset_px = float(self.reg_offset_s.value())
        dot_gap_pct = 0.0  # Fixed at 0%
        dither_enabled = False  # Dithering disabled
        dither_method = "bayer 4x4"  # Default method
        dither_amount = 50.0  # Default amount
        dither_threshold = 50.0  # Default threshold
        scaling_centers = getattr(self.prev_lbl, 'scaling_centers', [])
        # Get opacity values from dropdowns (parse "100%" -> 100.0)
        opacity_c = float(self.opacity_c.currentText().rstrip('%'))
        opacity_m = float(self.opacity_m.currentText().rstrip('%'))
        opacity_y = float(self.opacity_y.currentText().rstrip('%'))
        opacity_k = float(self.opacity_k.currentText().rstrip('%'))
        return Params(self.mode.currentText(), cell, elem, stroke, contrast, brightness, smoothing,
                      self.preview_chan.currentText(),
                      float(self.ang_c.value()), float(self.ang_m.value()),
                      float(self.ang_y.value()), float(self.ang_k.value()),
                      self.full_comp.isChecked(),
                      regs_on, reg_size_px, reg_offset_px,
                      self.gray_chk.isChecked(), self.invert_chk.isChecked(), self.mirror_chk.isChecked(),
                      dot_gap_pct, dither_enabled, dither_method, dither_amount, dither_threshold,
                      scaling_centers, opacity_c, opacity_m, opacity_y, opacity_k,
                      self.color_c, self.color_m, self.color_y, self.color_k)

    # ---- events ----
    def on_open(self):
        path, _ = QFileDialog.getOpenFileName(self, "open image", "", "Images (*.png *.jpg *.jpeg *.tif *.tiff)")
        if not path: return
        try:
            im = load_image_capped(path)
        except Exception as e:
            QMessageBox.critical(self,"error",str(e)); return
        self.img_full_rgba = im
        self._art_cache.clear(); self._cmyk_cache.clear(); self._arr_cache.clear()
        self._last_render_base_rgba = None
        self._mark_halftone_dirty()
        self.export_png_btn.setEnabled(True); self.export_svg_btn.setEnabled(True)
        self.export_pdf_btn.setEnabled(True); self.export_tiff_btn.setEnabled(True)
        w,h = im.size; self.info.setText(f"{os.path.basename(path)} • {w}×{h}")
        
        # Auto-detect and set orientation based on image aspect ratio
        self._auto_detect_orientation(w, h)
        
        self.update_preview(force=True)
        # Auto-fit artboard to preview window
        self.fit_view_to_window()

    def on_file_menu(self):
        """Show file menu with options for open, save project, load project."""
        menu = QMenu(self)
        
        # Open Image action
        open_action = QAction("Open Image...", self)
        open_action.setToolTip("Load an image file (PNG, JPG, TIFF)")
        open_action.triggered.connect(self.on_open)
        menu.addAction(open_action)
        
        menu.addSeparator()
        
        # Save Project action
        save_action = QAction("Save Project...", self)
        save_action.setToolTip("Save current image and all settings to a project file")
        save_action.triggered.connect(self.on_save_project)
        if self.img_full_rgba is None:
            save_action.setEnabled(False)
        menu.addAction(save_action)
        
        # Load Project action
        load_action = QAction("Load Project...", self)
        load_action.setToolTip("Load a previously saved project file")
        load_action.triggered.connect(self.on_load_project)
        menu.addAction(load_action)
        
        # Show menu at button position
        button = self.sender()
        if button and isinstance(button, QWidget):
            menu.exec(button.mapToGlobal(QPoint(0, button.height())))
        else:
            menu.exec(self.mapToGlobal(QPoint(100, 100)))

    def open_instagram(self):
        """Open the @drc_art Instagram profile in the default web browser."""
        import webbrowser
        try:
            webbrowser.open("https://www.instagram.com/drc_art")
            self.statusBar().showMessage("Opening @drc_art on Instagram...")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Could not open Instagram: {str(e)}")
            self.statusBar().showMessage("Failed to open Instagram")

    def on_save_project(self):
        """Save current project settings and image to a JSON file."""
        try:
            if self.img_full_rgba is None:
                QMessageBox.warning(self, "No Image", "Please load an image first before saving a project.")
                return
            
            path, _ = QFileDialog.getSaveFileName(
                self, 
                "Save Halftone Project", 
                "halftone_project.json", 
                "Halftone Project Files (*.json);;All files (*)"
            )
            
            if not path:
                return
            
            # Convert image to base64 for storage
            import io
            img_buffer = io.BytesIO()
            self.img_full_rgba.save(img_buffer, format='PNG')
            img_data = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
            
            # Collect all current settings
            project_data = {
                'version': '1.0',
                'image_data': img_data,
                'image_size': self.img_full_rgba.size,
                'settings': {
                    'artboard_size': self.art_size.currentText(),
                    'orientation': self.orientation.currentText(),
                    'mode': self.mode.currentText(),
                    'preview_channel': self.preview_chan.currentText(),
                    'cell_size': self.cell_s.value(),
                    'scale': self.scale_s.value(),
                    'contrast': 100,
                    'brightness': 100,
                    'angle_c': self.ang_c.value(),
                    'angle_m': self.ang_m.value(),
                    'angle_y': self.ang_y.value(),
                    'angle_k': self.ang_k.value(),
                    'preset': self.preset_combo.currentText(),
                    'grayscale_mode': self.gray_chk.isChecked(),
                    'invert_gray': self.invert_chk.isChecked(),
                    'dithering_enabled': False,
                    'dithering_algorithm': 'bayer 4x4',
                    'full_composite': self.full_comp.isChecked(),
                    'registration_marks': self.regs_chk.isChecked(),
                    'offset_x': self._offset_x,
                    'offset_y': self._offset_y,
                    # Dot scaling settings
                    'scaling_enabled': self.dot_scale_enabled.isChecked(),
                    'scale_amount': self.scale_amount_s.value(),
                    'scale_radius': self.scale_radius_s.value(),
                    'scale_falloff': self.scale_falloff.currentText(),
                    'scaling_centers': getattr(self.prev_lbl, 'scaling_centers', []),
                    # Custom channel colors
                    'color_c': list(self.color_c),
                    'color_m': list(self.color_m),
                    'color_y': list(self.color_y),
                    'color_k': list(self.color_k),
                }
            }
            
            # Save to file
            with open(path, 'w') as f:
                json.dump(project_data, f, indent=2)
            
            QMessageBox.information(self, "Project Saved", f"Project saved successfully to:\n{path}")
            
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Failed to save project:\n{str(e)}")

    def on_load_project(self):
        """Load project settings and image from a JSON file."""
        try:
            path, _ = QFileDialog.getOpenFileName(
                self, 
                "Load Halftone Project", 
                "", 
                "Halftone Project Files (*.json);;All files (*)"
            )
            
            if not path:
                return
            
            # Load and parse project file
            with open(path, 'r') as f:
                project_data = json.load(f)
            
            # Validate project file
            if not isinstance(project_data, dict) or 'version' not in project_data:
                QMessageBox.warning(self, "Invalid File", "This doesn't appear to be a valid halftone project file.")
                return
            
            # Load image from base64
            if 'image_data' in project_data:
                import io
                img_bytes = base64.b64decode(project_data['image_data'])
                img_buffer = io.BytesIO(img_bytes)
                if HAS_PIL and Image:
                    self.img_full_rgba = Image.open(img_buffer).convert('RGBA')
                else:
                    QMessageBox.warning(self, "PIL Not Available", "PIL/Pillow is required to load project images.")
                    return
                
                # Clear caches and update
                self._art_cache.clear(); self._cmyk_cache.clear(); self._arr_cache.clear()
                self._last_render_base_rgba = None
                self._mark_halftone_dirty()
                
                # Enable export buttons
                self.export_png_btn.setEnabled(True); self.export_svg_btn.setEnabled(True)
                self.export_pdf_btn.setEnabled(True); self.export_tiff_btn.setEnabled(True)
                
                # Update info display
                w, h = self.img_full_rgba.size
                self.info.setText(f"{os.path.basename(path)} • {w}×{h}")
            
            # Restore settings
            if 'settings' in project_data:
                settings = project_data['settings']
                
                # Restore UI controls
                if 'artboard_size' in settings:
                    index = self.art_size.findText(settings['artboard_size'])
                    if index >= 0: self.art_size.setCurrentIndex(index)
                
                if 'orientation' in settings:
                    index = self.orientation.findText(settings['orientation'])
                    if index >= 0: self.orientation.setCurrentIndex(index)
                
                if 'mode' in settings:
                    index = self.mode.findText(settings['mode'])
                    if index >= 0: self.mode.setCurrentIndex(index)
                
                if 'preview_channel' in settings:
                    index = self.preview_chan.findText(settings['preview_channel'])
                    if index >= 0: self.preview_chan.setCurrentIndex(index)
                
                if 'cell_size' in settings:
                    self.cell_s.setValue(settings['cell_size'])
                
                if 'scale' in settings:
                    self.scale_s.setValue(settings['scale'])
                
                # contrast and brightness are now fixed values
                
                if 'angle_c' in settings:
                    self.ang_c.setValue(settings['angle_c'])
                if 'angle_m' in settings:
                    self.ang_m.setValue(settings['angle_m'])
                if 'angle_y' in settings:
                    self.ang_y.setValue(settings['angle_y'])
                if 'angle_k' in settings:
                    self.ang_k.setValue(settings['angle_k'])
                
                if 'preset' in settings:
                    index = self.preset_combo.findText(settings['preset'])
                    if index >= 0: self.preset_combo.setCurrentIndex(index)
                
                if 'grayscale_mode' in settings:
                    self.gray_chk.setChecked(settings['grayscale_mode'])
                
                if 'invert_gray' in settings:
                    self.invert_chk.setChecked(settings['invert_gray'])
                
                # dithering settings are now fixed values
                
                if 'full_composite' in settings:
                    self.full_comp.setChecked(settings['full_composite'])
                
                if 'registration_marks' in settings:
                    self.regs_chk.setChecked(settings['registration_marks'])
                
                if 'offset_x' in settings:
                    self._offset_x = settings['offset_x']
                if 'offset_y' in settings:
                    self._offset_y = settings['offset_y']
                
                # Restore dot scaling settings
                if 'scaling_enabled' in settings:
                    self.dot_scale_enabled.setChecked(settings['scaling_enabled'])
                
                if 'scale_amount' in settings:
                    self.scale_amount_s.setValue(settings['scale_amount'])
                
                if 'scale_radius' in settings:
                    self.scale_radius_s.setValue(settings['scale_radius'])
                
                if 'scale_falloff' in settings:
                    index = self.scale_falloff.findText(settings['scale_falloff'])
                    if index >= 0: self.scale_falloff.setCurrentIndex(index)
                
                if 'scaling_centers' in settings:
                    if hasattr(self.prev_lbl, 'scaling_centers'):
                        self.prev_lbl.scaling_centers = settings['scaling_centers']
                        # Update clear button state
                        self.clear_scaling_btn.setEnabled(len(self.prev_lbl.scaling_centers) > 0)
                
                # Restore custom channel colors
                for ch in ['c', 'm', 'y', 'k']:
                    key = f'color_{ch}'
                    if key in settings:
                        rgba = tuple(settings[key])
                        setattr(self, key, rgba)
                        swatch = getattr(self, f'swatch_{ch}', None)
                        if swatch:
                            self._update_swatch_style(swatch, rgba)
                        dial = getattr(self, f'ang_{ch}', None)
                        if dial:
                            r, g, b = rgba[0], rgba[1], rgba[2]
                            lum = 0.299 * r + 0.587 * g + 0.114 * b
                            border_col = '#fff' if lum < 80 else '#000'
                            dial.setStyleSheet(f"background: rgb({r},{g},{b}); border: 2px solid {border_col}; border-radius: 21px;")
            
            # Update preview and UI
            self.update_preview(force=True)
            self.fit_view_to_window()
            
            QMessageBox.information(self, "Project Loaded", f"Project loaded successfully from:\n{path}")
            
        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Failed to load project:\n{str(e)}")

    def on_load_shape(self):
        if not HAS_SVGPATHTOOLS:
            QMessageBox.warning(self,"unavailable","Install 'svgpathtools' to use custom shapes."); return
        path, _ = QFileDialog.getOpenFileName(self, "load dot shape (SVG)", "", "SVG (*.svg)")
        if not path: return
        try:
            if svg2paths2 is None or SVGPath is None:
                QMessageBox.warning(self, "Error", "svg.path module not available")
                return
            result = svg2paths2(path)
            if len(result) == 3:
                paths, attrs, svg_attrs = result
            else:
                paths, attrs = result
                svg_attrs = {}
            paths = [p for p in paths if isinstance(p, SVGPath) and len(p)>0]
            if not paths: raise ValueError("no vector paths found.")
            minx=miny=1e9; maxx=maxy=-1e9
            for p in paths:
                x0,x1,y0,y1 = p.bbox()
                minx=min(minx,x0); maxx=max(maxx,x1); miny=min(miny,y0); maxy=max(maxy,y1)
            self.shape_paths = paths; self.shape_bbox=(minx,miny,maxx,maxy)
            self._sprite_cache["custom-96"] = self._rasterize_svg_paths(paths, self.shape_bbox, out_size=96)
            if "custom" not in [self.mode.itemText(i) for i in range(self.mode.count())]:
                self.mode.addItem("custom")
            self.mode.setCurrentText("custom")
            self._clear_sprite_caches()
            self.statusBar().showMessage(f"dot shape loaded: {os.path.basename(path)}")
            self._mark_halftone_dirty(); self._on_control_changed(); self._update_moire_tile()
        except Exception as e:
            QMessageBox.critical(self,"load failed",str(e))

    def on_load_reg(self):
        if not HAS_SVGPATHTOOLS:
            QMessageBox.warning(self,"unavailable","Install 'svgpathtools' to load a custom reg mark SVG."); return
        path, _ = QFileDialog.getOpenFileName(self, "load registration mark (SVG)", "", "SVG (*.svg)")
        if not path: return
        try:
            if not svg2paths2 or not SVGPath:
                QMessageBox.warning(self, "Error", "svg.path module not available")
                return
            result = svg2paths2(path)
            if len(result) == 3:
                paths, attrs, svg_attrs = result
            else:
                paths, attrs = result
                svg_attrs = {}
            paths = [p for p in paths if isinstance(p, SVGPath) and len(p)>0]
            if not paths: raise ValueError("no vector paths found.")
            minx=miny=1e9; maxx=maxy=-1e9
            for p in paths:
                x0,x1,y0,y1 = p.bbox()
                minx=min(minx,x0); maxx=max(maxx,x1); miny=min(miny,y0); maxy=max(maxy,y1)
            self.reg_paths = paths; self.reg_bbox=(minx,miny,maxx,maxy)
            self.statusBar().showMessage(f"reg mark loaded: {os.path.basename(path)}")
            self._on_regs_changed()
        except Exception as e:
            QMessageBox.critical(self,"load failed",str(e))

    def _apply_preset(self, name:str):
        if not name: return
        c,m,y,k = self.PRESETS.get(name, (15,75,0,45))
        for d,v in ((self.ang_c,c),(self.ang_m,m),(self.ang_y,y),(self.ang_k,k)):
            d.blockSignals(True); d.setValue(v); d.blockSignals(False)
        self._mark_halftone_dirty(); self._on_control_changed(); self._update_moire_tile()

    def _on_slider_pressed(self): 
        self.interacting = True
        self._interaction_start_time = time.time()
        self._interaction_scale = 0.01  # Extreme 1% scale for maximum speed
        
    def _on_slider_released(self):
        self.interacting = False
        self._interaction_scale = 1.0  # Full resolution when idle
        
        # Immediate full quality update - no delays!
        self.update_preview(force=True)
        self._interaction_start_time = None

    def _on_control_changed(self):
        # Prevent redundant updates
        if hasattr(self, '_updating') and self._updating:
            return
        
        try:
            current_time = time.time()
            if self.interacting:
                # Ultra-fast frame rate limiting (500 FPS max) for instant response
                if hasattr(self, '_last_update_time') and current_time - self._last_update_time < 0.002:
                    return
                self._last_update_time = current_time
                # Immediate updates during interaction
                self.update_preview(force=False)
            else:
                # Immediate updates even when not interacting - no more delays!
                self.update_preview(force=False)
            # Removed random cache cleanup - no background interruptions!
        except Exception as e:
            print(f"Control change error: {e}")
            # Fallback to basic update
            try:
                self.update_preview(force=False)
            except:
                pass  # Prevent cascade crashes

    def _on_regs_changed(self):
        if self._last_render_base_rgba is not None:
            p = self.params(); img = self._last_render_base_rgba.copy()
            if p.regs_on: img = self._paste_regmarks_bitmap(img, p)
            self._prev_pix = pil_to_qpixmap(img); self.prev_lbl.setPixmap(self._prev_pix); self.prev_lbl.update()
        else:
            self.update_preview(force=False)

    def _on_gray_toggled(self, _=None):
        self._refresh_gray_ui(); self._mark_halftone_dirty()
        self.update_preview(force=True); self._update_moire_tile()

    def _on_mirror_toggled(self, _state):
        if self.mirror_chk.isChecked():
            # Show popup dialog to select mirror direction
            from PySide6.QtWidgets import QMessageBox
            msg = QMessageBox(self)
            msg.setWindowTitle("Mirror Direction")
            msg.setText("Select mirror direction:")
            msg.setIcon(QMessageBox.Icon.Question)
            
            horizontal_btn = msg.addButton("Horizontal", QMessageBox.ButtonRole.AcceptRole)
            vertical_btn = msg.addButton("Vertical", QMessageBox.ButtonRole.AcceptRole)
            cancel_btn = msg.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)
            
            msg.exec()
            
            if msg.clickedButton() == horizontal_btn:
                self.mirror_direction = "horizontal"
            elif msg.clickedButton() == vertical_btn:
                self.mirror_direction = "vertical"
            else:
                # Cancel was clicked, uncheck the checkbox
                self.mirror_chk.setChecked(False)
                return
        
        # Clear all caches when mirror state changes
        self._art_cache.clear()
        self._cmyk_cache.clear()
        self._arr_cache.clear()
        self._mark_halftone_dirty()
        self._on_control_changed()

    def _smart_cache_cleanup(self):
        """Ultra-aggressive cache cleanup to eliminate bloat and improve speed"""
        # Much more aggressive cleanup to reduce memory bloat
        if len(self._art_cache) > 5:  # Keep only 3 entries instead of 10
            keys = list(self._art_cache.keys())
            for key in keys[:-3]:
                self._art_cache.pop(key, None)
        
        if len(self._arr_cache) > 10:  # Keep only 5 entries instead of 25
            keys = list(self._arr_cache.keys())
            for key in keys[:-5]:
                self._arr_cache.pop(key, None)
        
        if len(self._grid_cache) > 15:  # Keep only 8 entries instead of 50
            keys = list(self._grid_cache.keys())
            for key in keys[:-8]:
                self._grid_cache.pop(key, None)
        
        if len(self._rotated_sprite_cache) > 5:  # Keep only 3 entries for maximum speed
            keys = list(self._rotated_sprite_cache.keys())
            for key in keys[:-3]:
                self._rotated_sprite_cache.pop(key, None)
                
        # Clear CMYK cache more aggressively as it's memory intensive
        if len(self._cmyk_cache) > 3:
            keys = list(self._cmyk_cache.keys())
            for key in keys[:-2]:  # Keep only 2 most recent
                self._cmyk_cache.pop(key, None)
    # Don't force an immediate update here; let the normal pipeline refresh

    def _auto_detect_orientation(self, width: int, height: int):
        """Auto-detect and set artboard orientation based on image dimensions"""
        try:
            aspect_ratio = width / height
            
            # Determine orientation with some hysteresis to avoid switching on square images
            if aspect_ratio > 1.1:  # Clearly landscape
                target_orientation = "landscape"
            elif aspect_ratio < 0.9:  # Clearly portrait
                target_orientation = "portrait"
            else:
                # Square-ish image, keep current orientation
                return
            
            # Only change if different from current
            current_orientation = self.orientation.currentText()
            if current_orientation != target_orientation:
                # Block signals temporarily to avoid triggering updates
                self.orientation.blockSignals(True)
                self.orientation.setCurrentText(target_orientation)
                self.orientation.blockSignals(False)
                
                # Trigger artboard update
                self._on_artboard_changed()
                
                # Show brief notification
                self.info.setText(f"{self.info.text()} • Auto-set to {target_orientation}")
        except Exception:
            # If auto-detection fails, just continue silently
            pass

    # ---------- grid / geometry ----------
    def _fast_grid_generation(self, w: int, h: int, cell: float, angle_deg: float):
        """Fast grid generation using Numba when available"""
        try:
            if HAS_NUMBA:
                return self._numba_grid_generation(w, h, cell, angle_deg)
            else:
                return self._standard_grid_generation(w, h, cell, angle_deg)
        except Exception:
            return self._standard_grid_generation(w, h, cell, angle_deg)
    
    def _standard_grid_generation(self, w: int, h: int, cell: float, angle_deg: float):
        """Standard grid generation fallback"""
        th = math.radians(angle_deg)
        c, s = abs(math.cos(th)), abs(math.sin(th))
        eff_w = w*c + h*s; eff_h = w*s + h*c
        nx = max(1, int(math.ceil(eff_w/max(1e-6,cell))))
        ny = max(1, int(math.ceil(eff_h/max(1e-6,cell))))
        cx, cy = w/2, h/2
        cos_t, sin_t = math.cos(th), math.sin(th)
        pts = []
        for j in range(ny):
            v = (-eff_h/2) + (j+0.5)*cell
            for i in range(nx):
                u = (-eff_w/2) + (i+0.5)*cell
                x = cx + u*cos_t - v*sin_t
                y = cy + u*sin_t + v*cos_t
                if 0<=x<w and 0<=y<h:
                    pts.append((x,y,cos_t,sin_t))
        if np:
            return np.array(pts, dtype=np.float32)
        else:
            return pts
    
    def _numba_grid_generation(self, w: int, h: int, cell: float, angle_deg: float):
        """Numba-accelerated grid generation (if available)"""
        # This would be decorated with @numba.jit but we'll fallback to standard for now
        return self._standard_grid_generation(w, h, cell, angle_deg)

    def _grid_iter(self, w: int, h: int, cell: float, angle_deg: float):
        key = (w, h, round(float(cell), 2), round(float(angle_deg), 2))
        arr = self._grid_cache.get(key)
        if arr is None:
            th = math.radians(angle_deg)
            c, s = abs(math.cos(th)), abs(math.sin(th))
            eff_w = w*c + h*s; eff_h = w*s + h*c
            nx = max(1, int(math.ceil(eff_w/max(1e-6, cell))))
            ny = max(1, int(math.ceil(eff_h/max(1e-6, cell))))
            cx, cy = w/2, h/2; cos_t, sin_t = math.cos(th), math.sin(th)
            pts = []
            for j in range(ny):
                v = (-eff_h/2) + (j+0.5)*cell
                for i in range(nx):
                    u = (-eff_w/2) + (i+0.5)*cell
                    x = cx + u*cos_t - v*sin_t; y = cy + u*sin_t + v*cos_t
                    if 0 <= x < w and 0 <= y < h: pts.append((x, y, j))
            if np:
                arr = np.array(pts, dtype=np.float32) if pts else np.empty((0, 3), dtype=np.float32)
            else:
                arr = pts if pts else []
            self._grid_cache[key] = arr
        return arr

    def _maybe_bump_cell_for_preview(self, w, h, cell, angle_deg):
        try:
            # Protect against invalid inputs
            if not all(isinstance(x, (int, float)) and x > 0 for x in [w, h, cell]):
                return max(cell, 1.0)  # Safe fallback
            
            th = math.radians(angle_deg); c, s = abs(math.cos(th)), abs(math.sin(th))
            eff_w = w*c + h*s; eff_h = w*s + h*c
            nx = max(1, int(math.ceil(eff_w/max(1e-6,cell))))
            ny = max(1, int(math.ceil(eff_h/max(1e-6,cell))))
            total = nx*ny
            
            # FIXED: Scale budget based on cell size to maintain proper resolution
            # Smaller cells should get larger budgets to maintain dot density
            base_cell = 16.0  # Reference cell size (16px)
            cell_scale_factor = max(1.0, base_cell / max(cell, 1.0))  # More budget for smaller cells
            
            # Use appropriate budget based on interaction state
            if hasattr(self, 'interacting') and self.interacting:
                budget = int(GRID_BUDGET_INTERACTIVE * cell_scale_factor)  # Scale budget with cell size
            else:
                budget = int(GRID_BUDGET_PREVIEW * cell_scale_factor)
                
            if total > budget and budget > 0:
                # Limit the bump factor to prevent excessive cell size increases
                f = min(2.0, math.sqrt(total / budget))  # Cap at 2x to preserve resolution
                return max(cell * f, cell)  # Never make cell smaller
            return max(cell, 0.1)  # Minimum safe cell size
            
        except Exception as e:
            print(f"Grid budget calculation error: {e}")
            return max(cell, 1.0)  # Safe fallback

    def _rdp(self, pts: List[tuple], eps: float) -> List[tuple]:
        if len(pts) < 3 or eps <= 0: return pts
        def _perp_dist(p,a,b):
            (x,y),(x1,y1),(x2,y2)=p,a,b; dx,dy=x2-x1,y2-y1
            if dx==dy==0: return math.hypot(x-x1,y-y1)
            t=((x-x1)*dx+(y-y1)*dy)/(dx*dx+dy*dy); t=max(0.0,min(1.0,t))
            px,py=x1+t*dx,y1+t*dy; return math.hypot(x-px,y-py)
        def _simp(P,eps):
            if len(P)<3: return P
            a,b=P[0],P[-1]; idx,dmax=0,0.0
            for i in range(1,len(P)-1):
                d=_perp_dist(P[i],a,b)
                if d>dmax: idx,dmax=i,d
            if dmax>eps:
                L=_simp(P[:idx+1],eps); R=_simp(P[idx:],eps); return L[:-1]+R
            else: return [a,b]
        return _simp(pts,eps)

    # ---------- renderers ----------
    def _render_plate_from_arr(self, arr, p: Params, angle: float, shape_override: Optional[str]=None):
        # Apply dithering to the array if enabled
        arr = self._maybe_dither_arr(arr, p)
        
        h, w = arr.shape
        base_cell = p.cell
        spacing_cell = base_cell * (1.0 + max(0.0, p.dot_gap_pct)/100.0)
        cell = self._maybe_bump_cell_for_preview(w, h, spacing_cell, angle)
        if not Image or not ImageDraw:
            return None
        out = Image.new("RGBA", (w,h), (255,255,255,0))
        dr = ImageDraw.Draw(out,'RGBA')
        black=(0,0,0,255)
        
        # Dot size multiplier: sqrt(2) allows full coverage when sizef=1.0
        # This ensures dots can overlap at grid corners for solid fill
        dot_size_mult = 1.414  # sqrt(2) for proper halftone coverage

        # Pre-calculate trig values for lines mode
        th = math.radians(angle)
        ct, st = math.cos(th), math.sin(th)

        # Adaptive quality based on interaction state for maximum detail
        if ADAPTIVE_QUALITY:
            tiny_dot_thresh = 0.08 if self.interacting else 0.03  # Better detail preservation
            tiny_line_thresh = 0.12 if self.interacting else 0.06  # Higher detail for lines
            size_quant = SIZE_QUANT_INTERACT if self.interacting else SIZE_QUANT_IDLE
        else:
            tiny_dot_thresh = 0.03
            tiny_line_thresh = 0.06
            size_quant = SIZE_QUANT_IDLE

        mode = shape_override or self._shape_id()

        # Smart parallelization based on grid complexity and system capability
        # _grid_iter now returns a numpy array (N, 3) -> x, y, row_idx
        grid_points = self._grid_iter(w, h, cell, angle)
        grid_size = len(grid_points)
        cpu_cores = cpu_count() or 4

        # Dynamic threshold based on available processing power
        parallel_threshold = max(1500, 4000 // cpu_cores)  # Scale with CPU capability
        should_parallelize = USE_PARALLEL_PROCESSING and grid_size > parallel_threshold

        if mode != "lines":
            if should_parallelize:
                # High-performance parallel processing with optimized batching
                def process_dot_batch(batch):
                    results = []
                    for x, y, _ in batch:
                        sizef = float(arr[int(y), int(x)])
                        
                        # Apply radial scaling if enabled
                        if p.scaling_centers:
                            for center_x, center_y, radius, max_scale, falloff in p.scaling_centers:
                                scale_factor = self._calculate_radial_scale(
                                    (x, y), (center_x, center_y), radius, max_scale / 100.0, falloff
                                )
                                sizef *= scale_factor
                                # Allow scaling beyond 1.0 for dramatic effects
                        
                        r = 0.5*cell*dot_size_mult*p.elem*sizef
                        if r > tiny_dot_thresh:
                            d = int(max(1, round(2*r)))
                            if size_quant > 1:
                                d = max(1, (d//size_quant)*size_quant)
                            results.append((x, y, r, d))
                    return results
                
                # Optimized batch size for better CPU utilization
                optimal_batch_size = max(150, min(500, grid_size // cpu_cores))
                batches = [grid_points[i:i+optimal_batch_size] for i in range(0, len(grid_points), optimal_batch_size)]
                
                # Process batches with optimal worker count
                max_workers = min(cpu_cores, len(batches))
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    batch_results = list(executor.map(process_dot_batch, batches))
                
                # High-quality rendering with optimized mask caching
                for batch_result in batch_results:
                    for x, y, r, d in batch_result:
                        if d <= 2:
                            out.putpixel((int(x),int(y)), black)
                        else:
                            mask = self._get_rotated_mask(mode, int(round(angle)), d, p)
                            out.paste(black, (int(x-r), int(y-r), int(x-r)+d, int(y-r)+d), mask)
            else:
                # Sequential processing with maximum quality rendering
                for x, y, _ in grid_points:
                    sizef = float(arr[int(y), int(x)])
                    
                    # Apply radial scaling if enabled
                    if p.scaling_centers:
                        for center_x, center_y, radius, max_scale, falloff in p.scaling_centers:
                            scale_factor = self._calculate_radial_scale(
                                (x, y), (center_x, center_y), radius, max_scale / 100.0, falloff
                            )
                            sizef *= scale_factor
                            # Allow scaling beyond 1.0 for dramatic effects
                    
                    r = 0.5*cell*dot_size_mult*p.elem*sizef
                    if r>tiny_dot_thresh:
                        d = int(max(1, round(2*r)))
                        if size_quant > 1:
                            d = max(1, (d//size_quant)*size_quant)
                        if d <= 2: 
                            out.putpixel((int(x),int(y)), black)
                        else:
                            mask = self._get_rotated_mask(mode, int(round(angle)), d, p)
                            out.paste(black, (int(x-r), int(y-r), int(x-r)+d, int(y-r)+d), mask)
        else:
            # Enhanced lines mode with adaptive stroke rendering
            for x, y, _ in grid_points:
                sizef = float(arr[int(y), int(x)])
                
                # Apply radial scaling if enabled
                if p.scaling_centers:
                    for center_x, center_y, radius, max_scale, falloff in p.scaling_centers:
                        scale_factor = self._calculate_radial_scale(
                            (x, y), (center_x, center_y), radius, max_scale / 100.0, falloff
                        )
                        sizef *= scale_factor
                        # Allow scaling beyond 1.0 for dramatic effects
                
                L = cell*dot_size_mult*p.elem*sizef
                if L>tiny_line_thresh:
                    dx=(L/2)*ct; dy=(L/2)*st
                    # High-quality stroke rendering
                    sw = max(1, int(round(p.stroke)))
                    dr.line((x-dx,y-dy,x+dx,y+dy), fill=black, width=sw)
        return out

    def _composite_from_cov(self, covC, covM, covY, covK,
                             colC=None, colM=None, colY=None, colK=None):
        if not np or not Image:
            return None
        # Default to standard CMYK if no custom colors
        if colC is None: colC = (0, 255, 255)
        if colM is None: colM = (255, 0, 255)
        if colY is None: colY = (255, 255, 0)
        if colK is None: colK = (0, 0, 0)
        # Start with white background, subtract each channel's ink contribution
        R = np.ones_like(covC, dtype=np.float64) * 255.0
        G = np.ones_like(covC, dtype=np.float64) * 255.0
        B = np.ones_like(covC, dtype=np.float64) * 255.0
        for cov, col in [(covC, colC), (covM, colM), (covY, colY), (covK, colK)]:
            R -= cov * (255.0 - col[0])
            G -= cov * (255.0 - col[1])
            B -= cov * (255.0 - col[2])
        R = np.clip(R, 0, 255)
        G = np.clip(G, 0, 255)
        B = np.clip(B, 0, 255)
        rgb = np.dstack([R, G, B]).astype(np.uint8)
        return Image.fromarray(rgb, mode="RGB").convert("RGBA")

   

    def _update_moire_tile(self):
        """Reliable and accurate CMYK angles previewer - trusted feature."""
        try:
            p = self.params()
            
            # Clean, reliable size for accurate preview
            base_w, base_h = 120, 90
            
            if not np or not Image:
                return

            if p.grayscale_mode:
                # Simple, reliable grayscale preview
                L = np.ones((base_h, base_w), dtype=np.float32) * 0.5  # 50% coverage
                cov = (1.0 - L) if p.invert_gray else L
                plate = self._render_plate_from_arr(cov, p, p.ang_k)
                if plate:
                    disp = plate.resize((self.moire_lbl.width(), self.moire_lbl.height()), 0)  # NEAREST
                    self.moire_lbl.setPixmap(pil_to_qpixmap(disp))
                return

            # Reliable CMYK preview with clean, predictable coverage
            # Use consistent moderate coverage to clearly show angle differences
            coverage = 0.45  # 45% coverage - optimal for seeing halftone patterns and angles
            
            # Same coverage for all plates to focus on angle differences, not coverage variation
            cov = np.ones((base_h, base_w), dtype=np.float32) * coverage
            
            # Render actual CMYK plates with real angles - this is the reliable core
            plateC = self._render_plate_from_arr(cov, p, p.ang_c)
            plateM = self._render_plate_from_arr(cov, p, p.ang_m) 
            plateY = self._render_plate_from_arr(cov, p, p.ang_y)
            plateK = self._render_plate_from_arr(cov, p, p.ang_k)
            
            if not all([plateC, plateM, plateY, plateK]):
                return
                
            # Clean composite showing reliable CMYK interaction
            try:
                # Check that all plates exist and have getchannel method
                if not all(plate is not None and hasattr(plate, 'getchannel') for plate in [plateC, plateM, plateY, plateK]):
                    return
                    
                # Extract alpha channels safely with null checking
                try:
                    covC_final = np.asarray(plateC.getchannel("A"), dtype=np.float32) / 255.0 if plateC and hasattr(plateC, 'getchannel') else np.zeros((base_h, base_w), dtype=np.float32)
                    covM_final = np.asarray(plateM.getchannel("A"), dtype=np.float32) / 255.0 if plateM and hasattr(plateM, 'getchannel') else np.zeros((base_h, base_w), dtype=np.float32)
                    covY_final = np.asarray(plateY.getchannel("A"), dtype=np.float32) / 255.0 if plateY and hasattr(plateY, 'getchannel') else np.zeros((base_h, base_w), dtype=np.float32)
                    covK_final = np.asarray(plateK.getchannel("A"), dtype=np.float32) / 255.0 if plateK and hasattr(plateK, 'getchannel') else np.zeros((base_h, base_w), dtype=np.float32)
                except Exception:
                    return
                
                # Apply opacity to moiré preview
                if p.opacity_c < 100.0:
                    covC_final = covC_final * p.opacity_c / 100.0
                if p.opacity_m < 100.0:
                    covM_final = covM_final * p.opacity_m / 100.0
                if p.opacity_y < 100.0:
                    covY_final = covY_final * p.opacity_y / 100.0
                if p.opacity_k < 100.0:
                    covK_final = covK_final * p.opacity_k / 100.0
                
                p_colors = (p.color_c[:3], p.color_m[:3], p.color_y[:3], p.color_k[:3])
                comp = self._composite_from_cov(covC_final, covM_final, covY_final, covK_final,
                                                *p_colors)
                if comp:
                    disp = comp.resize((self.moire_lbl.width(), self.moire_lbl.height()), 0)  # NEAREST = fastest
                    self.moire_lbl.setPixmap(pil_to_qpixmap(disp))
            except Exception:
                pass
                
        except:
            pass  # Silent fail for stability

    def _paste_regmarks_bitmap(self, img, p):
        if not p.regs_on: return img
        if not ImageDraw:
            return img
        w,h = img.size; size = p.reg_size_px; off = p.reg_offset_px
        draw = ImageDraw.Draw(img, 'RGBA')
        t = max(2, int(round(size*0.06))); half = size*0.5
        for (ax, ay) in [(off,off),(w-off,off),(w-off,h-off),(off,h-off)]:
            draw.rectangle((ax-half, ay-half, ax+half, ay+half), outline=(0,0,0,255), width=t)
            draw.line((ax-half, ay, ax+half, ay), fill=(0,0,0,255), width=t)
            draw.line((ax, ay-half, ax, ay+half), fill=(0,0,0,255), width=t)
        return img

    def _on_dot_scale_toggled(self, state):
        """Handle dot scaling tool enable/disable"""
        if not state:
            # If disabling, clear all scaling areas
            if hasattr(self.prev_lbl, 'scaling_centers'):
                self.prev_lbl.scaling_centers.clear()
                self._mark_halftone_dirty()
                self.update_preview(force=True)
        self.clear_scaling_btn.setEnabled(state and hasattr(self.prev_lbl, 'scaling_centers') and len(self.prev_lbl.scaling_centers) > 0)
        
        # Update cursor when tool state changes
        if hasattr(self.prev_lbl, '_update_cursor'):
            self.prev_lbl._update_cursor()
        
        self._on_control_changed()

    def _clear_scaling_areas(self):
        """Clear all dot scaling areas"""
        if hasattr(self.prev_lbl, 'scaling_centers'):
            self.prev_lbl.scaling_centers.clear()
            self.clear_scaling_btn.setEnabled(False)
            self._mark_halftone_dirty()
            self.update_preview(force=True)

    def _maybe_dither_arr(self, arr, p):
        """Apply dithering to the array if enabled in parameters - crash-proof version."""
        try:
            if not np or not Image or not p.dither_enabled or p.dither_amount == 0:
                return arr
            
            # Validate array
            if arr is None or arr.size == 0:
                return arr
                
            h, w = arr.shape
            if h < 2 or w < 2:  # Too small to dither
                return arr
                
            # Find the actual image content area to avoid dithering empty artboard
            content_mask = arr > 0.01  # Pixels with actual content (not pure white background)
            
            if not np.any(content_mask):
                return arr  # No content to dither
            
            # Find bounding box of content
            content_rows = np.any(content_mask, axis=1)
            content_cols = np.any(content_mask, axis=0)
            
            if not np.any(content_rows) or not np.any(content_cols):
                return arr
                
            # Get the bounding rectangle of the content safely
            top = int(np.argmax(content_rows))
            bottom = h - 1 - int(np.argmax(content_rows[::-1]))
            left = int(np.argmax(content_cols))
            right = w - 1 - int(np.argmax(content_cols[::-1]))
            
            # Add small padding to ensure we capture edge pixels
            padding = 2
            top = max(0, top - padding)
            bottom = min(h - 1, bottom + padding)
            left = max(0, left - padding)
            right = min(w - 1, right + padding)
            
            content_h = bottom - top + 1
            content_w = right - left + 1
            
            # Skip if content area is too small
            if content_w < 2 or content_h < 2:
                return arr
                
            # Extract only the content area for dithering
            content_arr = arr[top:bottom+1, left:right+1].copy()
            
            # Apply optimized dithering to content area only
            max_pixels = 3_000_000
            if content_h * content_w > max_pixels:
                # Use smart downscaling for large content areas
                scale = np.sqrt(max_pixels / (content_h * content_w))
                scale = max(scale, 0.25)  # Never go below 25%
                
                small_h, small_w = max(int(content_h * scale), 32), max(int(content_w * scale), 32)
                
                # Downsample content for dithering
                img_arr = np.clip(content_arr * 255, 0, 255).astype(np.uint8)
                small_img = Image.fromarray(img_arr, mode='L').resize((small_w, small_h), Image.Resampling.LANCZOS)
                small_arr = np.asarray(small_img, dtype=np.float32) / 255.0
                
                # Apply dithering to downsampled content
                dithered_small = self._apply_dithering_fast(small_arr, p)
                
                # Upsample back to original content size
                dithered_img_arr = np.clip(dithered_small * 255, 0, 255).astype(np.uint8)
                dithered_content = Image.fromarray(dithered_img_arr, mode='L').resize((content_w, content_h), Image.Resampling.NEAREST)
                dithered_content_arr = np.asarray(dithered_content, dtype=np.float32) / 255.0
            else:
                # Direct dithering for smaller content areas
                dithered_content_arr = self._apply_dithering_fast(content_arr, p)
            
            # Create result array and paste dithered content back
            result = arr.copy()
            result[top:bottom+1, left:right+1] = dithered_content_arr
            
            return result
            
        except Exception as e:
            print(f"Dithering error: {e}")
            return arr  # Return original array on any error
        
    def _apply_dithering_fast(self, arr, p):
        """Ultra-safe dithering algorithms - crash-proof with size limits."""
        try:
            if not np or arr is None or arr.size == 0:
                return arr
                
            h, w = arr.shape
            if h < 1 or w < 1:
                return arr
                
            # Safely clamp parameters
            threshold = max(0.0, min(1.0, p.dither_threshold / 100.0))
            amount = max(0.0, min(1.0, p.dither_amount / 100.0))
            
            # Boost dithering amount for small images to make effects more visible
            total_pixels = h * w
            if total_pixels < 50000:  # For small preview images
                amount = min(1.0, amount * 2.0)  # Double the amount for much better visibility
            elif total_pixels < 100000:  # For medium images
                amount = min(1.0, amount * 1.5)  # 50% boost for visibility
            
            # Ensure we have a valid method
            method = getattr(p, 'dither_method', 'bayer 4x4')
            if not method:
                method = 'bayer 4x4'
            
            # ULTRA-SAFE SIZE LIMITS - prevent all crashes
            total_pixels = h * w
            
            # Keep all algorithms available - just use internal size limits for safety
            
            # CRASH-PROOF ALGORITHMS FOR PRINTMAKERS
            if method == "stochastic":
                # Fast stochastic with vectorized operations
                np.random.seed(42)
                noise = np.random.random((h, w)) * amount
                result = (arr + noise > threshold).astype(np.float32)
                
            elif method == "error diffusion":
                # ULTRA-SAFE Floyd-Steinberg with strict bounds checking
                result = arr.copy().astype(np.float32)
                weights = np.array([7/16, 3/16, 5/16, 1/16])
                
                # Limit processing size to prevent crashes
                if h * w > 50000:  # Aggressive speed limit
                    # Skip error diffusion for very large images - use simple threshold
                    result = (arr > threshold).astype(np.float32)
                else:
                    # Safe error diffusion for smaller images
                    for y in range(h - 1):  # Stop one row early to prevent bounds issues
                        for x in range(w - 1):  # Stop one column early
                            old_pixel = result[y, x]
                            new_pixel = 1.0 if old_pixel > threshold else 0.0
                            result[y, x] = new_pixel
                            
                            error = (old_pixel - new_pixel) * amount
                            if abs(error) > 0.001:  # Skip negligible errors
                                # Ultra-safe bounds checking
                                if x + 1 < w:
                                    result[y, x + 1] = max(0.0, min(1.0, result[y, x + 1] + error * weights[0]))
                                if y + 1 < h:
                                    if x > 0:
                                        result[y + 1, x - 1] = max(0.0, min(1.0, result[y + 1, x - 1] + error * weights[1]))
                                    result[y + 1, x] = max(0.0, min(1.0, result[y + 1, x] + error * weights[2]))
                                    if x + 1 < w:
                                        result[y + 1, x + 1] = max(0.0, min(1.0, result[y + 1, x + 1] + error * weights[3]))
                                    
            elif method == "blue noise":
                # ULTRA-SAFE blue noise with minimal computation
                try:
                    np.random.seed(42)
                    noise = np.random.random((h, w))
                    
                    # Skip complex filtering for large images
                    if h * w > 25000:
                        # Simple noise for large images
                        result = (arr + (noise - 0.5) * amount > threshold).astype(np.float32)
                    else:
                        # Safe filtering only for smaller images
                        try:
                            from scipy import ndimage
                            # Simple kernel to avoid crashes
                            kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]) / 8.0
                            filtered = ndimage.convolve(noise, kernel, mode='constant', cval=0.5)
                            noise = np.clip(filtered, 0, 1)
                        except:
                            # Ultra-safe manual fallback
                            try:
                                if h >= 3 and w >= 3:
                                    smoothed = np.zeros_like(noise)
                                    for y in range(1, h-1):
                                        for x in range(1, w-1):
                                            # Simple 3x3 average
                                            smoothed[y, x] = np.mean(noise[y-1:y+2, x-1:x+2])
                                    noise = noise - 0.3 * (smoothed - 0.5)
                                    noise = np.clip(noise, 0, 1)
                            except:
                                pass  # Use original noise if any filtering fails
                        
                        result = (arr + (noise - 0.5) * amount > threshold).astype(np.float32)
                        
                except Exception:
                    # Ultimate fallback - simple random
                    np.random.seed(42)
                    noise = np.random.random((h, w)) * amount
                    result = (arr + noise > threshold).astype(np.float32)
                
            elif method == "riemersma":
                # ULTRA-SAFE Riemersma with strict size limits and simple fallback
                result = arr.copy().astype(np.float32)
                
                # Very conservative size limit for Riemersma
                if h * w > 15000:  # Ultra-fast Riemersma limit
                    # Use simple stochastic instead of complex Riemersma for large images
                    np.random.seed(42)
                    noise = np.random.random((h, w)) * amount
                    result = (arr + noise > threshold).astype(np.float32)
                else:
                    try:
                        # Minimal Riemersma implementation for small images only
                        error_buffer_size = min(16, max(4, min(w, h) // 8))  # Even smaller buffer
                        errors = np.zeros(error_buffer_size, dtype=np.float32)
                        error_idx = 0
                        
                        # Simple linear scan instead of serpentine to avoid crashes
                        for y in range(h):
                            for x in range(w):
                                old_pixel = result[y, x]
                                
                                # Minimal error influence
                                error_sum = np.sum(errors) * amount * 0.02  # Very reduced influence
                                adjusted_pixel = max(0.0, min(1.0, old_pixel + error_sum))
                                
                                new_pixel = 1.0 if adjusted_pixel > threshold else 0.0
                                result[y, x] = new_pixel
                                
                                # Simple error update
                                error_val = max(-0.5, min(0.5, old_pixel - new_pixel))  # Limited error range
                                errors[error_idx] = error_val
                                error_idx = (error_idx + 1) % error_buffer_size
                    except Exception:
                        # Absolute fallback to simple threshold
                        result = (arr > threshold).astype(np.float32)
                        
            elif method == "sierra":
                # ULTRA-SAFE Sierra dithering with size limits
                result = arr.copy().astype(np.float32)
                
                if h * w > 25000:  # Speed optimized Sierra
                    result = (arr > threshold).astype(np.float32)
                else:
                    sierra_weights = np.array([
                        [0, 0, 5, 3],
                        [2, 4, 5, 4, 2],
                        [0, 2, 3, 2, 0]
                    ], dtype=np.float32) / 32.0
                    
                    for y in range(h - 2):  # Extra safety margin
                        for x in range(2, w - 2):  # Extra safety margin
                            old_pixel = result[y, x]
                            new_pixel = 1.0 if old_pixel > threshold else 0.0
                            result[y, x] = new_pixel
                            
                            error = (old_pixel - new_pixel) * amount
                            if abs(error) > 0.001:
                                # Ultra-safe error distribution
                                for dy in range(3):
                                    for dx in range(5):
                                        ny, nx = y + dy, x + dx - 2
                                        if 0 <= ny < h and 0 <= nx < w and sierra_weights[dy, dx] > 0:
                                            old_val = result[ny, nx]
                                            new_val = old_val + error * sierra_weights[dy, dx]
                                            result[ny, nx] = max(0.0, min(1.0, new_val))
                                        
            elif method in ["bayer 2x2", "bayer 4x4", "bayer 8x8"]:
                # Fast Bayer dithering with safe matrix handling
                try:
                    size = int(method.split()[-1].split('x')[0])
                except:
                    size = 4  # Safe fallback
                
                if size == 2:
                    bayer = np.array([[0, 2], [3, 1]], dtype=np.float32) / 4.0
                elif size == 8:
                    # Generate 8x8 Bayer matrix safely
                    bayer = np.zeros((8, 8), dtype=np.float32)
                    for i in range(8):
                        for j in range(8):
                            bayer[i, j] = ((i ^ j) + (i << 1)) % 64
                    bayer = bayer / 64.0
                else:  # Default to 4x4
                    bayer = np.array([
                        [0, 8, 2, 10], [12, 4, 14, 6],
                        [3, 11, 1, 9], [15, 7, 13, 5]
                    ], dtype=np.float32) / 16.0
                
                # Safe vectorized Bayer dithering
                try:
                    bayer_tiled = np.tile(bayer, (h // size + 1, w // size + 1))[:h, :w]
                    threshold_map = threshold + (bayer_tiled - 0.5) * amount
                    result = (arr > threshold_map).astype(np.float32)
                except:
                    # Fallback to simple threshold
                    result = (arr > threshold).astype(np.float32)
                    
            elif method == "floyd-steinberg":
                # Enhanced Classic Floyd-Steinberg - more visible effects
                result = arr.copy().astype(np.float32)
                
                # Aggressive size limit for maximum speed
                if h * w > 50000:
                    result = (arr > threshold).astype(np.float32)
                else:
                    try:
                        weights = np.array([7/12, 3/12, 5/12, 1/12])  # Boosted weights for more visible effect
                        
                        # Ensure minimum error threshold for maximum visibility
                        min_error = 0.00001  # Much lower threshold for extreme visibility
                        
                        for y in range(h - 1):  
                            for x in range(w - 1):  
                                old_pixel = result[y, x]
                                new_pixel = 1.0 if old_pixel > threshold else 0.0
                                result[y, x] = new_pixel
                                
                                error = (old_pixel - new_pixel) * amount
                                if abs(error) > min_error:
                                    # Enhanced bounds checking with error distribution
                                    if x + 1 < w:
                                        result[y, x + 1] = max(0.0, min(1.0, result[y, x + 1] + error * weights[0]))
                                    if y + 1 < h and x > 0:
                                        result[y + 1, x - 1] = max(0.0, min(1.0, result[y + 1, x - 1] + error * weights[1]))
                                    if y + 1 < h:
                                        result[y + 1, x] = max(0.0, min(1.0, result[y + 1, x] + error * weights[2]))
                                    if y + 1 < h and x + 1 < w:
                                        result[y + 1, x + 1] = max(0.0, min(1.0, result[y + 1, x + 1] + error * weights[3]))
                    except Exception:
                        # Fallback to simple threshold if anything goes wrong
                        result = (arr > threshold).astype(np.float32)
                                        
            elif method == "ordered":
                # Simple ordered dithering (same as bayer 4x4)
                bayer = np.array([
                    [0, 8, 2, 10], [12, 4, 14, 6],
                    [3, 11, 1, 9], [15, 7, 13, 5]
                ], dtype=np.float32) / 16.0
                
                try:
                    bayer_tiled = np.tile(bayer, (h // 4 + 1, w // 4 + 1))[:h, :w]
                    threshold_map = threshold + (bayer_tiled - 0.5) * amount
                    result = (arr > threshold_map).astype(np.float32)
                except:
                    result = (arr > threshold).astype(np.float32)
                    
            else:
                # Safe fallback for unknown methods
                result = (arr > threshold).astype(np.float32)
            
            return np.clip(result, 0, 1)  # Ensure valid range
            
        except Exception as e:
            print(f"Dithering algorithm error: {e}")
            return arr  # Return original array on any error
        
    def _apply_dithering(self, arr, p):
        """Redirect to optimized fast dithering method."""
        return self._apply_dithering_fast(arr, p)

    # ---------- preview ----------
    def update_preview(self, force=False, only_regs=False):
        # Prevent redundant updates
        if self._updating and not force:
            return
        self._updating = True
        
        try:
            # Skip redundant work when nothing changed
            if not force and not self._halftone_dirty and not self.interacting and not only_regs:
                return
            if self.img_full_rgba is None:
                tr = self._get_transform()
                if Image:
                    empty = Image.new("RGBA", self._artboard_size_px_from_transform(tr), (255,255,255,255))
                    self._prev_pix = pil_to_qpixmap(empty)
                    self.prev_lbl.setPixmap(self._prev_pix)
                    self.prev_lbl.update()
                return

            p = self.params()
            tr = self._get_transform()

            # Ultra-aggressive optimization for interactions
            if self.interacting and not force:
                preview_scale = self._interaction_scale  # e.g., 0.08 during interaction
                # Disable ALL expensive features during interaction for maximum speed
                p = replace(p, 
                           dither_enabled=False,    # No dithering
                           regs_on=False,          # No registration marks
                           smoothing_pct=0,        # No smoothing
                           contrast_pct=100,       # No contrast adjustment
                           brightness_pct=100)     # No brightness adjustment
            else:
                preview_scale = 1.0
        finally:
            self._updating = False

        if only_regs and self._last_render_base_rgba is not None:
            img = self._last_render_base_rgba.copy()
            if p.regs_on:
                img = self._paste_regmarks_bitmap(img, p)
            self._prev_pix = pil_to_qpixmap(img)
            self.prev_lbl.setPixmap(self._prev_pix)
            self.prev_lbl.update()
            return

        # Compose artboard at appropriate resolution
        if preview_scale < 1.0:
            art_rgba = self._compose_on_artboard_scaled_preview(tr, preview_scale)
            if not art_rgba:
                return
            tr_scaled = tr
        else:
            art_rgba = self._artboard_rgba_cached_with_invert(tr, p.invert_gray)
            tr_scaled = tr

        # Cache key (includes invert + interaction state)
        base_art_key = self._art_key(tr_scaled)
        cache_suffix = f"_inv_{p.invert_gray}_int_{self.interacting}_scale_{preview_scale}"
        art_key = f"{base_art_key}{cache_suffix}"

        # Render
        if p.grayscale_mode:
            arrK_full = self._arr_gray_cached(tr, invert=p.invert_gray, contrast_pct=p.contrast_pct, brightness_pct=p.brightness_pct)
            plate = self._render_plate_from_arr(arrK_full, p, p.ang_k)
            if not Image or not plate:
                return
            base_img = Image.new("RGBA", plate.size, (255,255,255,255))
            base_img.paste((0,0,0,255), (0,0,plate.width,plate.height), plate.getchannel("A"))
        else:
            if p.preview_channel == "composite" and self.full_comp.isChecked():
                def _mk(ch):
                    return self._arr_cmyk_cached(art_key, art_rgba, ch, p.contrast_pct, p.brightness_pct)
                arrC, arrM, arrY, arrK = (_mk("c"), _mk("m"), _mk("y"), _mk("k"))
                max_workers = min(4, cpu_count() or 2)
                with ThreadPoolExecutor(max_workers=max_workers) as ex:
                    futs = [
                        ex.submit(self._render_plate_from_arr, arrC, p, p.ang_c),
                        ex.submit(self._render_plate_from_arr, arrM, p, p.ang_m),
                        ex.submit(self._render_plate_from_arr, arrY, p, p.ang_y),
                        ex.submit(self._render_plate_from_arr, arrK, p, p.ang_k),
                    ]
                    plates = [f.result() for f in futs]
                if not np:
                    return
                cov = [np.asarray(im.getchannel("A"), dtype=np.float32)/255.0 for im in plates if im]
                # Apply opacity to preview
                if len(cov) >= 4:
                    if p.opacity_c < 100.0:
                        cov[0] = cov[0] * p.opacity_c / 100.0
                    if p.opacity_m < 100.0:
                        cov[1] = cov[1] * p.opacity_m / 100.0
                    if p.opacity_y < 100.0:
                        cov[2] = cov[2] * p.opacity_y / 100.0
                    if p.opacity_k < 100.0:
                        cov[3] = cov[3] * p.opacity_k / 100.0
                base_img = self._composite_from_cov(cov[0], cov[1], cov[2], cov[3],
                                                     p.color_c[:3], p.color_m[:3], p.color_y[:3], p.color_k[:3])
            else:
                ch = p.preview_channel if p.preview_channel != "composite" else "k"
                arr = self._arr_cmyk_cached(art_key, art_rgba, ch, p.contrast_pct, p.brightness_pct)
                ang = {"c": p.ang_c, "m": p.ang_m, "y": p.ang_y, "k": p.ang_k}[ch]
                plate = self._render_plate_from_arr(arr, p, ang)
                tint = {"c": p.color_c, "m": p.color_m, "y": p.color_y, "k": p.color_k}[ch]
                if not Image or not plate:
                    return
                base_img = Image.new("RGBA", plate.size, (255,255,255,255))
                base_img.paste(tint, (0,0,plate.width,plate.height), plate.getchannel("A"))

        # Save last and overlay regs if needed
        self._last_render_base_rgba = base_img.copy() if base_img else None
        p2 = self.params()  # re-read in case state changed
        if p2.regs_on and base_img:
            base_img = self._paste_regmarks_bitmap(base_img, p2)
        if base_img:
            self._prev_pix = pil_to_qpixmap(base_img)
            self.prev_lbl.setPixmap(self._prev_pix)
            self.prev_lbl.update()
        self._halftone_dirty = False

    # ---------- exports ----------
    def _final_artboard_rgba(self, tr):
        return self._compose_on_artboard(self._get_mirrored_rgba(), tr)

    def _final_artboard_rgba_with_invert(self, tr, invert_enabled=False):
        """Get the final artboard with optional invert applied only to visible image content.
        
        Respects transparency AND white backgrounds: 
        - Transparent pixels (alpha <= 2) are not inverted
        - White/near-white pixels are not inverted (treated as background)
        Only actual content (non-white, non-transparent) gets inverted.
        """
        src_rgba = self._get_mirrored_rgba()
        
        # Apply invert to the source image before compositing if enabled
        if invert_enabled and src_rgba and Image and np:
            # Convert to numpy array with float32 for clean math
            img_array = np.array(src_rgba, dtype=np.float32)
            
            # Get the alpha channel (0-255)
            alpha = img_array[:, :, 3]
            
            # Get RGB channels
            r = img_array[:, :, 0]
            g = img_array[:, :, 1]
            b = img_array[:, :, 2]
            
            # Calculate luminance/brightness of each pixel
            # White pixels have high values in all RGB channels
            luminance = 0.299 * r + 0.587 * g + 0.114 * b
            
            # Content mask: pixel must have:
            # 1. Meaningful alpha (not transparent)
            # 2. Not be white/near-white (luminance < 250)
            # This ensures we only invert the actual visible content, not backgrounds
            has_alpha = alpha > 2
            is_not_white = luminance < 250  # Threshold for "not white"
            content_mask = has_alpha & is_not_white
            
            # Invert only the RGB channels where there's actual content
            for channel in range(3):  # R, G, B channels
                channel_data = img_array[:, :, channel]
                
                # Invert: new_value = 255 - old_value
                inverted = 255.0 - channel_data
                
                # Apply inversion only to content pixels (not transparent, not white)
                img_array[:, :, channel] = np.where(content_mask, inverted, channel_data)
            
            # Convert back to PIL Image, keeping alpha intact
            img_array = np.clip(img_array, 0, 255).astype(np.uint8)
            src_rgba = Image.fromarray(img_array, 'RGBA')
        
        return self._compose_on_artboard(src_rgba, tr)

    def on_export_png(self):
        if self.img_full_rgba is None: return
        p = self.params(); tr = self._get_transform()
        base = self._final_artboard_rgba_with_invert(tr, p.invert_gray)

        if p.grayscale_mode:
            if not base:
                return None
            L = base.convert("L")
            if not np:
                return None
            arrK = np.asarray(L, dtype=np.float32)/255.0
            # Convert lightness to ink coverage: black pixels should have high coverage
            # Since _final_artboard_rgba_with_invert already handled the invert logic,
            # we always need to convert lightness to ink coverage (1.0 - lightness)
            cov = 1.0 - arrK  # Black (0) becomes high coverage (1), White (1) becomes low coverage (0)
            cov = np.clip(cov,0,1) ** self._gamma_from_contrast(p.contrast_pct)
            plate = self._render_plate_from_arr(cov, p, p.ang_k)
            if not Image or not plate:
                return None
            out = Image.new("RGBA", plate.size, (255,255,255,255))
            out.paste((0,0,0,255), (0,0,plate.width,plate.height), plate.getchannel("A"))
            if p.regs_on: out = self._paste_regmarks_bitmap(out, p)
            
            # Save single grayscale file (dithering doesn't affect single channel much)
            path, _ = QFileDialog.getSaveFileName(self,"export grayscale halftone (PNG)","halftone_k.png","PNG (*.png)")
            if not path: return
            if out:
                out.save(path, "PNG"); QMessageBox.information(self,"saved",path)
            else:
                QMessageBox.warning(self,"Error","Failed to generate preview")
            return
            
        else:
            base_cmyk = rgba_to_cmyk_with_icc(base, self.cmyk_icc_path)
            c,m,y,k = base_cmyk.split()
            def cov_of(ch_img):
                if not np:
                    return None
                a = np.asarray(ch_img, dtype=np.float32)/255.0
                # Invert logic now handled at image level before CMYK conversion
                return np.clip(a,0,1) ** self._gamma_from_contrast(p.contrast_pct)
            arrC,arrM,arrY,arrK = (cov_of(c), cov_of(m), cov_of(y), cov_of(k))
            
            # Always use folder approach for CMYK exports (professional workflow)
            folder_path = QFileDialog.getExistingDirectory(self, "Choose folder for CMYK PNG export", "")
            if not folder_path: return
            
            # Create a subfolder for this export
            import time
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            dither_suffix = "_dithered" if p.dither_enabled else ""
            export_folder = os.path.join(folder_path, f"halftone_cmyk{dither_suffix}_png_{timestamp}")
            os.makedirs(export_folder, exist_ok=True)
            
            with ThreadPoolExecutor(max_workers=self._threads) as ex:
                futs = [
                    ex.submit(self._render_plate_from_arr, arrC, p, p.ang_c),
                    ex.submit(self._render_plate_from_arr, arrM, p, p.ang_m),
                    ex.submit(self._render_plate_from_arr, arrY, p, p.ang_y),
                    ex.submit(self._render_plate_from_arr, arrK, p, p.ang_k),
                ]
                plates = [f.result() for f in futs]
            
            channel_names = ["c", "m", "y", "k"]
            channel_colors = [p.color_c, p.color_m, p.color_y, p.color_k]  # User-chosen channel colors
            exported_files = []
            
            for i, (plate, ch_name, color) in enumerate(zip(plates, channel_names, channel_colors)):
                if plate and hasattr(plate, 'getchannel') and Image:
                    out = Image.new("RGBA", plate.size, (255,255,255,255))
                    out.paste(color, (0,0,plate.width,plate.height), plate.getchannel("A"))
                    if p.regs_on: out = self._paste_regmarks_bitmap(out, p)
                    
                    out_path = os.path.join(export_folder, f"halftone_{ch_name}.png")
                    out.save(out_path, "PNG")
                    exported_files.append(f"halftone_{ch_name}.png")
            
            # Create a composite preview as well
            if len(plates) >= 4 and all(plates) and np:
                cov = []
                for plate in plates:
                    if plate and hasattr(plate, 'getchannel'):
                        try:
                            cov.append(np.asarray(plate.getchannel("A"), dtype=np.float32)/255.0)
                        except:
                            cov.append(None)
                    else:
                        cov.append(None)
                
                if len(cov) >= 4 and all(c is not None for c in cov):
                    composite = self._composite_from_cov(cov[0],cov[1],cov[2],cov[3],
                                                         p.color_c[:3], p.color_m[:3], p.color_y[:3], p.color_k[:3])
                    if composite:
                        if p.regs_on: composite = self._paste_regmarks_bitmap(composite, p)
                        comp_path = os.path.join(export_folder, "halftone_composite.png")
                        composite.save(comp_path, "PNG")
                        exported_files.append("halftone_composite.png")
            
            # Create a readme file
            readme_path = os.path.join(export_folder, "README.txt")
            with open(readme_path, 'w') as f:
                f.write("CMYK HALFTONE PNG EXPORT\n")
                f.write("=" * 25 + "\n\n")
                f.write("This folder contains separate halftone plates for each CMYK channel.\n")
                f.write("Each file is ready for professional printing workflow.\n\n")
                f.write("Files:\n")
                f.write("- halftone_c.png (Cyan plate)\n")
                f.write("- halftone_m.png (Magenta plate)\n")
                f.write("- halftone_y.png (Yellow plate)\n")
                f.write("- halftone_k.png (Black plate)\n")
                f.write("- halftone_composite.png (Combined preview)\n\n")
                f.write("Settings used:\n")
                if p.dither_enabled:
                    f.write(f"Dithering: {p.dither_method} at {p.dither_amount}%\n")
                else:
                    f.write("Dithering: Disabled\n")
                f.write(f"Cell size: {p.cell:.2f}px\n")
                f.write(f"Angles: C={p.ang_c}°, M={p.ang_m}°, Y={p.ang_y}°, K={p.ang_k}°\n")
                f.write(f"Element scale: {p.elem*100:.0f}%\n")
            
            QMessageBox.information(self,"CMYK PNG Export Complete", 
                f"CMYK plates exported to folder:\n{os.path.basename(export_folder)}\n\n" +
                f"Files: {', '.join(exported_files)}\n" +
                "README.txt included with export details.")

    def on_export_tiff_cmyk(self):
        if self.img_full_rgba is None: return
        p = self.params(); tr = self._get_transform()
        
        # Warn about dithering
        if p.dither_enabled:
            reply = QMessageBox.question(self, "Dithering Enabled", 
                "Dithering is enabled. TIFF export will create a composite image.\n" +
                "For separate channel files, use SVG or PNG export instead.\n\n" +
                "Continue with TIFF composite export?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if reply == QMessageBox.StandardButton.No:
                return
        
        base = self._final_artboard_rgba_with_invert(tr, p.invert_gray)
        if not base or not Image or not np:
            QMessageBox.warning(self, "Error", "Missing required libraries for TIFF export")
            return

        if p.grayscale_mode:
            L = base.convert("L")
            arrK = np.asarray(L, dtype=np.float32)/255.0
            # Convert lightness to ink coverage: black pixels should have high coverage
            # Since _final_artboard_rgba_with_invert already handled the invert logic,
            # we always need to convert lightness to ink coverage (1.0 - lightness)
            covK = 1.0 - arrK  # Black (0) becomes high coverage (1), White (1) becomes low coverage (0)
            covK = np.clip(covK,0,1) ** self._gamma_from_contrast(p.contrast_pct)
            plateK = self._render_plate_from_arr(covK, p, p.ang_k)
            if not plateK:
                return
            maskK = plateK.getchannel("A")
            zero = Image.new("L", maskK.size, 0)
            C,M,Y,K = zero, zero, zero, maskK
            cmyk_halftone = Image.merge("CMYK", (C,M,Y,K))
            if p.regs_on and ImageChops:
                tmp = Image.new("RGBA", cmyk_halftone.size, (255,255,255,0))
                tmp = self._paste_regmarks_bitmap(tmp, p)
                reg_mask = tmp.getchannel("A")
                C,M,Y,K = cmyk_halftone.split(); K = ImageChops.lighter(K, reg_mask)
                cmyk_halftone = Image.merge("CMYK", (C,M,Y,K))
        else:
            base_cmyk = rgba_to_cmyk_with_icc(base, self.cmyk_icc_path)
            c,m,y,k = base_cmyk.split()
            def cov_of(ch_img):
                if not np:
                    return None
                a = np.asarray(ch_img, dtype=np.float32)/255.0
                # Invert logic now handled at image level before CMYK conversion
                return np.clip(a,0,1) ** self._gamma_from_contrast(p.contrast_pct)
            arrC,arrM,arrY,arrK = (cov_of(c), cov_of(m), cov_of(y), cov_of(k))
            plateC = self._render_plate_from_arr(arrC, p, p.ang_c)
            plateM = self._render_plate_from_arr(arrM, p, p.ang_m)
            plateY = self._render_plate_from_arr(arrY, p, p.ang_y)
            plateK = self._render_plate_from_arr(arrK, p, p.ang_k)
            if not all([plateC, plateM, plateY, plateK]):
                return
            masks = {}
            for ch, plate in zip("cmyk",[plateC,plateM,plateY,plateK]):
                if plate and hasattr(plate, 'getchannel'):
                    try:
                        masks[ch] = plate.getchannel("A")
                    except:
                        masks[ch] = None
                else:
                    masks[ch] = None
            # Apply opacity to each channel mask (for TIFF export)
            if masks["c"] is not None and p.opacity_c < 100.0:
                masks["c"] = Image.fromarray((np.asarray(masks["c"]) * p.opacity_c / 100.0).astype(np.uint8), mode="L")
            if masks["m"] is not None and p.opacity_m < 100.0:
                masks["m"] = Image.fromarray((np.asarray(masks["m"]) * p.opacity_m / 100.0).astype(np.uint8), mode="L")
            if masks["y"] is not None and p.opacity_y < 100.0:
                masks["y"] = Image.fromarray((np.asarray(masks["y"]) * p.opacity_y / 100.0).astype(np.uint8), mode="L")
            if masks["k"] is not None and p.opacity_k < 100.0:
                masks["k"] = Image.fromarray((np.asarray(masks["k"]) * p.opacity_k / 100.0).astype(np.uint8), mode="L")
            if all(masks.values()) and Image:
                cmyk_halftone = Image.merge("CMYK", (masks["c"],masks["m"],masks["y"],masks["k"]))
            else:
                cmyk_halftone = None
            if p.regs_on and ImageChops and cmyk_halftone:
                try:
                    tmp = Image.new("RGBA", cmyk_halftone.size, (255,255,255,0))
                    tmp = self._paste_regmarks_bitmap(tmp, p)
                    reg_mask = tmp.getchannel("A")
                    C,M,Y,K = cmyk_halftone.split(); K = ImageChops.lighter(K, reg_mask)
                    cmyk_halftone = Image.merge("CMYK", (C,M,Y,K))
                except:
                    pass

        path, _ = QFileDialog.getSaveFileName(self, "export composite (CMYK TIFF)", "halftone_cmyk.tif", "TIFF (*.tif *.tiff)")
        if not path: return
        try:
            if cmyk_halftone:
                cmyk_halftone.save(path, format="TIFF", compression="tiff_lzw")
                QMessageBox.information(self,"saved",path)
            else:
                QMessageBox.warning(self,"Error","Failed to generate CMYK halftone")
        except Exception as e:
            QMessageBox.critical(self,"save failed",str(e))

    def on_export_svg(self):
        if self.img_full_rgba is None: return
        p = self.params(); tr = self._get_transform()
        base = self._final_artboard_rgba_with_invert(tr, p.invert_gray)
        if not base:
            return
        W,H = base.size

        if p.grayscale_mode:
            L = base.convert("L")
            if not np:
                return
            arrK = np.asarray(L, dtype=np.float32)/255.0
            # Convert lightness to ink coverage: black pixels should have high coverage
            # Since _final_artboard_rgba_with_invert already handled the invert logic,
            # we always need to convert lightness to ink coverage (1.0 - lightness)
            covK = 1.0 - arrK  # Black (0) becomes high coverage (1), White (1) becomes low coverage (0)
            covK = np.clip(covK,0,1) ** self._gamma_from_contrast(p.contrast_pct)
            base_name, _ = QFileDialog.getSaveFileName(self,"export grayscale plate SVG (choose base name)","halftone","All Files (*.*)")
            if not base_name: return
            root = os.path.splitext(base_name)[0]
            out_path = f"{root}_k.svg"
            if not HAS_SVGWRITE:
                QMessageBox.warning(self,"Error","svgwrite module not available")
                return
            try:
                import svgwrite as svg_module
                dwg = svg_module.Drawing(filename=out_path, size=(f"{W}px", f"{H}px"), viewBox=f"0 0 {W} {H}")
                self._export_svg_channel_into(dwg, covK, p, p.ang_k, fill_color="#000000")
                self._add_regmarks_svg(dwg, p, W, H); dwg.save()
                QMessageBox.information(self,"saved", out_path)
            except ImportError:
                QMessageBox.warning(self,"Error","svgwrite module not available")
            return

        base_cmyk = rgba_to_cmyk_with_icc(base, self.cmyk_icc_path)
        c,m,y,k = base_cmyk.split()
        def cov_of(ch_img):
            if not np:
                return None
            a = np.asarray(ch_img, dtype=np.float32)/255.0
            # Invert logic now handled at image level before CMYK conversion
            return np.clip(a,0,1) ** self._gamma_from_contrast(p.contrast_pct)
        arrs = { "c": cov_of(c), "m": cov_of(m), "y": cov_of(y), "k": cov_of(k) }
        angles = {"c":p.ang_c, "m":p.ang_m, "y":p.ang_y, "k":p.ang_k}

        # Always create a folder with separate CMYK channel files for professional print workflow
        folder_path = QFileDialog.getExistingDirectory(self, "Choose folder for CMYK halftone plates", "")
        if not folder_path: return
        
        # Create a subfolder for this export
        import time
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        dither_suffix = "_dithered" if p.dither_enabled else ""
        export_folder = os.path.join(folder_path, f"halftone_cmyk{dither_suffix}_{timestamp}")
        os.makedirs(export_folder, exist_ok=True)
        
        if not HAS_SVGWRITE:
            QMessageBox.warning(self,"Error","svgwrite module not available")
            return
        
        try:
            import svgwrite as svg_module
            exported_files = []
            for ch in ["c","m","y","k"]:
                out_path = os.path.join(export_folder, f"halftone_{ch}.svg")
                dwg = svg_module.Drawing(filename=out_path, size=(f"{W}px", f"{H}px"), viewBox=f"0 0 {W} {H}")
                self._export_svg_channel_into(dwg, arrs[ch], p, angles[ch], fill_color="#000000")
                self._add_regmarks_svg(dwg, p, W, H); dwg.save()
                exported_files.append(f"halftone_{ch}.svg")
            
            # Create a readme file explaining the export
            readme_path = os.path.join(export_folder, "README.txt")
            with open(readme_path, 'w') as f:
                f.write("CMYK HALFTONE SVG EXPORT\n")
                f.write("=" * 25 + "\n\n")
                f.write("This folder contains separate halftone plates for each CMYK channel.\n")
                f.write("Each file is ready for professional printing workflow.\n\n")
                f.write("Files:\n")
                f.write("- halftone_c.svg (Cyan plate)\n")
                f.write("- halftone_m.svg (Magenta plate)\n")
                f.write("- halftone_y.svg (Yellow plate)\n")
                f.write("- halftone_k.svg (Black plate)\n\n")
                f.write("Settings used:\n")
                if p.dither_enabled:
                    f.write(f"Dithering: {p.dither_method} at {p.dither_amount}%\n")
                else:
                    f.write("Dithering: Disabled\n")
                f.write(f"Cell size: {p.cell:.2f}px\n")
                f.write(f"Angles: C={p.ang_c}°, M={p.ang_m}°, Y={p.ang_y}°, K={p.ang_k}°\n")
                f.write(f"Mode: {p.mode}\n")
                f.write(f"Element scale: {p.elem*100:.0f}%\n")
            
            dither_msg = " (dithered)" if p.dither_enabled else ""
            QMessageBox.information(self,"CMYK Export Complete", 
                f"CMYK halftone plates{dither_msg} exported to folder:\n{export_folder}\n\n" +
                f"Files: {', '.join(exported_files)}\n" +
                "README.txt included with export details.")
            
        except ImportError:
            QMessageBox.warning(self,"Error","svgwrite module not available")
            return
        try:
            import svgwrite as svg_module
            # Create individual channel files in the folder
            for ch in ["c","m","y","k"]:
                out_path = os.path.join(folder_path, f"halftone_{ch}.svg")
                dwg = svg_module.Drawing(filename=out_path, size=(f"{W}px", f"{H}px"), viewBox=f"0 0 {W} {H}")
                self._export_svg_channel_into(dwg, arrs[ch], p, angles[ch], fill_color="#000000")
                self._add_regmarks_svg(dwg, p, W, H); dwg.save()
            QMessageBox.information(self,"Saved", f"CMYK SVG files saved to folder:\n{os.path.basename(export_folder)}")
        except ImportError:
            QMessageBox.warning(self,"Error","svgwrite module not available")

    def _export_svg_channel_into(self, dwg, ink_arr, p, angle_deg: float, fill_color: str):
        # Apply dithering to the ink array if enabled (same as preview rendering)
        ink_arr = self._maybe_dither_arr(ink_arr, p)
        
        h, w = ink_arr.shape
        th = math.radians(angle_deg); ct,st = math.cos(th), math.sin(th)
        eps = (p.smoothing_pct / 100.0) * (p.cell * 0.8)
        mode = self._shape_id()

        spacing_cell = p.cell * (1.0 + max(0.0, p.dot_gap_pct)/100.0)
        base_cell = p.cell
        
        # Get grid points (vectorized)
        grid_points = self._grid_iter(w, h, spacing_cell, angle_deg)

        if mode == 'lines':
            diag=math.hypot(w,h); span=diag*1.3; step=max(0.75,base_cell*0.3)
            # Iterate over grid points. 
            # Note: The original code used y0 from the grid. 
            # We can iterate over the array.
            for x_pt, y_pt, _ in grid_points:
                pts=[]; u=-span/2
                while u<=span/2:
                    x = w/2 + u*ct - (y_pt-h/2)*st
                    y = h/2 + u*st + (y_pt-h/2)*ct
                    pts.append((x,y)); u+=step
                if len(pts)>=2:
                    simp = self._rdp(pts, eps) if eps>0 else pts
                    dwg.add(dwg.polyline(points=simp, stroke=fill_color, fill='none',
                                         stroke_width=max(0.25,p.stroke), stroke_linecap='round', stroke_linejoin='round'))
            return

        for x, y, _ in grid_points:
            sizef=float(ink_arr[int(y),int(x)]); r=0.5*base_cell*1.414*p.elem*sizef  # sqrt(2) for proper coverage
            if r>0.05:
                if mode == "circle outline":
                    dwg.add(dwg.circle(center=(x,y), r=r, fill='none', stroke=fill_color, stroke_width=max(0.25,p.stroke)))
                else:
                    dwg.add(dwg.circle(center=(x,y), r=r, fill=fill_color, stroke='none'))

    def _add_regmarks_svg(self, dwg, p: Params, w: int, h: int):
        if not p.regs_on: return
        size = p.reg_size_px; off = p.reg_offset_px
        t = max(1.0, size*0.06); half=size*0.5
        for ax,ay in [(off,off),(w-off,off),(w-off,h-off),(off,h-off)]:
            dwg.add(dwg.rect(insert=(ax-half, ay-half), size=(size,size), fill="none", stroke="#000000", stroke_width=t))
            dwg.add(dwg.line(start=(ax-half, ay), end=(ax+half, ay), stroke="#000000", stroke_width=t))
            dwg.add(dwg.line(start=(ax, ay-half), end=(ax, ay+half), stroke="#000000", stroke_width=t))

    def on_export_pdf(self):
        if self.img_full_rgba is None: return
        p = self.params(); tr = self._get_transform()
        
        # Warn about dithering
        if p.dither_enabled:
            reply = QMessageBox.question(self, "Dithering Enabled", 
                "Dithering is enabled. PDF export will create a composite document.\n" +
                "For separate channel files, use SVG or PNG export instead.\n\n" +
                "Continue with PDF composite export?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if reply == QMessageBox.StandardButton.No:
                return
        
        W_IN, H_IN = self._get_doc_size_in(tr.orientation)
        base = self._final_artboard_rgba_with_invert(tr, p.invert_gray)

        if p.grayscale_mode:
            if not base or not np or not Image:
                return
            try:
                L = base.convert("L")
                arrK = np.asarray(L, dtype=np.float32)/255.0
                # Convert lightness to ink coverage: black pixels should have high coverage
                # Since _final_artboard_rgba_with_invert already handled the invert logic,
                # we always need to convert lightness to ink coverage (1.0 - lightness)
                covK = 1.0 - arrK  # Black (0) becomes high coverage (1), White (1) becomes low coverage (0)
                covK = np.clip(covK,0,1) ** self._gamma_from_contrast(p.contrast_pct)
                plateK = self._render_plate_from_arr(covK, p, p.ang_k)
                if plateK and hasattr(plateK, 'size'):
                    comp = Image.new("RGBA", plateK.size, (255,255,255,255))
                    if hasattr(plateK, 'width') and hasattr(plateK, 'height') and hasattr(plateK, 'getchannel'):
                        comp.paste((0,0,0,255), (0,0,plateK.width,plateK.height), plateK.getchannel("A"))
                else:
                    comp = None
            except:
                comp = None
                plateK = None
            if p.regs_on:
                comp = self._paste_regmarks_bitmap(comp, p)
                plateK = self._paste_regmarks_bitmap(plateK, p)
            base_name, _ = QFileDialog.getSaveFileName(self,"export grayscale pdf (composite + K) — choose base name","halftone_11x15","All Files (*.*)")
            if not base_name: return
            root = os.path.splitext(base_name)[0]
            self._save_image_as_pdf(comp, f"{root}_COMPOSITE.pdf", W_IN, H_IN, DOC_DPI)
            self._save_image_as_pdf(plateK, f"{root}_k.pdf", W_IN, H_IN, DOC_DPI)
            QMessageBox.information(self,"saved", f"{root}_COMPOSITE.pdf + _k.pdf")
            return

        base_cmyk = rgba_to_cmyk_with_icc(base, self.cmyk_icc_path)
        c,m,y,k = base_cmyk.split()
        def cov_of(ch_img):
            if not np:
                return None
            try:
                a = np.asarray(ch_img, dtype=np.float32)/255.0
                # Invert logic now handled at image level before CMYK conversion
                return np.clip(a,0,1) ** self._gamma_from_contrast(p.contrast_pct)
            except:
                return None
        arrC,arrM,arrY,arrK = (cov_of(c), cov_of(m), cov_of(y), cov_of(k))
        with ThreadPoolExecutor(max_workers=self._threads) as ex:
            futs = [
                ex.submit(self._render_plate_from_arr, arrC, p, p.ang_c),
                ex.submit(self._render_plate_from_arr, arrM, p, p.ang_m),
                ex.submit(self._render_plate_from_arr, arrY, p, p.ang_y),
                ex.submit(self._render_plate_from_arr, arrK, p, p.ang_k),
            ]
            plates = [f.result() for f in futs]
        if not np:
            return
        cov = []
        for im in plates:
            if im and hasattr(im, 'getchannel'):
                try:
                    cov.append(np.asarray(im.getchannel("A"), dtype=np.float32)/255.0)
                except:
                    cov.append(None)
            else:
                cov.append(None)
        # Apply opacity to coverage arrays (for PDF export)
        opacities = [p.opacity_c, p.opacity_m, p.opacity_y, p.opacity_k]
        if len(cov) >= 4:
            if cov[0] is not None and p.opacity_c < 100.0:
                cov[0] = cov[0] * p.opacity_c / 100.0
            if cov[1] is not None and p.opacity_m < 100.0:
                cov[1] = cov[1] * p.opacity_m / 100.0
            if cov[2] is not None and p.opacity_y < 100.0:
                cov[2] = cov[2] * p.opacity_y / 100.0
            if cov[3] is not None and p.opacity_k < 100.0:
                cov[3] = cov[3] * p.opacity_k / 100.0
            comp = self._composite_from_cov(cov[0],cov[1],cov[2],cov[3],
                                             p.color_c[:3], p.color_m[:3], p.color_y[:3], p.color_k[:3])
        else:
            comp = None
        if p.regs_on and comp:
            comp = self._paste_regmarks_bitmap(comp, p)
            plates = [self._paste_regmarks_bitmap(pl, p) for pl in plates]

        # Use folder approach for PDF exports (like PNG/SVG)
        folder_path = QFileDialog.getExistingDirectory(self, "Choose folder for CMYK PDF export", "")
        if not folder_path: return
        
        # Create a subfolder for this export
        import time
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        export_folder = os.path.join(folder_path, f"halftone_cmyk_pdf_{timestamp}")
        os.makedirs(export_folder, exist_ok=True)
        
        exported_files = []
        if comp:
            comp_path = os.path.join(export_folder, "halftone_COMPOSITE.pdf")
            self._save_image_as_pdf(comp, comp_path, W_IN, H_IN, DOC_DPI, grayscale=False)
            exported_files.append("halftone_COMPOSITE.pdf")
        
        # Export individual plates as grayscale with opacity applied
        for ch, plate, opacity in zip(["c","m","y","k"], plates, opacities):
            if plate and hasattr(plate, 'getchannel'):
                # Get the alpha channel as grayscale
                gray = plate.getchannel("A")
                # Apply opacity if less than 100%
                if opacity < 100.0:
                    gray_arr = np.asarray(gray, dtype=np.float32) * opacity / 100.0
                    if Image and np:
                        gray = Image.fromarray(np.clip(gray_arr, 0, 255).astype(np.uint8), mode="L")
                plate_path = os.path.join(export_folder, f"halftone_{ch}.pdf")
                self._save_image_as_pdf(gray, plate_path, W_IN, H_IN, DOC_DPI, grayscale=True)
                exported_files.append(f"halftone_{ch}.pdf")
        
        # Create README file
        readme_path = os.path.join(export_folder, "README.txt")
        try:
            with open(readme_path, "w") as f:
                f.write(f"CMYK Halftone PDF Export\n")
                f.write(f"========================\n\n")
                f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write(f"Files:\n")
                for fn in exported_files:
                    f.write(f"  - {fn}\n")
                f.write(f"\nOpacity settings:\n")
                f.write(f"  Cyan:    {p.opacity_c:.0f}%\n")
                f.write(f"  Magenta: {p.opacity_m:.0f}%\n")
                f.write(f"  Yellow:  {p.opacity_y:.0f}%\n")
                f.write(f"  Black:   {p.opacity_k:.0f}%\n")
                f.write(f"\nNote: Individual channel PDFs are grayscale with opacity applied.\n")
                f.write(f"The COMPOSITE PDF shows the combined CMYK preview.\n")
        except:
            pass
        
        QMessageBox.information(self,"saved", f"Exported to:\n{export_folder}\n\nFiles: {', '.join(exported_files)}")

    def _save_image_as_pdf(self, img_rgba, pdf_path: str, width_in: float, height_in: float, dpi: int, grayscale: bool = False):
        if not img_rgba:
            return
        target_w, target_h = int(round(width_in*dpi)), int(round(height_in*dpi))
        if hasattr(img_rgba, 'size') and img_rgba.size != (target_w, target_h):
            try:
                if Image and hasattr(Image, 'Resampling'):
                    img_rgba = img_rgba.resize((target_w, target_h), Image.Resampling.LANCZOS)
                else:
                    img_rgba = img_rgba.resize((target_w, target_h), 1)  # 1 = LANCZOS
            except:
                pass
        
        # Determine the output image based on grayscale flag
        if grayscale and hasattr(img_rgba, 'mode'):
            # Keep as grayscale ("L" mode) or convert to it
            if img_rgba.mode != "L":
                out_img = img_rgba.convert("L")
            else:
                out_img = img_rgba
        else:
            out_img = img_rgba
        
        if HAS_REPORTLAB and rl_canvas and 'RL_INCH' in globals():
            try:
                # Get RL_INCH safely - it should be available if HAS_REPORTLAB is True
                rl_inch_val = globals().get('RL_INCH', 72.0)  # 72 points per inch fallback
                c = rl_canvas.Canvas(pdf_path, pagesize=(width_in*rl_inch_val, height_in*rl_inch_val))
                if hasattr(out_img, 'convert'):
                    ir = None
                    if ImageReader:
                        # For grayscale, convert to RGB for ReportLab compatibility
                        # but the visual appearance stays grayscale
                        if grayscale:
                            ir = ImageReader(out_img.convert("RGB"))
                        else:
                            ir = ImageReader(out_img.convert("RGBA"))
                    if ir:
                        c.drawImage(ir, 0, 0, width=width_in*rl_inch_val, height=height_in*rl_inch_val, mask='auto')
                c.showPage(); c.save()
            except:
                pass
        else:
            try:
                if hasattr(out_img, 'convert'):
                    if grayscale:
                        # Save grayscale directly - Pillow PDF supports "L" mode
                        out_img.save(pdf_path, "PDF", resolution=dpi)
                    else:
                        rgb = out_img.convert("RGB")
                        rgb.save(pdf_path, "PDF", resolution=dpi)
            except:
                pass

    # -------- zoom ----------
    def on_wheel_zoom(self, d:int):
        step = 5
        self.zoom.setValue(max(self.zoom.minimum(), min(self.zoom.maximum(), self.zoom.value() + (step if d>0 else -step))))

# ---- Windows AppUserModelID (for taskbar pinning/grouping) ----
def _set_win_appusermodelid(app_id: str = "com.drc.halftone_cmyk"):
    try:
        if sys.platform.startswith("win"):
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(app_id)
    except Exception:
        pass

def main():
    _set_win_appusermodelid()
    app = QApplication(sys.argv)
    app.setApplicationName("drc_halftone_cmyk")
    app.setApplicationDisplayName("DRC Halftone CMYK")
    app.setApplicationVersion("1.0")

    # High-quality rendering settings
    try:
        app.setAttribute(Qt.ApplicationAttribute.AA_UseHighDpiPixmaps, True)
    except Exception:
        pass

    app.setStyleSheet(QSS)

    # Set app icon before creating the window as well (extra robust)
    ic = _load_app_icon()
    if ic:
        app.setWindowIcon(ic)

    w = Main()
    w.resize(1320, 900)
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
