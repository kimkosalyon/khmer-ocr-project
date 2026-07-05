import os
import sys
import glob
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

import numpy as np
import streamlit as st
import torch
from PIL import Image, ImageDraw
from safetensors import safe_open
from safetensors.torch import load_file
import torchvision.transforms as T

from src.model import KhmerCRNN_BiGRU
from src.decode import ctc_greedy_decode

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

def get_prediction_transform(preserve_aspect: bool):
    if preserve_aspect:
        return T.Compose([
            T.ToTensor(),
            T.Normalize((0.5,), (0.5,)),
        ])
    else:
        return T.Compose([
            T.Resize((48, 256)),
            T.ToTensor(),
            T.Normalize((0.5,), (0.5,)),
        ])


@st.cache_resource
def load_model(checkpoint_path: str):
    with safe_open(checkpoint_path, framework="pt") as f:
        metadata = f.metadata()
    c2i = json.loads(metadata["c2i"])
    vocab = len(c2i) + 1  # Include CTC blank token (0)
    i2c = {v: k for k, v in c2i.items()}

    state_dict = load_file(checkpoint_path)
    
    # Strip '_orig_mod.' prefix if present from JIT compilation
    cleaned_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."):
            cleaned_state_dict[k[len("_orig_mod."):]] = v
        else:
            cleaned_state_dict[k] = v
            
    # Always load BiGRU as it is the only supported architecture in src/
    model = KhmerCRNN_BiGRU(vocab)
    arch = "BiGRU"

    model.load_state_dict(cleaned_state_dict)
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    return model, i2c, device, metadata, arch


def predict(model, image: Image.Image, i2c: dict, device: str, metadata: dict):
    # 1. Convert to grayscale first
    gray_img = image.convert("L")
    
    # 2. Detect background shade after grayscale
    arr = np.array(gray_img)
    if arr.mean() < 127:
        # Invert colors (make it black text on white background)
        gray_img = Image.fromarray(255 - arr)

    preserve_aspect = metadata.get("preserve_aspect_ratio", "false") == "true"
    if preserve_aspect:
        w, h = gray_img.size
        new_w = max(8, int(w * (48 / h)))
        gray_img = gray_img.resize((new_w, 48), Image.Resampling.BILINEAR)

    transform = get_prediction_transform(preserve_aspect)
    x = transform(gray_img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        log_probs = logits.log_softmax(2).squeeze(1)
    indices = ctc_greedy_decode(log_probs)
    return "".join(i2c.get(i, "?") for i in indices)


@st.cache_resource
def load_detector(detector_path: str):
    if YOLO is None:
        raise RuntimeError("ultralytics is not installed. Run `uv sync` after adding the dependency.")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO(detector_path)
    try:
        model.to(device)
    except Exception as e:
        st.warning(f"Could not move YOLO detector to {device}: {e}")
    return model


def trim_blank_space(image: Image.Image, margin: int = 4) -> Image.Image:
    gray = image.convert("L")
    arr = np.array(gray)
    if arr.mean() < 127:
        mask = arr > 40
    else:
        mask = arr < 245

    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return image

    x1 = max(0, int(xs.min()) - margin)
    y1 = max(0, int(ys.min()) - margin)
    x2 = min(image.width, int(xs.max()) + margin + 1)
    y2 = min(image.height, int(ys.max()) + margin + 1)
    return image.crop((x1, y1, x2, y2))


def detect_text_boxes(detector, image: Image.Image, conf: float):
    results = detector(image, conf=conf, verbose=False)
    boxes = []
    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes.xyxy.cpu().tolist():
            x1, y1, x2, y2 = [int(round(v)) for v in box]
            x1 = max(0, min(image.width - 1, x1))
            y1 = max(0, min(image.height - 1, y1))
            x2 = max(x1 + 1, min(image.width, x2))
            y2 = max(y1 + 1, min(image.height, y2))
            boxes.append((x1, y1, x2, y2))
    boxes.sort(key=lambda b: (b[1], b[0]))
    return boxes


def draw_boxes(image: Image.Image, boxes: list[tuple[int, int, int, int]]) -> Image.Image:
    out = image.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    for i, box in enumerate(boxes, start=1):
        draw.rectangle(box, outline="#ef4444", width=3)
        draw.text((box[0] + 2, max(0, box[1] - 14)), str(i), fill="#ef4444")
    return out


def expand_box(box: tuple[int, int, int, int], image_size: tuple[int, int], padding: int) -> tuple[int, int, int, int]:
    width, height = image_size
    x1, y1, x2, y2 = box
    return (
        max(0, x1 - padding),
        max(0, y1 - padding),
        min(width, x2 + padding),
        min(height, y2 + padding),
    )



st.set_page_config(page_title="Khmer OCR Demo", layout="centered")
st.title("Khmer OCR — CRNN + Decoder")
st.caption("Upload a Khmer text image to test the trained BiGRU or BiLSTM model")

# Search recursively for all safetensors checkpoints in checkpoints and checkpoints_200k
checkpoints = sorted(
    glob.glob("checkpoints*/**/*.safetensors", recursive=True),
    key=os.path.getmtime,
    reverse=True
)

with st.sidebar:
    st.header("Model Setup")
    if not checkpoints:
        st.error("No checkpoints found in checkpoints/ or checkpoints_200k/ directories.")
        st.stop()

    # Display relative path from workspace root
    selected_rel = st.selectbox("Select Checkpoint", [os.path.relpath(c, ".") for c in checkpoints])
    checkpoint_path = selected_rel


    model, i2c, device, meta, arch = load_model(checkpoint_path)
    
    st.success(f"Loaded successfully!")
    st.metric("Architecture", arch)
    st.metric("Epoch", meta.get("epoch", "N/A"))
    st.metric("Device", device.upper())
    st.metric("Loss (at save)", f"{float(meta.get('loss', 0.0)):.4f}")
    st.caption(f"Path: `{checkpoint_path}`")

    st.divider()
    st.header("Text Detection")
    
    # Ensure detectors directory exists
    os.makedirs("detectors", exist_ok=True)
    
    detector_paths = sorted(
        glob.glob("runs/detect/**/weights/best.pt", recursive=True)
        + glob.glob("detectors/**/*.pt", recursive=True),
        key=os.path.getmtime,
        reverse=True,
    )
    
    # Compile options, adding yolov8n.pt as default base detector option
    detector_options = [os.path.relpath(p, ".") for p in detector_paths]
    if "yolov8n.pt" not in detector_options:
        detector_options.append("yolov8n.pt")
        
    custom_detector_path = st.text_input("Or enter custom YOLO path", "")
    if custom_detector_path and os.path.exists(custom_detector_path):
        rel_custom = os.path.relpath(custom_detector_path, ".")
        if rel_custom not in detector_options:
            detector_options.insert(0, rel_custom)
            
    use_detector = st.checkbox("Use YOLO text detector", value=True)
    detector = None
    det_conf = 0.25
    trim_crops = False
    crop_padding = 8
    
    detector_rel = st.selectbox("Detector checkpoint", detector_options, index=detector_options.index("yolov8n.pt") if "yolov8n.pt" in detector_options else 0)
    det_conf = st.slider("Detector confidence", 0.05, 0.95, 0.25, 0.05)
    crop_padding = st.slider("Crop padding for Khmer marks", 0, 32, 8, 1)
    trim_crops = st.checkbox("Tight-trim crops (using pixel threshold)", value=False, help="May cut off faint subscripts/superscripts")
    
    if use_detector:
        detector = load_detector(detector_rel)

uploaded = st.file_uploader("Upload a Khmer text image", type=["png", "jpg", "jpeg", "bmp", "webp"])

if uploaded is not None:
    image = Image.open(uploaded)
    
    # 1. Run YOLO text detector first if loaded
    boxes = []
    if detector is not None:
        with st.spinner("Detecting text regions..."):
            boxes = detect_text_boxes(detector, image.convert("RGB"), det_conf)
            
    col1, col2 = st.columns(2)
    with col1:
        if detector is not None and boxes:
            st.image(draw_boxes(image, boxes), caption=f"{len(boxes)} detected text region(s) (YOLO)", use_container_width=True)
        else:
            st.image(image, caption="Uploaded Image", use_container_width=True)
            if detector is not None:
                st.warning("No text boxes detected by YOLO. Falling back to whole-image prediction.")
                
    with col2:
        with st.spinner("Running OCR..."):
            if detector is not None and boxes:
                # Crop and predict each box
                preds = []
                crops_and_preds = []
                for idx, box in enumerate(boxes, start=1):
                    padded_box = expand_box(box, image.size, crop_padding)
                    crop = image.crop(padded_box)
                    if trim_crops:
                        crop = trim_blank_space(crop, margin=crop_padding)
                    pred = predict(model, crop, i2c, device, meta)
                    preds.append(pred)
                    crops_and_preds.append((idx, crop, pred))
                text = " ".join(preds)
            else:
                # Fallback: OCR on the entire image
                text = predict(model, trim_blank_space(image), i2c, device, meta)
                crops_and_preds = []
                
        st.markdown("### Predicted Text")
        st.markdown(f"# {text}")
        st.caption(f"{len(text)} characters")
        
        # Show crops details
        if crops_and_preds:
            st.markdown("### Detected Regions Detail")
            for idx, crop, pred in crops_and_preds:
                with st.expander(f"Region {idx}: {pred[:80]}", expanded=idx == 1):
                    st.image(crop, caption=f"Crop {idx} ({crop.size[0]}x{crop.size[1]}) with {crop_padding}px padding")
                    st.code(pred)
else:
    st.info("Upload an image to see the OCR prediction.")
