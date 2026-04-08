import os
import uuid
import numpy as np
import cv2
import easyocr

from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO

MODEL_PATH = "best.pt"
OUTPUT_DIR = "output"
UPLOAD_DIR = "uploads"

OCR_FALLBACK_THRESHOLD = 0.3
MAX_FILE_SIZE_MB = 15

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
ALLOWED_MIME_TYPES = {
    "image/jpeg",
    "image/png",
    "image/bmp",
    "image/tiff",
    "image/webp",
    "application/octet-stream",
}

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "crops"), exist_ok=True)

app = FastAPI(title="Coil OCR API")

# statyczne pliki z output/
app.mount("/output", StaticFiles(directory=OUTPUT_DIR), name="output")

model = YOLO(MODEL_PATH)
reader = easyocr.Reader(['en'], gpu=False)

ROTATIONS = {
    "rot0": None,
    "rot90_ccw": cv2.ROTATE_90_COUNTERCLOCKWISE,
    "rot180": cv2.ROTATE_180
}

def decode_image_from_bytes(contents: bytes):
    arr = np.frombuffer(contents, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)

def run_rotation_fallback(reader, image):
    best_text = ""
    best_conf = 0.0
    best_variant = ""

    for name, rot_flag in ROTATIONS.items():
        rotated = image if rot_flag is None else cv2.rotate(image, rot_flag)
        results = reader.readtext(rotated, detail=1)

        for (_, text, conf) in results:
            text = text.strip()
            conf = float(conf)
            if text and conf > best_conf:
                best_text = text
                best_conf = conf
                best_variant = name

    return best_text, best_conf, best_variant

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/ocr")
async def ocr_image(request: Request, file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename")

    contents = await file.read()

    if not contents:
        raise HTTPException(status_code=400, detail="Empty file")

    if len(contents) > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large")

    ext = os.path.splitext(file.filename)[1].lower()
    if ext and ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported extension: {ext}. Allowed: {sorted(ALLOWED_EXTENSIONS)}"
        )

    if file.content_type and file.content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported content type: {file.content_type}"
        )

    img = decode_image_from_bytes(contents)
    if img is None:
        raise HTTPException(status_code=400, detail="Cannot decode image")

    job_id = str(uuid.uuid4())
    safe_ext = ext if ext in ALLOWED_EXTENSIONS else ".jpg"

    input_filename = f"{job_id}{safe_ext}"
    input_path = os.path.join(UPLOAD_DIR, input_filename)
    with open(input_path, "wb") as f:
        f.write(contents)

    # ===== PREPROCESS POD YOLO =====
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    den = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    clahe_img = clahe.apply(den)
    blur = cv2.GaussianBlur(clahe_img, (0, 0), 2.0)
    sharp = cv2.addWeighted(clahe_img, 1.6, blur, -0.6, 0)
    enhanced = cv2.convertScaleAbs(sharp, alpha=1.1, beta=5)
    img_pre = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

    preprocessed_filename = f"{job_id}_preprocessed.jpg"
    preprocessed_path = os.path.join(OUTPUT_DIR, preprocessed_filename)
    cv2.imwrite(preprocessed_path, img_pre)

    # ===== YOLO =====
    results = model.predict(
        source=img_pre,
        conf=0.1,
        imgsz=640,
        verbose=False
    )

    annotated = img.copy()
    detections = []
    box_id = 0

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            det_conf = float(box.conf[0])

            pad = 10
            x1p = max(0, x1 - pad)
            y1p = max(0, y1 - pad)
            x2p = min(img.shape[1], x2 + pad)
            y2p = min(img.shape[0], y2 + pad)

            crop = img[y1p:y2p, x1p:x2p]
            if crop.size == 0:
                continue

            crop_filename = f"{job_id}_crop_{box_id}.jpg"
            crop_path = os.path.join(OUTPUT_DIR, "crops", crop_filename)
            cv2.imwrite(crop_path, crop)

            crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            crop_gray = cv2.fastNlMeansDenoising(crop_gray, None, 10, 7, 21)
            clahe_crop = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(crop_gray)

            ocr_result = reader.readtext(clahe_crop, detail=1)

            best_text = ""
            best_conf = 0.0
            best_variant = "clahe"

            for item in ocr_result:
                text = item[1].strip()
                conf = float(item[2])
                if text and conf > best_conf:
                    best_text = text
                    best_conf = conf

            if best_text == "" or best_conf < OCR_FALLBACK_THRESHOLD:
                fb_text, fb_conf, fb_var = run_rotation_fallback(reader, crop)
                if fb_text and fb_conf > best_conf:
                    best_text = fb_text
                    best_conf = fb_conf
                    best_variant = fb_var

            if best_conf > 0:
                detections.append({
                    "box_id": box_id,
                    "bbox": [x1p, y1p, x2p, y2p],
                    "det_conf": det_conf,
                    "ocr_text": best_text,
                    "ocr_conf": best_conf,
                    "variant": best_variant,
                    "crop_path": os.path.abspath(crop_path),
                    "crop_url": f"/output/crops/{crop_filename}"
                })

                cv2.rectangle(annotated, (x1p, y1p), (x2p, y2p), (255, 0, 0), 2)
                cv2.putText(
                    annotated,
                    f"{best_text} ({best_conf:.2f})",
                    (x1p, max(25, y1p - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 0, 0),
                    2,
                    cv2.LINE_AA
                )

            box_id += 1

    output_filename = f"{job_id}_output_with_ocr.jpg"
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    cv2.imwrite(output_path, annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

    base_url = str(request.base_url).rstrip("/")

    return JSONResponse({
        "success": True,
        "job_id": job_id,
        "filename": file.filename,
        "detections": detections,
        "image_url": f"{base_url}/output/{output_filename}",
        "output_path": os.path.abspath(output_path),
        "preprocessed_url": f"{base_url}/output/{preprocessed_filename}",
        "preprocessed_path": os.path.abspath(preprocessed_path),
        "input_path": os.path.abspath(input_path)
    })