from __future__ import annotations

import logging
import os
import traceback
import uuid
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from flask import (Flask, flash, jsonify, redirect, render_template,
                   request, send_from_directory, url_for)
from werkzeug.utils import secure_filename

logging.basicConfig(level=logging.INFO)

BASE_DIR        = Path(__file__).resolve().parent.parent
UPLOAD_FOLDER   = BASE_DIR / "uploads"
RESULT_FOLDER   = BASE_DIR / "results"
MODEL_PATH      = BASE_DIR / "model" / "best.onnx"
ALLOWED_EXT     = {"jpg", "jpeg", "png", "webp"}
CLASS_NAMES     = {0: "head"}   # single class trained
IMG_SIZE        = 640
CONF_THRESH     = 0.25
IOU_THRESH      = 0.45


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


def letterbox(img: np.ndarray, size: int = IMG_SIZE):
    """Resize with padding to square, return (padded_img, scale, pad_x, pad_y)."""
    h, w = img.shape[:2]
    scale = min(size / w, size / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    pad_x, pad_y = (size - nw) // 2, (size - nh) // 2
    canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = resized
    return canvas, scale, pad_x, pad_y


def preprocess(image_path: Path):
    img_bgr = cv2.imread(str(image_path))
    orig_h, orig_w = img_bgr.shape[:2]
    padded, scale, pad_x, pad_y = letterbox(img_bgr)
    blob = padded[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
    return np.expand_dims(blob, 0), img_bgr, scale, pad_x, pad_y


def postprocess(output: np.ndarray, scale: float, pad_x: int, pad_y: int):
    """YOLOv8 output: [1, 5, 8400] → (cx,cy,w,h, cls0_score)."""
    pred = output[0].T          # [8400, 5]
    cx, cy, bw, bh = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
    scores = pred[:, 4:]        # [8400, num_classes]
    class_ids = np.argmax(scores, axis=1)
    confidences = np.max(scores, axis=1)

    mask = confidences >= CONF_THRESH
    cx, cy, bw, bh = cx[mask], cy[mask], bw[mask], bh[mask]
    confidences = confidences[mask]
    class_ids = class_ids[mask]

    # Convert from padded 640×640 space back to original image coords
    x1 = (cx - bw / 2 - pad_x) / scale
    y1 = (cy - bh / 2 - pad_y) / scale
    x2 = (cx + bw / 2 - pad_x) / scale
    y2 = (cy + bh / 2 - pad_y) / scale

    boxes_xywh = [[float(x1[i]), float(y1[i]),
                   float(x2[i] - x1[i]), float(y2[i] - y1[i])]
                  for i in range(len(x1))]

    if not boxes_xywh:
        return []

    indices = cv2.dnn.NMSBoxes(boxes_xywh, confidences.tolist(),
                                CONF_THRESH, IOU_THRESH)
    predictions = []
    for i in (indices.flatten() if hasattr(indices, "flatten") else indices):
        predictions.append({
            "class_id":   int(class_ids[i]),
            "class_name": CLASS_NAMES.get(int(class_ids[i]), str(class_ids[i])),
            "confidence": round(float(confidences[i]), 4),
            "bbox":       [float(x1[i]), float(y1[i]), float(x2[i]), float(y2[i])],
        })
    return predictions


def draw_boxes(img_bgr: np.ndarray, predictions: list[dict]) -> np.ndarray:
    for p in predictions:
        x1, y1, x2, y2 = [int(v) for v in p["bbox"]]
        label = f"{p['class_name']} {p['confidence']:.0%}"
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 200, 0), 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(img_bgr, (x1, y1 - th - 6), (x1 + tw + 4, y1), (0, 200, 0), -1)
        cv2.putText(img_bgr, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
    return img_bgr


def create_app() -> Flask:
    app = Flask(
        __name__,
        template_folder=str(BASE_DIR / "templates"),
        static_folder=str(BASE_DIR / "static"),
    )
    app.config["SECRET_KEY"] = os.getenv("FLASK_SECRET_KEY", "helmet-onnx-key")
    UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
    RESULT_FOLDER.mkdir(parents=True, exist_ok=True)

    # Load ONNX model once at startup
    session = ort.InferenceSession(str(MODEL_PATH),
                                   providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    logging.info("ONNX model loaded: %s  input=%s", MODEL_PATH.name, input_name)

    def run_inference(image_path: Path):
        blob, img_bgr, scale, pad_x, pad_y = preprocess(image_path)
        output = session.run(None, {input_name: blob})
        predictions = postprocess(output[0], scale, pad_x, pad_y)
        annotated = draw_boxes(img_bgr.copy(), predictions)
        annotated_name = f"result_{image_path.stem}.jpg"
        cv2.imwrite(str(RESULT_FOLDER / annotated_name), annotated)
        return predictions, annotated_name

    @app.route("/", methods=["GET"])
    def index():
        return render_template("index.html")

    @app.route("/predict", methods=["POST"])
    def predict_web():
        if "image" not in request.files:
            flash("Please choose an image file.")
            return redirect(url_for("index"))
        file = request.files["image"]
        if not file.filename:
            flash("Please choose an image file.")
            return redirect(url_for("index"))
        if not allowed_file(file.filename):
            flash("Allowed types: jpg, jpeg, png, webp.")
            return redirect(url_for("index"))

        unique_name = f"{uuid.uuid4().hex}_{secure_filename(file.filename)}"
        upload_path = UPLOAD_FOLDER / unique_name
        file.save(upload_path)

        try:
            predictions, annotated_name = run_inference(upload_path)
        except Exception:
            logging.error("Inference failed:\n%s", traceback.format_exc())
            flash("Detection failed — please try a different image.")
            return redirect(url_for("index"))

        return render_template(
            "result.html",
            original_image=url_for("uploaded_file", filename=unique_name),
            annotated_image=url_for("result_file", filename=annotated_name),
            predictions=predictions,
            prediction_count=len(predictions),
            filename=file.filename,
        )

    @app.route("/api/predict", methods=["POST"])
    def predict_api():
        if "image" not in request.files:
            return jsonify({"error": "no image provided"}), 400
        file = request.files["image"]
        if not allowed_file(file.filename or ""):
            return jsonify({"error": "unsupported file type"}), 400

        unique_name = f"{uuid.uuid4().hex}_{secure_filename(file.filename)}"
        upload_path = UPLOAD_FOLDER / unique_name
        file.save(upload_path)

        try:
            predictions, annotated_name = run_inference(upload_path)
        except Exception:
            logging.error("Inference failed:\n%s", traceback.format_exc())
            return jsonify({"error": "inference failed"}), 500

        return jsonify({
            "filename": file.filename,
            "uploaded_image":  url_for("uploaded_file", filename=unique_name, _external=True),
            "annotated_image": url_for("result_file",   filename=annotated_name, _external=True),
            "prediction_count": len(predictions),
            "predictions": predictions,
        })

    @app.route("/uploads/<path:filename>")
    def uploaded_file(filename: str):
        return send_from_directory(UPLOAD_FOLDER, filename)

    @app.route("/results/<path:filename>")
    def result_file(filename: str):
        return send_from_directory(RESULT_FOLDER, filename)

    @app.route("/health")
    def health():
        return jsonify({"status": "ok", "model": "best.onnx", "classes": CLASS_NAMES})

    return app


app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
