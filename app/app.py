from __future__ import annotations

import logging
import os
import traceback
import uuid
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # headless backend — must be set before any other matplotlib import

import cv2
import numpy as np
from flask import Flask, flash, jsonify, redirect, render_template, request, send_from_directory, url_for
from werkzeug.utils import secure_filename
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO)


BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_FOLDER = BASE_DIR / "uploads"
RESULT_FOLDER = BASE_DIR / "results"
MODEL_CANDIDATES = [BASE_DIR / "model" / "best.pt", BASE_DIR / "best.pt"]
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}


def get_model_path() -> Path:
    for candidate in MODEL_CANDIDATES:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Could not find best.pt in model/ or project root.")


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def create_app() -> Flask:
    app = Flask(
        __name__,
        template_folder=str(BASE_DIR / "templates"),
        static_folder=str(BASE_DIR / "static"),
    )
    app.config["SECRET_KEY"] = os.getenv("FLASK_SECRET_KEY", "helmet-detection-dev-key")
    app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
    app.config["RESULT_FOLDER"] = str(RESULT_FOLDER)

    UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
    RESULT_FOLDER.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(get_model_path()))

    def run_inference(image_path: Path) -> tuple[list[dict], str]:
        results = model(str(image_path), verbose=False)
        result = results[0]

        predictions: list[dict] = []
        if result.boxes is not None:
            for box in result.boxes:
                class_id = int(box.cls)
                bbox = [float(v) for v in box.xyxy[0].tolist()]
                predictions.append(
                    {
                        "class_id": class_id,
                        "class_name": result.names[class_id],
                        "confidence": round(float(box.conf), 4),
                        "bbox": bbox,
                    }
                )

        annotated = result.plot()
        # ultralytics may return PIL Image or numpy array depending on version
        if not isinstance(annotated, np.ndarray):
            annotated = np.array(annotated)
        annotated_name = f"result_{image_path.stem}.jpg"
        annotated_path = RESULT_FOLDER / annotated_name
        cv2.imwrite(str(annotated_path), annotated)
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
        if file.filename == "":
            flash("Please choose an image file.")
            return redirect(url_for("index"))

        if not allowed_file(file.filename):
            flash("Allowed file types: jpg, jpeg, png, webp.")
            return redirect(url_for("index"))

        unique_name = f"{uuid.uuid4().hex}_{secure_filename(file.filename)}"
        upload_path = UPLOAD_FOLDER / unique_name
        file.save(upload_path)

        try:
            predictions, annotated_name = run_inference(upload_path)
        except Exception:
            logging.error("Inference failed:\n%s", traceback.format_exc())
            flash("Detection failed — please try again with a different image.")
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
        if file.filename == "":
            return jsonify({"error": "empty filename"}), 400

        if not allowed_file(file.filename):
            return jsonify({"error": "unsupported file type"}), 400

        unique_name = f"{uuid.uuid4().hex}_{secure_filename(file.filename)}"
        upload_path = UPLOAD_FOLDER / unique_name
        file.save(upload_path)

        try:
            predictions, annotated_name = run_inference(upload_path)
        except Exception:
            logging.error("Inference failed:\n%s", traceback.format_exc())
            return jsonify({"error": "inference failed", "detail": traceback.format_exc()}), 500

        return jsonify(
            {
                "filename": file.filename,
                "uploaded_image": url_for("uploaded_file", filename=unique_name, _external=True),
                "annotated_image": url_for("result_file", filename=annotated_name, _external=True),
                "prediction_count": len(predictions),
                "predictions": predictions,
            }
        )

    @app.route("/uploads/<path:filename>")
    def uploaded_file(filename: str):
        return send_from_directory(UPLOAD_FOLDER, filename)

    @app.route("/results/<path:filename>")
    def result_file(filename: str):
        return send_from_directory(RESULT_FOLDER, filename)

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({"status": "ok"})

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)





''''Approach 2: Run YOLO Client-Side (In the Browser)
If you want to keep deployment 100% free and don't want to 
maintain a backend, you can convert your YOLO model into 
ONNX or TensorFlow.js format.How it works: This allows the 
AI model to run entirely in the user’s web browser utilizing

 their local GPU/CPU.Benefits: Absolutely zero dependencies,
   no heavy server RAM required, and can be deployed entirely 
   on Vercel.Steps to do it: Use the Ultralytics built-in export
     tool to convert your model to ONNX, then use TensorFlow.js
 inside your Next.js or React application to handle the inference'''