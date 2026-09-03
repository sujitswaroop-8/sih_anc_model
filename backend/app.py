"""
app.py

The web server. Exposes the model through two endpoints:

  GET  /api/health   -> tells the frontend whether the model is ready
  POST /api/enhance   -> accepts an uploaded audio file, returns the
                          cleaned version

Run locally with:
    python app.py
Then it's live at http://localhost:5000
"""

import os
import uuid
import tempfile

from flask import Flask, request, send_file, jsonify
from flask_cors import CORS

from inference import enhance_audio, load_model, is_model_loaded, ModelNotLoadedError

app = Flask(__name__)

# Allows your frontend (running on a different domain once deployed) to
# call this backend. For a hackathon demo this is fine wide-open; if you
# want to lock it down later, replace "*" with your actual frontend URL.
CORS(app, expose_headers=["X-Original-SNR", "X-Enhanced-SNR", "X-SNR-Improvement"])

ALLOWED_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}
MAX_CONTENT_LENGTH = 25 * 1024 * 1024  # 25 MB upload limit
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

TMP_DIR = tempfile.gettempdir()

# Try to load the model at startup so the first request isn't slow.
# If the .pkl files aren't there yet, we don't crash the server - we just
# report it as "not ready" via /api/health, so you can build/test the
# frontend before the model files exist.
try:
    load_model()
    print("Model loaded successfully.")
except ModelNotLoadedError as e:
    print(f"[warning] {e}")
    print("Server will still start, but /api/enhance will return an error")
    print("until you add mlp_model.pkl, scaler_X.pkl and scaler_y.pkl to backend/model/")


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": is_model_loaded(),
    })


@app.route("/api/enhance", methods=["POST"])
def enhance():
    if not is_model_loaded():
        try:
            load_model()
        except ModelNotLoadedError as e:
            return jsonify({"error": str(e)}), 503

    if "audio" not in request.files:
        return jsonify({"error": "No file uploaded. Expected form field 'audio'."}), 400

    file = request.files["audio"]
    if file.filename == "":
        return jsonify({"error": "Empty filename."}), 400

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        return jsonify({
            "error": f"Unsupported file type '{ext}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        }), 400

    job_id = uuid.uuid4().hex
    input_path = os.path.join(TMP_DIR, f"{job_id}_in{ext}")
    output_path = os.path.join(TMP_DIR, f"{job_id}_out.wav")

    file.save(input_path)

    try:
        result = enhance_audio(input_path, output_path)
    except Exception as e:
        return jsonify({"error": f"Processing failed: {e}"}), 500
    finally:
        if os.path.exists(input_path):
            os.remove(input_path)

    response = send_file(
        result["output_path"],
        mimetype="audio/wav",
        as_attachment=True,
        download_name="enhanced.wav",
    )
    response.headers["X-Original-SNR"] = str(result["original_snr_db"])
    response.headers["X-Enhanced-SNR"] = str(result["enhanced_snr_db"])
    response.headers["X-SNR-Improvement"] = str(result["improvement_db"])
    return response


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
