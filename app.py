"""
app.py - Flask web interface for Fake News Detection.

Usage:
    python app.py
    Open http://127.0.0.1:5000 in your browser.
"""

from flask import Flask, render_template, request, jsonify
from predict import load_artifacts, predict

app = Flask(__name__)

# Load model once at startup
vectorizer, model = load_artifacts("lr")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict_route():
    data = request.get_json()
    text = (data or {}).get("text", "").strip()

    if not text:
        return jsonify({"error": "No text provided"}), 400

    result = predict(text, vectorizer, model)
    return jsonify({
        "label":      result["label"],
        "confidence": result["confidence"],
    })


if __name__ == "__main__":
    # use_reloader=False prevents Flask from importing modules twice,
    # which caused MemoryError on systems with limited RAM
    app.run(debug=True, use_reloader=False)
