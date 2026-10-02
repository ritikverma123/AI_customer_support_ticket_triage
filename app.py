from pathlib import Path
from datetime import datetime
import csv
import json
import joblib
from flask import Flask, render_template, request, jsonify

BASE = Path(__file__).resolve().parent
MODELS = BASE / "models"
LOGS = BASE / "logs"
LOGS.mkdir(exist_ok=True)

category_model = joblib.load(MODELS / "category_model.joblib")
urgency_model = joblib.load(MODELS / "urgency_model.joblib")

app = Flask(__name__)

QUEUE_MAP = {
    "billing": "Billing Support Queue",
    "technical": "Technical Support Queue",
    "account": "Account Support Queue",
    "product": "Product Support Queue",
}

def predict_ticket(text):
    cat_probs = category_model.predict_proba([text])[0]
    urg_probs = urgency_model.predict_proba([text])[0]

    cat_idx = cat_probs.argmax()
    urg_idx = urg_probs.argmax()

    category = category_model.classes_[cat_idx]
    urgency = urgency_model.classes_[urg_idx]
    category_conf = float(cat_probs[cat_idx])
    urgency_conf = float(urg_probs[urg_idx])
    confidence = min(category_conf, urgency_conf)

    return {
        "category": category,
        "urgency": urgency,
        "category_confidence": round(category_conf, 4),
        "urgency_confidence": round(urgency_conf, 4),
        "confidence": round(confidence, 4),
        "queue": QUEUE_MAP.get(category, "General Support Queue"),
        "human_review": confidence < 0.60
    }

def log_low_confidence(text, result):
    if not result["human_review"]:
        return
    path = LOGS / "low_confidence.csv"
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow([
                "timestamp", "ticket_text", "category", "urgency",
                "confidence", "queue"
            ])
        writer.writerow([
            datetime.now().isoformat(timespec="seconds"),
            text, result["category"], result["urgency"],
            result["confidence"], result["queue"]
        ])

@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    ticket = ""
    if request.method == "POST":
        ticket = request.form.get("ticket", "").strip()
        if ticket:
            result = predict_ticket(ticket)
            log_low_confidence(ticket, result)
    return render_template("index.html", result=result, ticket=ticket)

@app.post("/api/predict")
def api_predict():
    data = request.get_json(silent=True) or {}
    ticket = str(data.get("ticket", "")).strip()
    if not ticket:
        return jsonify({"error": "ticket is required"}), 400
    result = predict_ticket(ticket)
    log_low_confidence(ticket, result)
    return jsonify(result)

@app.get("/health")
def health():
    return jsonify({"status": "ok"})

@app.get("/metrics")
def metrics():
    with (MODELS / "metrics.json").open(encoding="utf-8") as f:
        return jsonify(json.load(f))

if __name__ == "__main__":
    app.run(debug=True)
