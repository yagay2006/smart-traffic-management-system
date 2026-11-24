import os
import json
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parent
try:
    from dotenv import load_dotenv
    load_dotenv(BASE_DIR / ".env", override=False)
except ImportError:
    env_file = BASE_DIR / ".env"
    if env_file.is_file():
        for line in env_file.read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                name, value = line.split("=", 1)
                name, value = name.strip(), value.strip().strip("\"'")
                if name and name not in os.environ:
                    os.environ[name] = value

import numpy as np
import pandas as pd
import requests
from flask import Flask, jsonify, request, render_template

try:
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

app = Flask(__name__)

@app.after_request
def disable_api_caching(response):
    if request.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response

DEFAULT_LATITUDE = float(os.getenv("TRAFFIC_LATITUDE", "28.4506"))
DEFAULT_LONGITUDE = float(os.getenv("TRAFFIC_LONGITUDE", "77.5840"))
WEATHER_API = "https://api.open-meteo.com/v1/forecast"
TOMTOM_API_KEY = os.getenv("TOMTOM_API_KEY", "").strip()
_dataset_setting = os.getenv("TRAFFIC_DATASET", "traffic_data.csv").strip()
DATASET_PATH = Path(_dataset_setting)
if not DATASET_PATH.is_absolute():
    DATASET_PATH = BASE_DIR / DATASET_PATH
REQUEST_TIMEOUT = 10
FEATURES = ["hour", "traffic_volume", "temperature", "rainfall"]
LABELS = ["Low", "Medium", "High"]


def get_weather(latitude=DEFAULT_LATITUDE, longitude=DEFAULT_LONGITUDE):
    params = {
        "latitude": latitude, "longitude": longitude,
        "current": "temperature_2m,rain,wind_speed_10m",
        "hourly": "temperature_2m,rain,precipitation_probability",
        "timezone": "auto", "forecast_days": 1,
    }
    try:
        response = requests.get(WEATHER_API, params=params, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        return response.json()
    except (requests.RequestException, ValueError) as exc:
        app.logger.warning("Weather API request failed: %s", exc)
        return None


def get_live_traffic(latitude=DEFAULT_LATITUDE, longitude=DEFAULT_LONGITUDE):
    if not TOMTOM_API_KEY:
        return {"available": False, "source": "TomTom", "message": "Set TOMTOM_API_KEY in your .env file to enable live road traffic."}
    url = "https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json"
    params = {"key": TOMTOM_API_KEY, "point": f"{latitude},{longitude}", "unit": "KMPH", "thickness": 10}
    try:
        response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        flow = response.json().get("flowSegmentData", {})
        current_seconds = flow.get("currentTravelTime")
        free_seconds = flow.get("freeFlowTravelTime")
        current_seconds = float(current_seconds) if current_seconds is not None else None
        free_seconds = float(free_seconds) if free_seconds is not None else None
        current_minutes = round(current_seconds / 60, 2) if current_seconds and current_seconds > 0 else None
        free_minutes = round(free_seconds / 60, 2) if free_seconds and free_seconds > 0 else None
        delay = round(max(0, current_minutes - free_minutes), 2) if current_minutes is not None and free_minutes is not None else None
        current_speed, free_speed = flow.get("currentSpeed"), flow.get("freeFlowSpeed")
        ratio = round(float(current_speed) / float(free_speed), 3) if current_speed is not None and free_speed not in (None, 0) else None
        if flow.get("roadClosure"):
            status = "Road closure reported"
        elif ratio is None:
            status = "Traffic status unavailable"
        elif ratio < 0.4:
            status = "Severe congestion"
        elif ratio < 0.7:
            status = "Heavy traffic"
        elif ratio < 0.9:
            status = "Moderate traffic"
        else:
            status = "Near free-flow"
        return {
            "available": True, "source": "TomTom Traffic Flow API",
            "current_speed_kmph": current_speed, "free_flow_speed_kmph": free_speed,
            "current_travel_time_seconds": current_seconds if current_seconds and current_seconds > 0 else None,
            "free_flow_travel_time_seconds": free_seconds if free_seconds and free_seconds > 0 else None,
            "current_travel_time_minutes": current_minutes, "free_flow_travel_time_minutes": free_minutes,
            "estimated_delay_minutes": delay, "congestion_ratio": ratio, "traffic_status": status,
            "confidence": flow.get("confidence"), "road_closure": flow.get("roadClosure"),
            "note": "Travel time refers to the road segment returned by TomTom, not a complete origin-to-destination route."
        }
    except (requests.RequestException, ValueError) as exc:
        app.logger.warning("TomTom traffic request failed: %s", exc)
        return {"available": False, "source": "TomTom", "message": str(exc)}


def get_time_period(hour):
    if 7 <= hour <= 9: return "Morning Peak"
    if 10 <= hour <= 12: return "Morning Off-Peak"
    if 13 <= hour <= 16: return "Afternoon"
    if 17 <= hour <= 20: return "Evening Peak"
    if 21 <= hour <= 23: return "Night"
    return "Late Night"


def calculate_congestion(traffic_volume, hour, rainfall, temperature):
    score = 50 if traffic_volume > 800 else 40 if traffic_volume > 600 else 25 if traffic_volume > 400 else 10
    if 7 <= hour <= 9: score += 30
    elif 17 <= hour <= 20: score += 35
    elif 10 <= hour <= 12: score += 15
    elif 13 <= hour <= 16: score += 10
    elif 21 <= hour <= 23: score += 5
    else: score += 2
    if rainfall > 5: score += 20
    elif rainfall > 2: score += 10
    elif rainfall > 0: score += 5
    if temperature > 35: score += 5
    return ("High" if score >= 75 else "Medium" if score >= 45 else "Low"), score


def get_recommendation(hour, congestion):
    if congestion == "High":
        if 7 <= hour <= 9: return "Morning peak is busy; consider travelling between 10:00 and 12:00."
        if 17 <= hour <= 20: return "Evening peak is busy; consider travelling after 21:00."
        return "Consider delaying the journey and checking live road conditions."
    if congestion == "Medium": return "Allow extra time and check road conditions before leaving."
    return "The estimate indicates relatively favourable conditions."


def make_demo_dataset(rows=1500):
    rng = np.random.default_rng(42)
    frame = pd.DataFrame({
        "hour": rng.integers(0, 24, rows), "traffic_volume": rng.integers(100, 1000, rows),
        "temperature": rng.uniform(15, 42, rows).round(1), "rainfall": rng.uniform(0, 10, rows).round(1)
    })
    labels_scores = frame.apply(lambda r: calculate_congestion(float(r.traffic_volume), int(r.hour), float(r.rainfall), float(r.temperature)), axis=1)
    frame["congestion"] = labels_scores.apply(lambda x: x[0])
    frame["congestion_score"] = labels_scores.apply(lambda x: x[1])
    frame["time_period"] = frame["hour"].apply(get_time_period)
    frame["data_source"] = "synthetic_demo"
    return frame


def load_dataset():
    if not DATASET_PATH.exists():
        return make_demo_dataset(), "synthetic_demo", f"{DATASET_PATH} not found; generated demonstration data was used."
    try:
        frame = pd.read_csv(DATASET_PATH)
        missing = [c for c in FEATURES if c not in frame.columns]
        if missing: raise ValueError("Missing required columns: " + ", ".join(missing))
        for column in FEATURES: frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame = frame.dropna(subset=FEATURES)
        frame = frame[frame.hour.between(0, 23) & frame.traffic_volume.ge(0) & frame.rainfall.ge(0)].copy()
        if frame.empty: raise ValueError("CSV contains no valid rows after cleaning.")
        if "congestion" not in frame.columns:
            values = frame.apply(lambda r: calculate_congestion(float(r.traffic_volume), int(r.hour), float(r.rainfall), float(r.temperature)), axis=1)
            frame["congestion"] = values.apply(lambda x: x[0])
            frame["congestion_score"] = values.apply(lambda x: x[1])
        else:
            frame["congestion"] = frame["congestion"].astype(str).str.title()
            frame = frame[frame.congestion.isin(LABELS)].copy()
            if frame.empty: raise ValueError("congestion must contain Low, Medium, or High labels.")
            if "congestion_score" not in frame.columns:
                frame["congestion_score"] = frame.apply(lambda r: calculate_congestion(float(r.traffic_volume), int(r.hour), float(r.rainfall), float(r.temperature))[1], axis=1)
        frame["time_period"] = frame.hour.astype(int).apply(get_time_period)
        frame["data_source"] = "csv_dataset"
        return frame, "csv_dataset", f"Loaded and validated {len(frame)} rows from {DATASET_PATH}."
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        app.logger.warning("Dataset could not be loaded: %s", exc)
        return make_demo_dataset(), "synthetic_demo", f"Dataset loading failed ({exc}); generated demonstration data was used."


df, DATA_SOURCE, DATASET_MESSAGE = load_dataset()


def train_models(frame):
    if not SKLEARN_AVAILABLE:
        return {}, {"available": False, "message": "Install scikit-learn to enable ML models."}
    if frame.congestion.nunique() < 2 or len(frame) < 20:
        return {}, {"available": False, "message": "At least 20 rows and two target classes are needed."}
    X, y = frame[FEATURES].astype(float), frame.congestion.astype(str)
    stratify = y if y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=stratify)
    models = {
        "Decision Tree": Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", DecisionTreeClassifier(max_depth=8, min_samples_leaf=3, random_state=42))]),
        "Logistic Regression": Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler()), ("model", LogisticRegression(max_iter=500, class_weight="balanced"))])
    }
    metrics = {"available": True, "training_rows": int(len(X_train)), "testing_rows": int(len(X_test)), "models": {}}
    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            predicted = model.predict(X_test)
            metrics["models"][name] = {
                "accuracy": round(float(accuracy_score(y_test, predicted)), 4),
                "report": classification_report(y_test, predicted, labels=LABELS, output_dict=True, zero_division=0),
                "confusion_matrix_labels": LABELS,
                "confusion_matrix": confusion_matrix(y_test, predicted, labels=LABELS).tolist()
            }
        except (ValueError, TypeError) as exc:
            metrics["models"][name] = {"error": str(exc)}
    return models, metrics


MODELS, MODEL_METRICS = train_models(df)


def parse_float_arg(name, minimum=None, maximum=None, default=None):
    raw = request.args.get(name)
    if raw is None or raw.strip() == "":
        if default is not None: return float(default)
        raise ValueError(f"Missing required parameter: {name}")
    try: value = float(raw)
    except ValueError: raise ValueError(f"{name} must be numeric.")
    if not np.isfinite(value): raise ValueError(f"{name} must be finite.")
    if minimum is not None and value < minimum: raise ValueError(f"{name} must be at least {minimum}.")
    if maximum is not None and value > maximum: raise ValueError(f"{name} must be at most {maximum}.")
    return value


@app.route("/")
def dashboard():
    return render_template("index.html")


@app.route("/api/health")
def health():
    return jsonify({"status": "running", "project": "Smart Traffic Management System", "time": datetime.now().astimezone().isoformat(), "sklearn_available": SKLEARN_AVAILABLE})


@app.route("/api/weather")
def weather_api():
    try:
        latitude = parse_float_arg("latitude", -90, 90, DEFAULT_LATITUDE)
        longitude = parse_float_arg("longitude", -180, 180, DEFAULT_LONGITUDE)
    except ValueError as exc: return jsonify({"error": str(exc)}), 400
    data = get_weather(latitude, longitude)
    if data is None: return jsonify({"error": "Weather API is unavailable. Check internet connection and try again."}), 502
    current = data.get("current", {})
    return jsonify({"source": "Open-Meteo", "location": {"latitude": latitude, "longitude": longitude}, "time": current.get("time"), "temperature": current.get("temperature_2m"), "rain": current.get("rain"), "wind_speed": current.get("wind_speed_10m")})


@app.route("/api/traffic")
def traffic_api():
    try:
        latitude = parse_float_arg("latitude", -90, 90, DEFAULT_LATITUDE)
        longitude = parse_float_arg("longitude", -180, 180, DEFAULT_LONGITUDE)
    except ValueError as exc: return jsonify({"error": str(exc)}), 400
    return jsonify(get_live_traffic(latitude, longitude))


@app.route("/api/predict")
def predict_api():
    try:
        hour = int(parse_float_arg("hour", 0, 23))
        traffic_volume = parse_float_arg("traffic_volume", 0, 10000000)
        temperature = parse_float_arg("temperature", -100, 70)
        rainfall = parse_float_arg("rainfall", 0, 1000)
    except ValueError as exc: return jsonify({"error": str(exc)}), 400
    selected_model = request.args.get("model", "rules")
    features = pd.DataFrame([{"hour": hour, "traffic_volume": traffic_volume, "temperature": temperature, "rainfall": rainfall}])
    if selected_model == "rules":
        level, score = calculate_congestion(traffic_volume, hour, rainfall, temperature)
        method, note = "Rule-based estimator", "A rule-based estimate, not an ML model prediction."
    else:
        if selected_model not in MODELS: return jsonify({"error": "Requested ML model is unavailable. Install scikit-learn and check the dataset."}), 400
        level, score = str(MODELS[selected_model].predict(features[FEATURES])[0]), None
        method, note = selected_model, "ML output depends on the training dataset; if the dataset is synthetic, this is only a demonstration."
    return jsonify({"method": method, "input": {"hour": hour, "traffic_volume": traffic_volume, "temperature": temperature, "rainfall": rainfall}, "congestion": level, "congestion_score": score, "time_period": get_time_period(hour), "recommendation": get_recommendation(hour, level), "note": note})


@app.route("/api/best-time")
def best_time_api():
    hourly = df.groupby("hour").agg(average_traffic=("traffic_volume", "mean"), average_score=("congestion_score", "mean"))
    candidates = []
    for hour in range(24):
        if hour in hourly.index and pd.notna(hourly.loc[hour, "average_score"]):
            volume, score = float(hourly.loc[hour, "average_traffic"]), float(hourly.loc[hour, "average_score"])
            basis = "loaded dataset" if DATA_SOURCE == "csv_dataset" else "synthetic demonstration dataset"
        else:
            volume = 750 if 7 <= hour <= 9 else 850 if 17 <= hour <= 20 else 450 if 10 <= hour <= 16 else 300 if 21 <= hour <= 23 else 250
            _, score = calculate_congestion(volume, hour, 0, 25)
            basis = "illustrative fallback for hours missing from the dataset"
        level = "High" if score >= 75 else "Medium" if score >= 45 else "Low"
        candidates.append({"hour": hour, "best_time": f"{hour:02d}:00", "congestion": level, "congestion_score": round(score, 2), "average_traffic": round(volume, 2), "time_period": get_time_period(hour), "basis": basis})
    best = min(candidates, key=lambda item: (item["congestion_score"], item["average_traffic"], item["hour"]))
    return jsonify({**best, "message": "This is a recommended departure time, not a journey-duration estimate.", "ranked_hours": sorted(candidates, key=lambda item: (item["congestion_score"], item["average_traffic"], item["hour"]))})


@app.route("/api/trip-time")
def trip_time_api():
    try:
        distance_km = parse_float_arg("distance_km", 0.1, 10000)
        speed_kmph = parse_float_arg("speed_kmph", 1, 180)
    except ValueError as exc: return jsonify({"error": str(exc)}), 400
    minutes = distance_km / speed_kmph * 60
    rounded_minutes = max(1, int(round(minutes)))
    hours, mins = divmod(rounded_minutes, 60)
    duration_text = f"{hours} h {mins} min" if hours else f"{mins} min"
    return jsonify({"distance_km": round(distance_km, 2), "speed_kmph": round(speed_kmph, 2), "duration_minutes": round(minutes, 2), "duration_text": duration_text, "basis": "distance divided by user-supplied average speed"})


@app.route("/api/analytics")
def analytics_api():
    counts = df.congestion.value_counts()
    hourly = df.groupby("hour").agg(average_traffic=("traffic_volume", "mean"), average_score=("congestion_score", "mean")).reindex(range(24))
    return jsonify({
        "total_records": int(len(df)), "dataset_source": DATA_SOURCE, "dataset_message": DATASET_MESSAGE,
        "distribution": {"labels": LABELS, "values": [int(counts.get(label, 0)) for label in LABELS]},
        "hourly": {"hours": list(range(24)), "average_traffic": [None if pd.isna(v) else round(float(v), 2) for v in hourly.average_traffic], "average_score": [None if pd.isna(v) else round(float(v), 2) for v in hourly.average_score]},
        "scatter": [{"x": float(row.traffic_volume), "y": float(row.temperature)} for row in df.itertuples()],
        "model_metrics": MODEL_METRICS
    })


@app.route("/api/dataset")
def dataset_api():
    return jsonify({"source": DATA_SOURCE, "message": DATASET_MESSAGE, "rows": int(len(df)), "columns": list(df.columns), "missing_values": {k: int(v) for k, v in df.isnull().sum().items()}, "sample": json.loads(df.head(10).to_json(orient="records"))})


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    print("Smart Traffic Management System")
    print(f"Dashboard: http://127.0.0.1:{port}")
    print(f"Health:    http://127.0.0.1:{port}/api/health")
    print(f"Dataset:   {DATASET_MESSAGE}")
    print(f"TomTom API key loaded: {bool(TOMTOM_API_KEY)}")
    print("Live traffic uses the configured TomTom key and default map point; change TRAFFIC_LATITUDE/LONGITUDE for your area.")
    if not SKLEARN_AVAILABLE: print("Machine-learning models disabled. Install scikit-learn to enable them.")
    app.run(host="127.0.0.1", port=port, debug=False)
