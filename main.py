import socket
import json
import threading
import time
from collections import deque
from datetime import datetime
from flask import Flask, render_template, jsonify
from flask_cors import CORS
import numpy as np

# Initialize Flask App
app = Flask(__name__)
CORS(app)

# Global variables untuk menyimpan data real-time
latest_data = deque(maxlen=100)
alerts = deque(maxlen=50)
statistics = {
    "total_patients": 0,
    "critical_count": 0,
    "warning_count": 0,
    "normal_count": 0,
    "avg_heart_rate": 0,
    "avg_systolic": 0,
    "avg_diastolic": 0,
    "last_updated": ""
}

class SimpleKMeans:
    """Simple K-Means implementation for anomaly detection"""
    def __init__(self, k=3):
        self.k = k
        self.centroids = None
    
    def fit_predict(self, data):
        if len(data) < self.k:
            return [0] * len(data)
        
        # Initialize centroids randomly
        np.random.seed(42)
        indices = np.random.choice(len(data), self.k, replace=False)
        self.centroids = [data[i] for i in indices]
        
        # Simple k-means (3 iterations)
        for _ in range(3):
            clusters = [[] for _ in range(self.k)]
            labels = []
            
            # Assign points to nearest centroid
            for point in data:
                distances = [np.linalg.norm(np.array(point) - np.array(c)) for c in self.centroids]
                label = distances.index(min(distances))
                clusters[label].append(point)
                labels.append(label)
            
            # Update centroids
            for i in range(self.k):
                if clusters[i]:
                    self.centroids[i] = np.mean(clusters[i], axis=0).tolist()
        
        return labels

def calculate_risk_score(heart_rate, systolic, diastolic):
    """Calculate risk score based on vital signs"""
    score = 0
    
    # Heart rate scoring
    if heart_rate >= 130:
        score += 10
    elif heart_rate >= 110:
        score += 5
    elif heart_rate < 60:
        score += 3
    
    # Systolic BP scoring
    if systolic >= 160:
        score += 10
    elif systolic >= 140:
        score += 5
    elif systolic < 90:
        score += 3
    
    # Diastolic BP scoring
    if diastolic >= 100:
        score += 10
    elif diastolic >= 90:
        score += 5
    elif diastolic < 60:
        score += 3
    
    return score

def classify_status(risk_score):
    """Classify patient status based on risk score"""
    if risk_score >= 15:
        return "Critical"
    elif risk_score >= 5:
        return "Warning"
    else:
        return "Normal"

def detect_anomaly(all_data):
    """Detect anomalies using clustering"""
    if len(all_data) < 10:
        return [False] * len(all_data)
    
    # Prepare data for clustering
    features = [[d['heart_rate'], d['systolic'], d['diastolic']] for d in all_data]
    
    # Normalize features
    features_array = np.array(features)
    mean = features_array.mean(axis=0)
    std = features_array.std(axis=0)
    std[std == 0] = 1  # Avoid division by zero
    normalized = ((features_array - mean) / std).tolist()
    
    # Apply K-Means
    kmeans = SimpleKMeans(k=3)
    labels = kmeans.fit_predict(normalized)
    
    # Find the smallest cluster (likely anomalies)
    cluster_sizes = {i: labels.count(i) for i in range(3)}
    anomaly_cluster = min(cluster_sizes, key=cluster_sizes.get)
    
    # Mark points in smallest cluster as anomalies
    return [labels[i] == anomaly_cluster for i in range(len(labels))]

def process_data(data):
    """Process incoming data with ML models"""
    global latest_data, alerts, statistics
    
    try:
        # Parse JSON
        patient_data = json.loads(data)
        
        # Calculate risk score
        risk_score = calculate_risk_score(
            patient_data['heart_rate'],
            patient_data['systolic'],
            patient_data['diastolic']
        )
        
        # Classify status
        status = classify_status(risk_score)
        
        # Create data point
        data_point = {
            "patient_id": patient_data['patient_id'],
            "patient_name": patient_data['patient_name'],
            "room": patient_data['room'],
            "heart_rate": patient_data['heart_rate'],
            "systolic": patient_data['systolic'],
            "diastolic": patient_data['diastolic'],
            "timestamp": patient_data['timestamp'],
            "status": status,
            "risk_score": risk_score,
            "is_anomaly": False
        }
        
        # Add to latest data
        latest_data.append(data_point)
        
        # Detect anomalies on recent data
        if len(latest_data) >= 10:
            recent_data = list(latest_data)[-20:]  # Last 20 points
            anomalies = detect_anomaly(recent_data)
            
            # Update anomaly status
            for i, is_anomaly in enumerate(anomalies):
                recent_data[i]['is_anomaly'] = is_anomaly
            
            # Update the last point
            data_point['is_anomaly'] = anomalies[-1]
        
        # Update statistics
        all_data = list(latest_data)
        total = len(all_data)
        critical = sum(1 for d in all_data if d['status'] == "Critical")
        warning = sum(1 for d in all_data if d['status'] == "Warning")
        normal = sum(1 for d in all_data if d['status'] == "Normal")
        
        avg_hr = sum(d['heart_rate'] for d in all_data) / total if total > 0 else 0
        avg_sys = sum(d['systolic'] for d in all_data) / total if total > 0 else 0
        avg_dia = sum(d['diastolic'] for d in all_data) / total if total > 0 else 0
        
        statistics.update({
            "total_patients": len(set(d['patient_id'] for d in all_data)),
            "critical_count": critical,
            "warning_count": warning,
            "normal_count": normal,
            "avg_heart_rate": round(avg_hr, 1),
            "avg_systolic": round(avg_sys, 1),
            "avg_diastolic": round(avg_dia, 1),
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        
        # Generate alerts
        if status == "Critical" or data_point['is_anomaly']:
            alert = {
                "patient_name": patient_data['patient_name'],
                "room": patient_data['room'],
                "status": status,
                "risk_score": risk_score,
                "is_anomaly": data_point['is_anomaly'],
                "timestamp": patient_data['timestamp'],
                "message": f"{patient_data['patient_name']} - {status} (Risk: {risk_score})"
            }
            alerts.append(alert)
        
        print(f"{patient_data['patient_name']} | Status: {status} | Risk: {risk_score} | Anomaly: {data_point['is_anomaly']}")
        
    except Exception as e:
        print(f"Error processing data: {e}")

def socket_listener():
    """Listen to socket stream from data simulator"""
    print("\n" + "="*60)
    print("Connecting to data stream...")
    print("="*60)
    
    while True:
        try:
            # Connect to data simulator
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_socket.connect(("localhost", 9999))
            print("Connected to data simulator on localhost:9999")
            print("Receiving data stream...")
            print("="*60 + "\n")
            
            buffer = ""
            while True:
                data = client_socket.recv(4096).decode('utf-8')
                if not data:
                    break
                
                buffer += data
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    if line.strip():
                        process_data(line.strip())
            
        except ConnectionRefusedError:
            print("Waiting for data simulator to start...")
            time.sleep(2)
        except Exception as e:
            print(f"Connection error: {e}")
            time.sleep(2)
        finally:
            try:
                client_socket.close()
            except:
                pass

# Flask Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/latest_data')
def get_latest_data():
    return jsonify(list(latest_data))

@app.route('/api/statistics')
def get_statistics():
    return jsonify(statistics)

@app.route('/api/alerts')
def get_alerts():
    return jsonify(list(alerts))

@app.route('/api/ml_insights')
def get_ml_insights():
    """Return ML-derived summaries for frontend charts."""
    all_data = list(latest_data)

    # Status counts
    status_counts = {"Critical": 0, "Warning": 0, "Normal": 0}
    for d in all_data:
        st = d.get('status')
        if st in status_counts:
            status_counts[st] += 1

    # Anomaly counts overall and by status
    anomaly_count = sum(1 for d in all_data if d.get('is_anomaly'))
    anomalies_by_status = {"Critical": 0, "Warning": 0, "Normal": 0}
    for d in all_data:
        if d.get('is_anomaly'):
            st = d.get('status')
            if st in anomalies_by_status:
                anomalies_by_status[st] += 1

    # KMeans cluster sizes and descriptive summaries on recent points (if available)
    cluster_sizes = []
    cluster_info = []
    try:
        if len(all_data) >= 5:
            features = [[d['heart_rate'], d['systolic'], d['diastolic']] for d in all_data[-50:]]
            # normalize
            import numpy as _np
            arr = _np.array(features)
            m = arr.mean(axis=0)
            s = arr.std(axis=0)
            s[s == 0] = 1
            normalized = ((_np.array(features) - m) / s).tolist()
            k = min(4, max(2, len(normalized)//5))
            kmeans = SimpleKMeans(k=k)
            labels = kmeans.fit_predict(normalized)
            sizes = {i: labels.count(i) for i in set(labels)}
            ordered_idxs = sorted(sizes.keys())
            cluster_sizes = [sizes.get(i, 0) for i in ordered_idxs]

            # Map centroids back to original feature scale and create descriptive labels
            centroids = getattr(kmeans, 'centroids', []) or []
            for idx_i, i in enumerate(ordered_idxs):
                count = sizes.get(i, 0)
                centroid_norm = _np.array(centroids[idx_i]) if idx_i < len(centroids) else _np.array([0,0,0])
                centroid_orig = (centroid_norm * s) + m
                hr = float(round(float(centroid_orig[0]), 1))
                sysv = float(round(float(centroid_orig[1]), 1))
                dia = float(round(float(centroid_orig[2]), 1))

                # Simple descriptive rules (not a clinical diagnosis)
                if hr >= 120 or sysv >= 150 or dia >= 95:
                    label = "High-risk"
                elif hr >= 100 or sysv >= 140 or dia >= 90:
                    label = "Elevated"
                else:
                    label = "Normal-like"

                summary = f"{label} — HR {hr} / SYS {sysv} / DIA {dia} (n={count})"
                cluster_info.append({
                    "index": idx_i + 1,
                    "count": count,
                    "avg_heart_rate": hr,
                    "avg_systolic": sysv,
                    "avg_diastolic": dia,
                    "label": label,
                    "summary": summary
                })
    except Exception:
        cluster_sizes = []
        cluster_info = []


    return jsonify({
        "status_counts": status_counts,
        "anomaly_count": anomaly_count,
        "anomalies_by_status": anomalies_by_status,
        "cluster_sizes": cluster_sizes,
        "cluster_info": cluster_info,
        "avg_heart_rate": statistics.get('avg_heart_rate', 0),
        "avg_systolic": statistics.get('avg_systolic', 0),
        "avg_diastolic": statistics.get('avg_diastolic', 0)
    })

@app.route('/api/status')
def get_status():
    return jsonify({
        "connected": len(latest_data) > 0,
        "data_count": len(latest_data),
        "alert_count": len(alerts)
    })

if __name__ == '__main__':
    print("\n" + "="*60)
    print("HOSPITAL VITAL SIGNS MONITORING SYSTEM")
    print("="*60)
    print("\nSTARTUP INSTRUCTIONS:")
    print("   1. Make sure data_simulator.py is running first")
    print("   2. This will connect automatically")
    print("   3. Open browser: http://localhost:5000")
    print("\nML FEATURES:")
    print("   Risk Score Prediction")
    print("   Status Classification (Normal/Warning/Critical)")
    print("   Anomaly Detection (K-Means Clustering)")
    print("\n" + "="*60 + "\n")
    
    # Start socket listener in background
    listener_thread = threading.Thread(target=socket_listener, daemon=True)
    listener_thread.start()
    
    # Give it time to connect
    time.sleep(2)
    
    print("\n" + "="*60)
    print("Flask Web Server Started")
    print("="*60)
    print("Dashboard: http://localhost:5000")
    print("Auto-refresh: Every 1 second")
    print("="*60 + "\n")
    
    # Start Flask
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)