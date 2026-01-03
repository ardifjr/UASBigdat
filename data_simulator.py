import socket
import time
import random
import json
from datetime import datetime

HOST = "localhost"
PORT = 9999

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((HOST, PORT))
server_socket.listen(1)

print("Waiting for Spark Streaming to connect...")
conn, addr = server_socket.accept()
print("Connected by", addr)

patients = [
    {"patient_id": "P001", "name": "Andi Pratama", "room": "ICU - Lantai 2, Kamar 201"},
    {"patient_id": "P002", "name": "Siti Aisyah", "room": "ICU - Lantai 2, Kamar 202"},
    {"patient_id": "P003", "name": "Budi Santoso", "room": "HCU - Lantai 3, Kamar 305"},
    {"patient_id": "P004", "name": "Rina Wulandari", "room": "HCU - Lantai 3, Kamar 306"},
    {"patient_id": "P005", "name": "Ahmad Fauzi", "room": "IGD - Lantai 1, Bed A1"},
    {"patient_id": "P006", "name": "Dewi Lestari", "room": "IGD - Lantai 1, Bed A2"},
    {"patient_id": "P007", "name": "Rizky Maulana", "room": "Rawat Inap - Lantai 4, Kamar 401A"},
    {"patient_id": "P008", "name": "Nur Halimah", "room": "Rawat Inap - Lantai 4, Kamar 401B"},
    {"patient_id": "P009", "name": "Eko Saputra", "room": "ICU - Lantai 2, Kamar 203"},
    {"patient_id": "P010", "name": "Maya Indah", "room": "VIP - Lantai 5, Kamar 501"},
    {"patient_id": "P011", "name": "Guntur Wijaya", "room": "Rawat Inap - Lantai 4, Kamar 402"},
    {"patient_id": "P012", "name": "Sari Kartika", "room": "HCU - Lantai 3, Kamar 307"},
    {"patient_id": "P013", "name": "Hendra Setiawan", "room": "IGD - Lantai 1, Bed B1"},
    {"patient_id": "P014", "name": "Ani Wijayanti", "room": "Isolasi - Lantai 2, Kamar 210"},
    {"patient_id": "P015", "name": "Fajar Nugraha", "room": "Rawat Inap - Lantai 4, Kamar 403"},
    {"patient_id": "P016", "name": "Lestari Putri", "room": "VIP - Lantai 5, Kamar 502"},
    {"patient_id": "P017", "name": "Bambang Heru", "room": "ICU - Lantai 2, Kamar 204"},
    {"patient_id": "P018", "name": "Yulia Rosa", "room": "HCU - Lantai 3, Kamar 308"},
    {"patient_id": "P019", "name": "Deni Faisal", "room": "IGD - Lantai 1, Bed B2"},
    {"patient_id": "P020", "name": "Ratna Sari", "room": "Rawat Inap - Lantai 4, Kamar 404A"},
    {"patient_id": "P021", "name": "Taufik Hidayat", "room": "Isolasi - Lantai 2, Kamar 211"},
    {"patient_id": "P022", "name": "Mega Utami", "room": "VIP - Lantai 5, Kamar 503"},
    {"patient_id": "P023", "name": "Aris Munandar", "room": "Rawat Inap - Lantai 4, Kamar 404B"},
    {"patient_id": "P024", "name": "Fitriani", "room": "HCU - Lantai 3, Kamar 309"},
    {"patient_id": "P025", "name": "Zulham Efendi", "room": "IGD - Lantai 1, Bed C1"},
    {"patient_id": "P026", "name": "Nanda Kurnia", "room": "ICU - Lantai 2, Kamar 205"},
    {"patient_id": "P027", "name": "Agus Salim", "room": "Rawat Inap - Lantai 4, Kamar 405"},
    {"patient_id": "P028", "name": "Indah Permata", "room": "VIP - Lantai 5, Kamar 504"},
    {"patient_id": "P029", "name": "Yanto Subagio", "room": "HCU - Lantai 3, Kamar 310"},
    {"patient_id": "P030", "name": "Diana Putri", "room": "IGD - Lantai 1, Bed C2"},
]

while True:
    patient = random.choice(patients)

    data = {
        "patient_id": patient["patient_id"],
        "patient_name": patient["name"],
        "room": patient["room"],
        "heart_rate": random.randint(55, 145),
        "systolic": random.randint(90, 170),
        "diastolic": random.randint(60, 105),
        "timestamp": datetime.now().isoformat()
    }

    message = json.dumps(data)
    conn.sendall((message + "\n").encode("utf-8"))
    
    time.sleep(1)