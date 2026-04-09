import requests
import random
import time

API = 'http://localhost:8000/api/predict'

# These 52 features match EXACTLY what cicids2017_cleaned.csv has
# (same column names the model was trained on)
FEATURE_NAMES = [
    'Destination Port', 'Flow Duration', 'Total Fwd Packets',
    'Total Length of Fwd Packets', 'Fwd Packet Length Max',
    'Fwd Packet Length Min', 'Fwd Packet Length Mean', 'Fwd Packet Length Std',
    'Bwd Packet Length Max', 'Bwd Packet Length Min', 'Bwd Packet Length Mean',
    'Bwd Packet Length Std', 'Flow Bytes/s', 'Flow Packets/s',
    'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min',
    'Fwd IAT Total', 'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min',
    'Bwd IAT Total', 'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min',
    'Fwd Header Length', 'Bwd Header Length', 'Fwd Packets/s', 'Bwd Packets/s',
    'Min Packet Length', 'Max Packet Length', 'Packet Length Mean',
    'Packet Length Std', 'Packet Length Variance', 'FIN Flag Count',
    'PSH Flag Count', 'ACK Flag Count', 'Average Packet Size',
    'Subflow Fwd Bytes', 'Init_Win_bytes_forward', 'Init_Win_bytes_backward',
    'act_data_pkt_fwd', 'min_seg_size_forward', 'Active Mean', 'Active Max',
    'Active Min', 'Idle Mean', 'Idle Max', 'Idle Min'
]

print('Sending 20 fake flows to API...')
print()

for i in range(20):
    # Build feature dict with random values
    data = {feat: round(random.uniform(0, 1000), 2) for feat in FEATURE_NAMES}

    try:
        r = requests.post(API, json=data, timeout=5)
        if r.status_code == 200:
            resp = r.json()
            pred = resp.get('prediction', '?')
            sev  = resp.get('severity', '?')
            conf = resp.get('confidence', 0)
            print(f'  [{i+1}/20]  {pred:<20}  Severity: {sev:<10}  Confidence: {conf*100:.1f}%')
        else:
            print(f'  [{i+1}/20]  Error {r.status_code}: {r.text[:80]}')
    except Exception as e:
        print(f'  [{i+1}/20]  Connection error: {e}')

    time.sleep(0.3)

print()
print('Done! Check dashboard now.')