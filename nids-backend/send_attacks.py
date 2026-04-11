import requests
import pandas as pd
import numpy as np
import time

FILE_PATH = 'data/raw/cicids2017_cleaned.csv'
API_URL = 'http://localhost:8000/api/predict'


def run_batch(skip_start, batch_size=400000, per_type=5):
    print(f"\nLoading rows from {skip_start} to {skip_start + batch_size}...")

    df = pd.read_csv(
        FILE_PATH,
        skiprows=range(1, skip_start),
        nrows=batch_size
    )

    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    feature_cols = [c for c in df.columns if c != 'Attack Type']

    print("Available types:", df['Attack Type'].value_counts().to_dict())

    # Balanced sampling
    samples_list = []
    for attack_type, group in df.groupby('Attack Type'):
        if attack_type == 'Normal Traffic':
            sample_n = min(per_type * 2, len(group))
        else:
            sample_n = min(per_type, len(group))

        samples_list.append(group.sample(sample_n))

    samples = pd.concat(samples_list).sample(frac=1).reset_index(drop=True)

    print("Final mix:", samples['Attack Type'].value_counts().to_dict())

    for i, (_, row) in enumerate(samples.iterrows()):
        payload = row[feature_cols].to_dict()
        payload['_source_ip'] = f'10.0.0.{50 + i}'

        try:
            r = requests.post(API_URL, json=payload, timeout=5)

            if r.status_code == 200:
                data = r.json()
                print(f"[{i+1}] {data.get('prediction')} | conf: {data.get('confidence')} | sev: {data.get('severity')}")
            else:
                print(f"[{i+1}] Error: {r.status_code}")

        except Exception as e:
            print(f"[{i+1}] Failed: {e}")

        time.sleep(0.05)


if __name__ == "__main__":
    # 4 lakh chunks
    run_batch(0)
    run_batch(400000)
    run_batch(800000)
    run_batch(1200000)

# python send_attacks.py