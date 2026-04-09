"""Test the predict API endpoint with real extracted features."""
import requests
import json
from src.features.extractor import FlowExtractor
packets = [
    {"src_ip": "192.168.1.10", "dst_ip": "93.184.216.34", "src_port": 54321, "dst_port": 443, "protocol": "TCP", "size": 74, "payload_len": 0, "header_len": 40, "time": 1000.0, "tcp_flags": "S", "window_size": 65535, "ttl": 128},
    {"src_ip": "93.184.216.34", "dst_ip": "192.168.1.10", "src_port": 443, "dst_port": 54321, "protocol": "TCP", "size": 74, "payload_len": 0, "header_len": 40, "time": 1000.05, "tcp_flags": "SA", "window_size": 65535, "ttl": 52},
    {"src_ip": "192.168.1.10", "dst_ip": "93.184.216.34", "src_port": 54321, "dst_port": 443, "protocol": "TCP", "size": 66, "payload_len": 0, "header_len": 32, "time": 1000.06, "tcp_flags": "A", "window_size": 65535, "ttl": 128},
    {"src_ip": "192.168.1.10", "dst_ip": "93.184.216.34", "src_port": 54321, "dst_port": 443, "protocol": "TCP", "size": 500, "payload_len": 434, "header_len": 32, "time": 1000.1, "tcp_flags": "PA", "window_size": 65535, "ttl": 128},
    {"src_ip": "93.184.216.34", "dst_ip": "192.168.1.10", "src_port": 443, "dst_port": 54321, "protocol": "TCP", "size": 1500, "payload_len": 1434, "header_len": 32, "time": 1000.15, "tcp_flags": "PA", "window_size": 65535, "ttl": 52},
    {"src_ip": "192.168.1.10", "dst_ip": "93.184.216.34", "src_port": 54321, "dst_port": 443, "protocol": "TCP", "size": 66, "payload_len": 0, "header_len": 32, "time": 1000.16, "tcp_flags": "A", "window_size": 65535, "ttl": 128},
]
e = FlowExtractor()
key = ("192.168.1.10", "93.184.216.34", 54321, 443, "TCP")
features = e.extract_from_dicts(packets, flow_key=key)
features["_source_ip"] = "192.168.1.10"
features["_destination_ip"] = "93.184.216.34"
features["_src_port"] = 54321
features["_dst_port"] = 443
print("Sending feature vector to API...")
print(f"Feature count (excluding metadata): {len([k for k in features if not k.startswith('_')])}")
r = requests.post("http://localhost:8000/api/predict", json=features, timeout=10)
print(f"\nHTTP Status: {r.status_code}")
print(f"Response:\n{json.dumps(r.json(), indent=2)}")
print("\n--- Port Scan simulation ---")
scan_packets = []
for i in range(50):
    scan_packets.append({
        "src_ip": "10.0.0.5", "dst_ip": "192.168.1.100",
        "src_port": 40000 + i, "dst_port": 21 + i,
        "protocol": "TCP", "size": 54, "payload_len": 0,
        "header_len": 40, "time": 3000.0 + i * 0.01,
        "tcp_flags": "S", "window_size": 1024, "ttl": 64,
    })
scan_features = e.extract_from_dicts(scan_packets,
    flow_key=("10.0.0.5", "192.168.1.100", 40000, 21, "TCP"))
scan_features["_source_ip"] = "10.0.0.5"
scan_features["_destination_ip"] = "192.168.1.100"
scan_features["_src_port"] = 40000
scan_features["_dst_port"] = 21
r2 = requests.post("http://localhost:8000/api/predict", json=scan_features, timeout=10)
print(f"HTTP Status: {r2.status_code}")
print(f"Response:\n{json.dumps(r2.json(), indent=2)}")
print("\n--- Stats ---")
r3 = requests.get("http://localhost:8000/api/stats")
print(json.dumps(r3.json(), indent=2))
print("\nAPI TEST COMPLETE")
