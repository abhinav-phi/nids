"""Test the full extraction → prediction pipeline."""
from src.features.extractor import FlowExtractor
from src.model.predict import predict
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
result = predict(features, feature_names=list(features.keys()))
print("=== END-TO-END PIPELINE TEST ===")
print("Prediction: ", result["prediction"])
print("Confidence: ", f'{result["confidence"]*100:.1f}%')
print("Severity:   ", result["severity"])
print("SHAP top 5: ", result["shap_top5"])
print("================================")
ddos_packets = []
for i in range(100):
    ddos_packets.append({
        "src_ip": f"10.{i%256}.{(i*7)%256}.{(i*13)%256}",
        "dst_ip": "192.168.1.10",
        "src_port": 1024 + (i * 17) % 60000,
        "dst_port": 80,
        "protocol": "UDP",
        "size": 64,
        "payload_len": 32,
        "header_len": 28,
        "time": 2000.0 + i * 0.001,  
        "tcp_flags": "",
        "window_size": 0,
        "ttl": 64,
    })
ddos_features = e.extract_from_dicts(ddos_packets, flow_key=("10.0.0.0", "192.168.1.10", 1024, 80, "UDP"))
ddos_result = predict(ddos_features, feature_names=list(ddos_features.keys()))
print("\n=== DDoS-LIKE FLOW TEST ===")
print("Prediction: ", ddos_result["prediction"])
print("Confidence: ", f'{ddos_result["confidence"]*100:.1f}%')
print("Severity:   ", ddos_result["severity"])
print("===========================")
print("\nPIPELINE TEST COMPLETE")
