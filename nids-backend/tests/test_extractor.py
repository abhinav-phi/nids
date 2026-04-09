"""
test_extractor.py — Unit tests for FlowExtractor
Run with:  pytest tests/test_extractor.py -v
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import time
import pytest
from src.features.extractor import FlowExtractor
@pytest.fixture
def extractor():
    return FlowExtractor()
@pytest.fixture
def normal_flow():
    """Simulate a normal HTTPS browsing flow."""
    now = time.time()
    return [
        {"src_ip": "192.168.1.10", "size": 60,   "time": now + 0.000, "tcp_flags": "S"},
        {"src_ip": "10.0.0.1",     "size": 60,   "time": now + 0.001, "tcp_flags": "SA"},
        {"src_ip": "192.168.1.10", "size": 52,   "time": now + 0.002, "tcp_flags": "A"},
        {"src_ip": "192.168.1.10", "size": 512,  "time": now + 0.003, "tcp_flags": "PA"},
        {"src_ip": "10.0.0.1",     "size": 1460, "time": now + 0.050, "tcp_flags": "PA"},
        {"src_ip": "10.0.0.1",     "size": 1460, "time": now + 0.051, "tcp_flags": "PA"},
        {"src_ip": "192.168.1.10", "size": 52,   "time": now + 0.052, "tcp_flags": "A"},
        {"src_ip": "192.168.1.10", "size": 52,   "time": now + 0.060, "tcp_flags": "F"},
        {"src_ip": "10.0.0.1",     "size": 52,   "time": now + 0.061, "tcp_flags": "FA"},
    ]
@pytest.fixture
def ddos_flow():
    """Simulate a DDoS-like flow: many tiny packets from one source."""
    now = time.time()
    return [
        {"src_ip": "1.2.3.4", "size": 64, "time": now + i*0.0001, "tcp_flags": "S"}
        for i in range(200)
    ]
@pytest.fixture
def portscan_flow():
    """Simulate a port scan: SYN to many ports, no responses."""
    now = time.time()
    return [
        {"src_ip": "5.5.5.5", "size": 60, "time": now + i*0.01, "tcp_flags": "S"}
        for i in range(30)
    ]
def test_returns_dict(extractor, normal_flow):
    features = extractor.extract_from_dicts(normal_flow)
    assert isinstance(features, dict)
def test_all_required_features_present(extractor, normal_flow):
    required = [
        "flow_duration", "total_fwd_packets", "total_bwd_packets",
        "total_fwd_bytes", "total_bwd_bytes",
        "packet_length_mean", "packet_length_max",
        "packet_length_min", "packet_length_std",
        "packets_per_second", "bytes_per_second",
        "syn_flag_count", "fin_flag_count",
        "rst_flag_count", "ack_flag_count",
    ]
    features = extractor.extract_from_dicts(normal_flow)
    for key in required:
        assert key in features, f"Missing feature: {key}"
def test_all_values_are_float(extractor, normal_flow):
    features = extractor.extract_from_dicts(normal_flow)
    for k, v in features.items():
        assert isinstance(v, float), f"{k} is not a float: {type(v)}"
def test_no_nan_or_inf(extractor, normal_flow):
    import math
    features = extractor.extract_from_dicts(normal_flow)
    for k, v in features.items():
        assert not math.isnan(v),  f"{k} is NaN"
        assert not math.isinf(v),  f"{k} is Inf"
def test_empty_packets_returns_zeros(extractor):
    features = extractor.extract_from_dicts([])
    assert features["flow_duration"] == 0.0
    assert features["total_fwd_packets"] == 0.0
def test_fwd_bwd_packet_split(extractor, normal_flow):
    features = extractor.extract_from_dicts(normal_flow)
    assert features["total_fwd_packets"] == 5.0
    assert features["total_bwd_packets"] == 4.0
def test_ddos_has_high_pps(extractor, ddos_flow):
    features = extractor.extract_from_dicts(ddos_flow)
    assert features["packets_per_second"] > 1000
def test_ddos_small_packets(extractor, ddos_flow):
    features = extractor.extract_from_dicts(ddos_flow)
    assert features["packet_length_mean"] < 100     
def test_portscan_has_many_syn(extractor, portscan_flow):
    features = extractor.extract_from_dicts(portscan_flow)
    assert features["syn_flag_count"] == 30.0       
    assert features["fin_flag_count"] == 0.0        
def test_normal_flow_has_fin(extractor, normal_flow):
    features = extractor.extract_from_dicts(normal_flow)
    assert features["fin_flag_count"] > 0           
def test_single_packet_no_crash(extractor):
    single = [{"src_ip": "1.1.1.1", "size": 100, "time": time.time(), "tcp_flags": "S"}]
    features = extractor.extract_from_dicts(single)
    assert features["total_fwd_packets"] == 1.0
    assert features["flow_iat_mean"] == 0.0         
