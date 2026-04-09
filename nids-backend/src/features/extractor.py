"""
extractor.py — CICIDS2017-Compatible Flow Feature Extractor
=============================================================
Converts raw packet dicts (from the sniffer) into the exact 52-feature
vector that the trained ML model expects.

The packet dicts arrive grouped by flow (5-tuple key).
This module computes statistical, temporal, and flag-based features
that match the CICIDS2017 dataset schema EXACTLY.

Usage:
    extractor = FlowExtractor()
    features = extractor.extract_from_dicts(packet_list, flow_key)
    # features is a dict with 52 keys matching CICIDS2017 column names
"""

import math
import logging
from typing import Dict, List, Tuple, Optional

log = logging.getLogger(__name__)


# ── All 52 CICIDS2017 features the model expects (in order) ──────────────────
CICIDS_FEATURES = [
    'Destination Port',
    'Flow Duration',
    'Total Fwd Packets',
    'Total Length of Fwd Packets',
    'Fwd Packet Length Max',
    'Fwd Packet Length Min',
    'Fwd Packet Length Mean',
    'Fwd Packet Length Std',
    'Bwd Packet Length Max',
    'Bwd Packet Length Min',
    'Bwd Packet Length Mean',
    'Bwd Packet Length Std',
    'Flow Bytes/s',
    'Flow Packets/s',
    'Flow IAT Mean',
    'Flow IAT Std',
    'Flow IAT Max',
    'Flow IAT Min',
    'Fwd IAT Total',
    'Fwd IAT Mean',
    'Fwd IAT Std',
    'Fwd IAT Max',
    'Fwd IAT Min',
    'Bwd IAT Total',
    'Bwd IAT Mean',
    'Bwd IAT Std',
    'Bwd IAT Max',
    'Bwd IAT Min',
    'Fwd Header Length',
    'Bwd Header Length',
    'Fwd Packets/s',
    'Bwd Packets/s',
    'Min Packet Length',
    'Max Packet Length',
    'Packet Length Mean',
    'Packet Length Std',
    'Packet Length Variance',
    'FIN Flag Count',
    'PSH Flag Count',
    'ACK Flag Count',
    'Average Packet Size',
    'Subflow Fwd Bytes',
    'Init_Win_bytes_forward',
    'Init_Win_bytes_backward',
    'act_data_pkt_fwd',
    'min_seg_size_forward',
    'Active Mean',
    'Active Max',
    'Active Min',
    'Idle Mean',
    'Idle Max',
    'Idle Min',
]


# ── Helper functions ──────────────────────────────────────────────────────────

def _safe_mean(lst: list) -> float:
    """Mean of a list, returns 0.0 for empty lists."""
    return sum(lst) / len(lst) if lst else 0.0


def _safe_std(lst: list) -> float:
    """Population standard deviation, returns 0.0 for <2 elements."""
    if len(lst) < 2:
        return 0.0
    m = sum(lst) / len(lst)
    variance = sum((x - m) ** 2 for x in lst) / len(lst)
    return math.sqrt(variance)


def _safe_var(lst: list) -> float:
    """Population variance, returns 0.0 for <2 elements."""
    if len(lst) < 2:
        return 0.0
    m = sum(lst) / len(lst)
    return sum((x - m) ** 2 for x in lst) / len(lst)


def _safe_min(lst: list) -> float:
    return min(lst) if lst else 0.0


def _safe_max(lst: list) -> float:
    return max(lst) if lst else 0.0


def _compute_iats(timestamps: list) -> list:
    """Compute inter-arrival times from sorted timestamps."""
    if len(timestamps) < 2:
        return []
    sorted_ts = sorted(timestamps)
    return [sorted_ts[i+1] - sorted_ts[i] for i in range(len(sorted_ts) - 1)]


def _count_flag(packets: list, flag_char: str) -> int:
    """Count how many packets have a specific TCP flag set."""
    count = 0
    for p in packets:
        flags = p.get("tcp_flags", "")
        if flag_char in flags:
            count += 1
    return count


# ── Active / Idle period detection ────────────────────────────────────────────

ACTIVE_TIMEOUT = 5.0  # seconds — threshold between active and idle


def _compute_active_idle(timestamps: list) -> Tuple[list, list]:
    """
    Classify time gaps between packets as active or idle periods.
    - Active: gap < ACTIVE_TIMEOUT seconds
    - Idle:   gap >= ACTIVE_TIMEOUT seconds

    Returns (active_durations, idle_durations) in microseconds.
    """
    if len(timestamps) < 2:
        return [], []

    sorted_ts = sorted(timestamps)
    active_periods = []
    idle_periods = []

    current_active_start = sorted_ts[0]
    current_active_end = sorted_ts[0]

    for i in range(1, len(sorted_ts)):
        gap = sorted_ts[i] - sorted_ts[i - 1]
        if gap < ACTIVE_TIMEOUT:
            # Still in active period
            current_active_end = sorted_ts[i]
        else:
            # End active period, record idle
            active_dur = (current_active_end - current_active_start) * 1e6
            if active_dur > 0:
                active_periods.append(active_dur)
            idle_periods.append(gap * 1e6)
            current_active_start = sorted_ts[i]
            current_active_end = sorted_ts[i]

    # Final active period
    final_active = (current_active_end - current_active_start) * 1e6
    if final_active > 0:
        active_periods.append(final_active)

    return active_periods, idle_periods


# ── Main FlowExtractor class ─────────────────────────────────────────────────

class FlowExtractor:
    """
    Extracts the 52 CICIDS2017-compatible features from a list of
    packet dictionaries that belong to a single network flow.

    Each packet dict must contain (at minimum):
        src_ip      : str   - source IP
        dst_ip      : str   - destination IP
        src_port    : int   - source port
        dst_port    : int   - destination port
        protocol    : str   - "TCP", "UDP", "ICMP", etc.
        size        : int   - total packet size (bytes)
        payload_len : int   - payload size (bytes, excluding headers)
        header_len  : int   - header size (bytes)
        time        : float - packet timestamp (epoch seconds)
        tcp_flags   : str   - TCP flags string (e.g. "SA", "PA", "F")
        window_size : int   - TCP window size (0 for non-TCP)

    Optional:
        ttl         : int   - time to live

    The first packet in the list defines the "forward" direction.
    All packets from the same src_ip are forward; others are backward.
    """

    def __init__(self):
        self.feature_names = CICIDS_FEATURES

    def extract_from_dicts(
        self,
        packets: List[dict],
        flow_key: Optional[Tuple] = None,
    ) -> Dict[str, float]:
        """
        Extract all 52 CICIDS2017 features from a list of packet dicts.

        Parameters
        ----------
        packets : list of dict
            Packets belonging to one flow (all same 5-tuple).
        flow_key : tuple, optional
            (src_ip, dst_ip, src_port, dst_port, protocol)
            If provided, used to determine forward direction.
            If not provided, first packet's src_ip defines forward.

        Returns
        -------
        dict : 52 features keyed by CICIDS2017 column names
        """
        if not packets:
            # Return all zeros for empty flows
            return {name: 0.0 for name in CICIDS_FEATURES}

        # ── Determine forward direction ───────────────────────────────────
        if flow_key:
            fwd_src_ip = flow_key[0]
            dst_port = flow_key[3]
        else:
            fwd_src_ip = packets[0].get("src_ip", "")
            dst_port = packets[0].get("dst_port", 0)

        # ── Split packets into forward and backward ───────────────────────
        fwd_packets = []
        bwd_packets = []
        all_timestamps = []
        all_sizes = []

        for p in packets:
            ts = float(p.get("time", 0))
            size = int(p.get("size", 0))
            all_timestamps.append(ts)
            all_sizes.append(size)

            if p.get("src_ip", "") == fwd_src_ip:
                fwd_packets.append(p)
            else:
                bwd_packets.append(p)

        # ── Basic counts ──────────────────────────────────────────────────
        total_packets = len(packets)
        total_fwd = len(fwd_packets)
        total_bwd = len(bwd_packets)

        # ── Sizes ─────────────────────────────────────────────────────────
        fwd_sizes = [int(p.get("size", 0)) for p in fwd_packets]
        bwd_sizes = [int(p.get("size", 0)) for p in bwd_packets]
        fwd_payload_sizes = [int(p.get("payload_len", 0)) for p in fwd_packets]
        bwd_payload_sizes = [int(p.get("payload_len", 0)) for p in bwd_packets]

        total_fwd_bytes = sum(fwd_sizes)
        total_bwd_bytes = sum(bwd_sizes)
        total_bytes = total_fwd_bytes + total_bwd_bytes

        # ── Duration (microseconds, matching CICIDS2017) ──────────────────
        if len(all_timestamps) >= 2:
            flow_duration_sec = max(all_timestamps) - min(all_timestamps)
        else:
            flow_duration_sec = 0.0
        flow_duration_us = flow_duration_sec * 1e6  # CICIDS uses microseconds

        # ── Timestamps for fwd and bwd ────────────────────────────────────
        fwd_timestamps = [float(p.get("time", 0)) for p in fwd_packets]
        bwd_timestamps = [float(p.get("time", 0)) for p in bwd_packets]

        # ── Inter-Arrival Times (in microseconds) ─────────────────────────
        flow_iats = [x * 1e6 for x in _compute_iats(all_timestamps)]
        fwd_iats = [x * 1e6 for x in _compute_iats(fwd_timestamps)]
        bwd_iats = [x * 1e6 for x in _compute_iats(bwd_timestamps)]

        # ── Rates ─────────────────────────────────────────────────────────
        duration_safe = flow_duration_sec if flow_duration_sec > 0 else 1e-6
        flow_bytes_per_s = total_bytes / duration_safe
        flow_packets_per_s = total_packets / duration_safe
        fwd_packets_per_s = total_fwd / duration_safe
        bwd_packets_per_s = total_bwd / duration_safe

        # ── Header lengths ────────────────────────────────────────────────
        fwd_header_lengths = [int(p.get("header_len", 20)) for p in fwd_packets]
        bwd_header_lengths = [int(p.get("header_len", 20)) for p in bwd_packets]
        fwd_header_len_total = sum(fwd_header_lengths)
        bwd_header_len_total = sum(bwd_header_lengths)

        # ── TCP flags ─────────────────────────────────────────────────────
        fin_count = _count_flag(packets, "F")
        psh_count = _count_flag(packets, "P")
        ack_count = _count_flag(packets, "A")

        # ── Window sizes ──────────────────────────────────────────────────
        fwd_windows = [int(p.get("window_size", 0)) for p in fwd_packets]
        bwd_windows = [int(p.get("window_size", 0)) for p in bwd_packets]
        init_win_fwd = fwd_windows[0] if fwd_windows else 0
        init_win_bwd = bwd_windows[0] if bwd_windows else 0

        # ── act_data_pkt_fwd: fwd packets with payload > 0 ───────────────
        act_data_pkt_fwd = sum(1 for p in fwd_packets if int(p.get("payload_len", 0)) > 0)

        # ── min_seg_size_forward: minimum header length in fwd direction ──
        min_seg_fwd = min(fwd_header_lengths) if fwd_header_lengths else 0

        # ── Active / Idle periods ─────────────────────────────────────────
        active_periods, idle_periods = _compute_active_idle(all_timestamps)

        # ── Average Packet Size ───────────────────────────────────────────
        avg_pkt_size = total_bytes / total_packets if total_packets > 0 else 0.0

        # ── Build the feature dictionary (EXACT CICIDS2017 order) ─────────
        features = {
            'Destination Port':             float(dst_port),
            'Flow Duration':                flow_duration_us,
            'Total Fwd Packets':            float(total_fwd),
            'Total Length of Fwd Packets':  float(total_fwd_bytes),
            'Fwd Packet Length Max':         _safe_max(fwd_sizes),
            'Fwd Packet Length Min':         _safe_min(fwd_sizes),
            'Fwd Packet Length Mean':        _safe_mean(fwd_sizes),
            'Fwd Packet Length Std':         _safe_std(fwd_sizes),
            'Bwd Packet Length Max':         _safe_max(bwd_sizes),
            'Bwd Packet Length Min':         _safe_min(bwd_sizes),
            'Bwd Packet Length Mean':        _safe_mean(bwd_sizes),
            'Bwd Packet Length Std':         _safe_std(bwd_sizes),
            'Flow Bytes/s':                 flow_bytes_per_s,
            'Flow Packets/s':               flow_packets_per_s,
            'Flow IAT Mean':                _safe_mean(flow_iats),
            'Flow IAT Std':                 _safe_std(flow_iats),
            'Flow IAT Max':                 _safe_max(flow_iats),
            'Flow IAT Min':                 _safe_min(flow_iats),
            'Fwd IAT Total':                sum(fwd_iats),
            'Fwd IAT Mean':                 _safe_mean(fwd_iats),
            'Fwd IAT Std':                  _safe_std(fwd_iats),
            'Fwd IAT Max':                  _safe_max(fwd_iats),
            'Fwd IAT Min':                  _safe_min(fwd_iats),
            'Bwd IAT Total':                sum(bwd_iats),
            'Bwd IAT Mean':                 _safe_mean(bwd_iats),
            'Bwd IAT Std':                  _safe_std(bwd_iats),
            'Bwd IAT Max':                  _safe_max(bwd_iats),
            'Bwd IAT Min':                  _safe_min(bwd_iats),
            'Fwd Header Length':            float(fwd_header_len_total),
            'Bwd Header Length':            float(bwd_header_len_total),
            'Fwd Packets/s':                fwd_packets_per_s,
            'Bwd Packets/s':                bwd_packets_per_s,
            'Min Packet Length':            _safe_min(all_sizes),
            'Max Packet Length':            _safe_max(all_sizes),
            'Packet Length Mean':           _safe_mean(all_sizes),
            'Packet Length Std':            _safe_std(all_sizes),
            'Packet Length Variance':       _safe_var(all_sizes),
            'FIN Flag Count':               float(fin_count),
            'PSH Flag Count':               float(psh_count),
            'ACK Flag Count':               float(ack_count),
            'Average Packet Size':          avg_pkt_size,
            'Subflow Fwd Bytes':            float(total_fwd_bytes),
            'Init_Win_bytes_forward':       float(init_win_fwd),
            'Init_Win_bytes_backward':      float(init_win_bwd),
            'act_data_pkt_fwd':             float(act_data_pkt_fwd),
            'min_seg_size_forward':         float(min_seg_fwd),
            'Active Mean':                  _safe_mean(active_periods),
            'Active Max':                   _safe_max(active_periods),
            'Active Min':                   _safe_min(active_periods),
            'Idle Mean':                    _safe_mean(idle_periods),
            'Idle Max':                     _safe_max(idle_periods),
            'Idle Min':                     _safe_min(idle_periods),
        }

        # ── Sanity: replace NaN / Inf with 0 ─────────────────────────────
        for k, v in features.items():
            if math.isnan(v) or math.isinf(v):
                features[k] = 0.0

        return features

    def get_feature_names(self) -> list:
        """Return the ordered list of feature names."""
        return list(CICIDS_FEATURES)
