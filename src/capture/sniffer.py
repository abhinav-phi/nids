"""
sniffer.py — Live Network Packet Capture
==========================================
Captures packets from a network interface using Scapy.
Groups packets into flows using a 5-tuple key.
When a flow ends (timeout or TCP FIN/RST), extracts features
using FlowExtractor and sends them to the FastAPI backend.

Run standalone (for testing):
    sudo python src/capture/sniffer.py --interface eth0
    sudo python src/capture/sniffer.py --interface lo  # loopback for demo
"""

import time
import logging
import threading
import argparse
import requests
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# Scapy imports
try:
    from scapy.all import sniff, IP, TCP, UDP, ICMP, conf
    SCAPY_AVAILABLE = True
except ImportError:
    SCAPY_AVAILABLE = False
    print("[WARNING] Scapy not installed. Sniffer will not capture live packets.")

# FlowExtractor from Step 3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.features.extractor import FlowExtractor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [SNIFFER]  %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

FLOW_TIMEOUT_SECONDS = 60       # close a flow if no new packets for 60s
MAX_PACKETS_PER_FLOW = 1000     # safety cap — very long flows get cut here
API_PREDICT_URL      = "http://localhost:8000/api/predict"


# ── Flow key type alias ───────────────────────────────────────────────────────
# A flow is uniquely identified by these 5 values.
# Type: (src_ip, dst_ip, src_port, dst_port, protocol)
FlowKey = Tuple[str, str, int, int, str]


@dataclass
class Flow:
    """
    Holds all packets belonging to one network flow,
    along with metadata for timeout tracking.
    """
    key: FlowKey
    packets: List         = field(default_factory=list)  # raw Scapy packets
    packet_dicts: List    = field(default_factory=list)  # plain dicts for extractor
    start_time: float     = field(default_factory=time.time)
    last_seen: float      = field(default_factory=time.time)
    is_closed: bool       = False


class NetworkSniffer:
    """
    Captures live network packets, assembles them into flows,
    and submits completed flows to the ML prediction API.

    Usage:
        sniffer = NetworkSniffer(interface="eth0")
        sniffer.start()       # starts capture in background thread
        ...
        sniffer.stop()
    """

    def __init__(
        self,
        interface: str = "lo",
        api_url: str = API_PREDICT_URL,
        flow_timeout: int = FLOW_TIMEOUT_SECONDS,
    ):
        self.interface   = interface
        self.api_url     = api_url
        self.flow_timeout = flow_timeout

        # Active flows: FlowKey → Flow object
        self._flows: Dict[FlowKey, Flow] = {}
        self._flows_lock = threading.Lock()

        # Feature extractor (from Step 3)
        self._extractor = FlowExtractor()

        # Stats
        self.total_packets   = 0
        self.total_flows     = 0
        self.total_api_calls = 0

        # Control flag
        self._running = False
        self._capture_thread: Optional[threading.Thread] = None
        self._timeout_thread: Optional[threading.Thread] = None

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def start(self):
        """Start packet capture and flow timeout checker in background threads."""
        if not SCAPY_AVAILABLE:
            log.error("Scapy is not installed. Cannot start sniffer.")
            return

        self._running = True
        log.info(f"Starting capture on interface '{self.interface}' ...")
        log.info(f"Sending predictions to: {self.api_url}")

        # Thread 1: packet capture (blocking — runs inside its own thread)
        self._capture_thread = threading.Thread(
            target=self._capture_loop,
            daemon=True,
            name="PacketCapture"
        )
        self._capture_thread.start()

        # Thread 2: flow timeout checker (runs every 5 seconds)
        self._timeout_thread = threading.Thread(
            target=self._timeout_loop,
            daemon=True,
            name="FlowTimeout"
        )
        self._timeout_thread.start()

        log.info("Sniffer running. Press Ctrl+C to stop.")

    def stop(self):
        """Signal all threads to stop gracefully."""
        self._running = False
        log.info(
            f"Sniffer stopped. "
            f"Packets: {self.total_packets} | "
            f"Flows: {self.total_flows} | "
            f"API calls: {self.total_api_calls}"
        )

    def is_running(self) -> bool:
        return self._running

    # ─────────────────────────────────────────────────────────────────────────
    # Thread 1 — Packet capture
    # ─────────────────────────────────────────────────────────────────────────

    def _capture_loop(self):
        """
        Runs Scapy's sniff() continuously.
        Each captured packet is passed to _process_packet().
        'store=False' prevents Scapy from keeping packets in memory.
        """
        try:
            sniff(
                iface=self.interface,
                prn=self._process_packet,   # callback for each packet
                store=False,                 # don't buffer packets in RAM
                stop_filter=lambda _: not self._running  # stop when flag is cleared
            )
        except PermissionError:
            log.error(
                "Permission denied. Run with sudo: sudo python src/capture/sniffer.py"
            )
            self._running = False
        except Exception as e:
            log.error(f"Capture error: {e}")
            self._running = False

    def _process_packet(self, pkt):
        """
        Called for every captured packet.
        Extracts the 5-tuple key and adds the packet to the correct flow.
        """
        if not SCAPY_AVAILABLE:
            return

        self.total_packets += 1

        # Only process IP packets
        if not pkt.haslayer(IP):
            return

        key = self._get_flow_key(pkt)
        if key is None:
            return

        packet_dict = self._packet_to_dict(pkt)

        with self._flows_lock:
            if key not in self._flows:
                # New flow — create a Flow object
                self._flows[key] = Flow(key=key)
                log.debug(f"New flow: {key[0]}:{key[2]} → {key[1]}:{key[3]} [{key[4]}]")

            flow = self._flows[key]
            flow.packets.append(pkt)
            flow.packet_dicts.append(packet_dict)
            flow.last_seen = time.time()

            # Check if this packet closes the flow (TCP FIN or RST)
            if self._is_flow_terminator(pkt):
                flow.is_closed = True

        # If flow is now closed, process it immediately
        if self._flows.get(key) and self._flows[key].is_closed:
            self._finalize_flow(key)

        # Safety cap — if flow is very large, process it now
        if key in self._flows and len(self._flows[key].packets) >= MAX_PACKETS_PER_FLOW:
            log.debug(f"Flow hit max packet cap ({MAX_PACKETS_PER_FLOW}), processing early.")
            self._finalize_flow(key)

    # ─────────────────────────────────────────────────────────────────────────
    # Thread 2 — Flow timeout checker
    # ─────────────────────────────────────────────────────────────────────────

    def _timeout_loop(self):
        """
        Runs every 5 seconds.
        Any flow that has not received a packet in FLOW_TIMEOUT_SECONDS
        is treated as complete and sent to the API.
        """
        while self._running:
            time.sleep(5)
            self._expire_timed_out_flows()

    def _expire_timed_out_flows(self):
        """Find and finalize all flows that have exceeded the timeout."""
        now = time.time()
        timed_out = []

        with self._flows_lock:
            for key, flow in self._flows.items():
                age = now - flow.last_seen
                if age >= self.flow_timeout:
                    timed_out.append(key)

        for key in timed_out:
            log.debug(f"Flow timed out after {self.flow_timeout}s: {key}")
            self._finalize_flow(key)

    # ─────────────────────────────────────────────────────────────────────────
    # Flow finalization — extract features and call API
    # ─────────────────────────────────────────────────────────────────────────

    def _finalize_flow(self, key: FlowKey):
        """
        Remove the flow from active tracking, extract features,
        and POST the feature dict to the prediction API.
        """
        with self._flows_lock:
            flow = self._flows.pop(key, None)

        if flow is None or not flow.packet_dicts:
            return

        self.total_flows += 1

        # Extract features using FlowExtractor
        try:
            features = self._extractor.extract_from_dicts(flow.packet_dicts)
        except Exception as e:
            log.warning(f"Feature extraction failed for flow {key}: {e}")
            return

        # Add metadata that the API needs for logging
        src_ip, dst_ip, src_port, dst_port, protocol = key
        features["_source_ip"]      = src_ip       # underscore prefix = metadata
        features["_destination_ip"] = dst_ip
        features["_src_port"]       = float(src_port)
        features["_dst_port"]       = float(dst_port)

        # Send to prediction API (non-blocking)
        threading.Thread(
            target=self._call_api,
            args=(features, src_ip, dst_ip),
            daemon=True
        ).start()

    def _call_api(self, features: dict, src_ip: str, dst_ip: str):
        """
        POST feature dictionary to the FastAPI /api/predict endpoint.
        Runs in its own thread so it doesn't block the capture pipeline.
        """
        try:
            response = requests.post(
                self.api_url,
                json=features,
                timeout=5           # don't wait more than 5 seconds
            )
            self.total_api_calls += 1

            if response.status_code == 200:
                result = response.json()
                pred   = result.get("prediction", "?")
                conf   = result.get("confidence", 0)
                sev    = result.get("severity", "?")

                if pred != "BENIGN":
                    log.warning(
                        f"ALERT [{sev}]  {src_ip} → {dst_ip}  "
                        f"{pred}  ({conf*100:.1f}% confidence)"
                    )
                else:
                    log.debug(f"BENIGN  {src_ip} → {dst_ip}")
            else:
                log.warning(f"API returned {response.status_code}: {response.text[:100]}")

        except requests.exceptions.ConnectionError:
            log.debug("API not reachable. Is the FastAPI server running?")
        except requests.exceptions.Timeout:
            log.warning("API call timed out after 5s.")
        except Exception as e:
            log.warning(f"API call failed: {e}")

    # ─────────────────────────────────────────────────────────────────────────
    # Packet parsing helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _get_flow_key(self, pkt) -> Optional[FlowKey]:
        """
        Build a 5-tuple (src_ip, dst_ip, src_port, dst_port, protocol)
        from a packet. Returns None if the packet doesn't have an IP layer.

        For bidirectional flows, we sort the endpoints so that
        A→B and B→A are treated as the same flow.
        """
        try:
            src_ip = pkt[IP].src
            dst_ip = pkt[IP].dst

            if pkt.haslayer(TCP):
                src_port = pkt[TCP].sport
                dst_port = pkt[TCP].dport
                protocol = "TCP"
            elif pkt.haslayer(UDP):
                src_port = pkt[UDP].sport
                dst_port = pkt[UDP].dport
                protocol = "UDP"
            elif pkt.haslayer(ICMP):
                src_port = 0
                dst_port = 0
                protocol = "ICMP"
            else:
                # Other IP protocol — use raw protocol number
                src_port = 0
                dst_port = 0
                protocol = str(pkt[IP].proto)

            return (src_ip, dst_ip, src_port, dst_port, protocol)

        except Exception:
            return None

    def _packet_to_dict(self, pkt) -> dict:
        """
        Convert a Scapy packet to a plain dict so the FlowExtractor
        can process it without needing Scapy installed everywhere.
        """
        src_ip = pkt[IP].src if pkt.haslayer(IP) else ""
        size   = len(pkt)
        ts     = float(pkt.time)

        # Extract TCP flags as a string (e.g. "SA", "FP", "R")
        tcp_flags = ""
        if pkt.haslayer(TCP):
            tcp_flags = str(pkt[TCP].flags)

        return {
            "src_ip":    src_ip,
            "size":      size,
            "time":      ts,
            "tcp_flags": tcp_flags,
        }

    def _is_flow_terminator(self, pkt) -> bool:
        """
        Return True if this packet signals the end of a TCP connection
        (FIN or RST flag set).
        """
        if not pkt.haslayer(TCP):
            return False
        flags = str(pkt[TCP].flags)
        return "F" in flags or "R" in flags

    # ─────────────────────────────────────────────────────────────────────────
    # Stats
    # ─────────────────────────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        """Return current sniffer statistics."""
        with self._flows_lock:
            active_flows = len(self._flows)
        return {
            "total_packets":   self.total_packets,
            "total_flows":     self.total_flows,
            "active_flows":    active_flows,
            "total_api_calls": self.total_api_calls,
            "running":         self._running,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Standalone entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="NIDS Network Sniffer")
    parser.add_argument(
        "--interface", "-i",
        default="lo",
        help="Network interface to listen on (default: lo)"
    )
    parser.add_argument(
        "--api-url",
        default=API_PREDICT_URL,
        help=f"FastAPI prediction endpoint (default: {API_PREDICT_URL})"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=FLOW_TIMEOUT_SECONDS,
        help=f"Flow timeout in seconds (default: {FLOW_TIMEOUT_SECONDS})"
    )
    args = parser.parse_args()

    sniffer = NetworkSniffer(
        interface=args.interface,
        api_url=args.api_url,
        flow_timeout=args.timeout,
    )

    sniffer.start()

    # Keep the main thread alive; print stats every 30 seconds
    try:
        while True:
            time.sleep(30)
            stats = sniffer.get_stats()
            log.info(
                f"Stats — Packets: {stats['total_packets']} | "
                f"Flows processed: {stats['total_flows']} | "
                f"Active flows: {stats['active_flows']} | "
                f"API calls: {stats['total_api_calls']}"
            )
    except KeyboardInterrupt:
        log.info("Stopping sniffer ...")
        sniffer.stop()


if __name__ == "__main__":
    main()