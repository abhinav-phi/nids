"""
sniffer.py — Live Network Packet Capture (PRODUCTION)
======================================================
Captures packets from a network interface using Scapy.
Groups packets into flows using a 5-tuple key.
When a flow ends (timeout or TCP FIN/RST), extracts features
using FlowExtractor and sends them to the FastAPI backend.
Run standalone:
    python src/capture/sniffer.py --interface "Wi-Fi"
    python src/capture/sniffer.py --interface auto      # auto-detect
Integrated mode (started by FastAPI):
    Controlled via /api/sniffer/start and /api/sniffer/stop
"""
import os
import sys
import time
import logging
import threading
import argparse
import requests
import platform
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable
try:
    from scapy.all import sniff, IP, TCP, UDP, ICMP, conf, get_if_list, get_if_addr
    SCAPY_AVAILABLE = True
except ImportError:
    SCAPY_AVAILABLE = False
    print("[WARNING] Scapy not installed. Sniffer will not capture live packets.")
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2]))
from src.features.extractor import FlowExtractor
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [SNIFFER]  %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)
FLOW_TIMEOUT_SECONDS = 30       
MAX_PACKETS_PER_FLOW = 500      
API_PREDICT_URL      = "http://localhost:8000/api/predict"
ACTIVE_FLOW_LOG_INTERVAL = 15   
FlowKey = Tuple[str, str, int, int, str]
@dataclass
class Flow:
    """Holds all packets belonging to one network flow."""
    key: FlowKey
    packets: List         = field(default_factory=list)   
    packet_dicts: List    = field(default_factory=list)   
    start_time: float     = field(default_factory=time.time)
    last_seen: float      = field(default_factory=time.time)
    is_closed: bool       = False
def detect_interface() -> str:
    """
    Auto-detect the best network interface for packet capture.
    On Windows: picks the interface with a non-loopback IP address.
    On Linux/Mac: defaults to the first non-loopback interface.
    """
    if not SCAPY_AVAILABLE:
        return "lo"
    try:
        ifaces = get_if_list()
        log.info(f"Available interfaces: {ifaces}")
        if platform.system() == "Windows":
            try:
                from scapy.arch.windows import get_windows_if_list
                win_ifaces = get_windows_if_list()
                for iface in win_ifaces:
                    name = iface.get("name", "")
                    desc = iface.get("description", "").lower()
                    ips = iface.get("ips", [])
                    if "loopback" in desc or "virtual" in desc:
                        continue
                    if "npcap" in desc:
                        continue
                    for ip in ips:
                        if ip and not ip.startswith("127.") and ":" not in ip:
                            log.info(f"Auto-detected interface: {name} ({desc}) [{ip}]")
                            return name
            except Exception as e:
                log.debug(f"Windows interface detection fallback: {e}")
            for candidate in ifaces:
                try:
                    addr = get_if_addr(candidate)
                    if addr and not addr.startswith("127.") and addr != "0.0.0.0":
                        log.info(f"Auto-detected interface: {candidate} [{addr}]")
                        return candidate
                except Exception:
                    continue
        else:
            for candidate in ifaces:
                if candidate == "lo":
                    continue
                try:
                    addr = get_if_addr(candidate)
                    if addr and not addr.startswith("127.") and addr != "0.0.0.0":
                        log.info(f"Auto-detected interface: {candidate} [{addr}]")
                        return candidate
                except Exception:
                    continue
        log.warning("Could not auto-detect interface, using first available")
        return ifaces[0] if ifaces else "lo"
    except Exception as e:
        log.warning(f"Interface detection failed: {e}")
        return "lo"
class NetworkSniffer:
    """
    Captures live network packets, assembles them into flows,
    and submits completed flows to the ML prediction API.
    Usage:
        sniffer = NetworkSniffer(interface="auto")
        sniffer.start()
        ...
        sniffer.stop()
    """
    def __init__(
        self,
        interface: str = "auto",
        api_url: str = API_PREDICT_URL,
        flow_timeout: int = FLOW_TIMEOUT_SECONDS,
        direct_predict_fn: Optional[Callable] = None,
        ws_broadcast_fn: Optional[Callable] = None,
    ):
        if interface == "auto":
            self.interface = detect_interface()
        else:
            self.interface = interface
        self.api_url = api_url
        self.flow_timeout = flow_timeout
        self._direct_predict_fn = direct_predict_fn
        self._ws_broadcast_fn = ws_broadcast_fn
        self._flows: Dict[FlowKey, Flow] = {}
        self._flows_lock = threading.Lock()
        self._extractor = FlowExtractor()
        self.total_packets   = 0
        self.total_flows     = 0
        self.total_api_calls = 0
        self.total_alerts    = 0
        self._running = False
        self._capture_thread: Optional[threading.Thread] = None
        self._timeout_thread: Optional[threading.Thread] = None
    def start(self):
        """Start packet capture and flow timeout checker in background threads."""
        if not SCAPY_AVAILABLE:
            log.error("Scapy is not installed. Cannot start sniffer.")
            return
        self._running = True
        log.info(f"Starting capture on interface '{self.interface}' ...")
        log.info(f"Sending predictions to: {self.api_url}")
        self._capture_thread = threading.Thread(
            target=self._capture_loop,
            daemon=True,
            name="PacketCapture"
        )
        self._capture_thread.start()
        self._timeout_thread = threading.Thread(
            target=self._timeout_loop,
            daemon=True,
            name="FlowTimeout"
        )
        self._timeout_thread.start()
        log.info("Sniffer is running.")
    def stop(self):
        """Signal all threads to stop gracefully."""
        self._running = False
        self._flush_all_flows()
        log.info(
            f"Sniffer stopped. "
            f"Packets: {self.total_packets} | "
            f"Flows: {self.total_flows} | "
            f"API calls: {self.total_api_calls} | "
            f"Alerts: {self.total_alerts}"
        )
    def is_running(self) -> bool:
        return self._running
    def _capture_loop(self):
        """
        Runs Scapy's sniff() continuously.
        On Windows without Npcap, falls back to Layer 3 capture.
        """
        try:
            log.info(f"Capture thread started on '{self.interface}'")
            if platform.system() == "Windows":
                try:
                    sniff(
                        iface=self.interface,
                        prn=self._process_packet,
                        store=False,
                        stop_filter=lambda _: not self._running,
                    )
                except (OSError, RuntimeError) as e:
                    err_msg = str(e).lower()
                    if "winpcap" in err_msg or "npcap" in err_msg or "not installed" in err_msg or "layer 2" in err_msg:
                        log.warning(
                            "Npcap/WinPcap not found — falling back to Layer 3 capture. "
                            "Install Npcap from https://npcap.com for best results."
                        )
                        from scapy.all import conf as scapy_conf
                        sniff(
                            prn=self._process_packet,
                            store=False,
                            stop_filter=lambda _: not self._running,
                            opened_socket=scapy_conf.L3socket(),
                        )
                    else:
                        raise
            else:
                sniff(
                    iface=self.interface,
                    prn=self._process_packet,
                    store=False,
                    stop_filter=lambda _: not self._running,
                )
        except PermissionError:
            log.error(
                "Permission denied. Run with admin/sudo privileges, "
                "or ensure Npcap is installed on Windows."
            )
            self._running = False
        except OSError as e:
            if "No such device" in str(e) or "No such file" in str(e):
                log.error(
                    f"Interface '{self.interface}' not found. "
                    f"Available: {get_if_list() if SCAPY_AVAILABLE else 'N/A'}"
                )
            else:
                log.error(f"Capture error: {e}")
            self._running = False
        except Exception as e:
            log.error(f"Capture error: {e}")
            self._running = False
    def _process_packet(self, pkt):
        """Called for every captured packet. Adds it to the correct flow."""
        if not SCAPY_AVAILABLE:
            return
        self.total_packets += 1
        if not pkt.haslayer(IP):
            return
        key = self._get_flow_key(pkt)
        if key is None:
            return
        packet_dict = self._packet_to_dict(pkt, key)
        with self._flows_lock:
            if key not in self._flows:
                self._flows[key] = Flow(key=key)
                log.debug(f"New flow: {key[0]}:{key[2]} → {key[1]}:{key[3]} [{key[4]}]")
            flow = self._flows[key]
            flow.packets.append(pkt)
            flow.packet_dicts.append(packet_dict)
            flow.last_seen = time.time()
            if self._is_flow_terminator(pkt):
                flow.is_closed = True
        with self._flows_lock:
            should_finalize = key in self._flows and self._flows[key].is_closed
        if should_finalize:
            self._finalize_flow(key)
        with self._flows_lock:
            should_cap = key in self._flows and len(self._flows[key].packets) >= MAX_PACKETS_PER_FLOW
        if should_cap:
            log.debug(f"Flow hit max packet cap ({MAX_PACKETS_PER_FLOW}), processing early.")
            self._finalize_flow(key)
    def _timeout_loop(self):
        """Runs every 5 seconds. Finalizes timed-out flows."""
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
    def _flush_all_flows(self):
        """Finalize all remaining active flows (called on stop)."""
        with self._flows_lock:
            keys = list(self._flows.keys())
        for key in keys:
            self._finalize_flow(key)
    def _finalize_flow(self, key: FlowKey):
        """
        Remove the flow from active tracking, extract features,
        and POST the feature dict to the prediction API.
        """
        with self._flows_lock:
            flow = self._flows.pop(key, None)
        if flow is None or not flow.packet_dicts:
            return
        if len(flow.packet_dicts) < 3:
            return
        self.total_flows += 1
        try:
            features = self._extractor.extract_from_dicts(flow.packet_dicts, flow_key=key)
        except Exception as e:
            log.warning(f"Feature extraction failed for flow {key}: {e}")
            return
        src_ip, dst_ip, src_port, dst_port, protocol = key
        features["_source_ip"]      = src_ip
        features["_destination_ip"] = dst_ip
        features["_src_port"]       = float(src_port)
        features["_dst_port"]       = float(dst_port)
        log.debug(
            f"Flow finalized: {src_ip}:{src_port} → {dst_ip}:{dst_port} "
            f"[{protocol}] ({len(flow.packet_dicts)} pkts, "
            f"{len(features) - 4} features)"
        )
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
                timeout=10
            )
            self.total_api_calls += 1
            if response.status_code == 200:
                result = response.json()
                pred   = result.get("prediction", "?")
                conf   = result.get("confidence", 0)
                sev    = result.get("severity", "?")
                if pred != "BENIGN":
                    self.total_alerts += 1
                    log.warning(
                        f"🚨 ALERT [{sev}]  {src_ip} → {dst_ip}  "
                        f"{pred}  ({conf*100:.1f}% confidence)"
                    )
                else:
                    log.debug(f"✓ BENIGN  {src_ip} → {dst_ip}")
            else:
                log.warning(f"API returned {response.status_code}: {response.text[:200]}")
        except requests.exceptions.ConnectionError:
            log.debug("API not reachable. Is the FastAPI server running?")
        except requests.exceptions.Timeout:
            log.warning("API call timed out.")
        except Exception as e:
            log.warning(f"API call failed: {e}")
    def _get_flow_key(self, pkt) -> Optional[FlowKey]:
        """
        Build a 5-tuple from a packet. Returns None for non-IP packets.
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
                src_port = 0
                dst_port = 0
                protocol = str(pkt[IP].proto)
            return (src_ip, dst_ip, src_port, dst_port, protocol)
        except Exception:
            return None
    def _packet_to_dict(self, pkt, flow_key: FlowKey) -> dict:
        """
        Convert a Scapy packet to a plain dict with ALL fields
        needed by FlowExtractor for CICIDS2017 feature computation.
        Fields extracted:
        - src_ip, dst_ip, src_port, dst_port, protocol
        - size (total packet bytes)
        - payload_len (application-layer payload bytes)
        - header_len (IP + transport header bytes)
        - time (epoch timestamp)
        - tcp_flags (string like "SA", "PA", "F", "R")
        - window_size (TCP window, 0 for non-TCP)
        - ttl (IP time-to-live)
        """
        src_ip = pkt[IP].src if pkt.haslayer(IP) else ""
        dst_ip = pkt[IP].dst if pkt.haslayer(IP) else ""
        total_size = len(pkt)
        ts = float(pkt.time)
        tcp_flags = ""
        window_size = 0
        src_port = 0
        dst_port = 0
        transport_header_len = 0
        if pkt.haslayer(TCP):
            tcp_flags = str(pkt[TCP].flags)
            window_size = int(pkt[TCP].window)
            src_port = pkt[TCP].sport
            dst_port = pkt[TCP].dport
            transport_header_len = (pkt[TCP].dataofs or 5) * 4
        elif pkt.haslayer(UDP):
            src_port = pkt[UDP].sport
            dst_port = pkt[UDP].dport
            transport_header_len = 8  
        elif pkt.haslayer(ICMP):
            transport_header_len = 8  
        ip_header_len = (pkt[IP].ihl or 5) * 4 if pkt.haslayer(IP) else 20
        header_len = ip_header_len + transport_header_len
        eth_header = 14  
        payload_len = max(0, total_size - eth_header - header_len)
        ttl = pkt[IP].ttl if pkt.haslayer(IP) else 64
        return {
            "src_ip":       src_ip,
            "dst_ip":       dst_ip,
            "src_port":     src_port,
            "dst_port":     dst_port,
            "protocol":     flow_key[4],
            "size":         total_size,
            "payload_len":  payload_len,
            "header_len":   header_len,
            "time":         ts,
            "tcp_flags":    tcp_flags,
            "window_size":  window_size,
            "ttl":          ttl,
        }
    def _is_flow_terminator(self, pkt) -> bool:
        """Return True if this packet signals end of a TCP connection."""
        if not pkt.haslayer(TCP):
            return False
        flags = str(pkt[TCP].flags)
        return "F" in flags or "R" in flags
    def get_stats(self) -> dict:
        """Return current sniffer statistics."""
        with self._flows_lock:
            active_flows = len(self._flows)
        return {
            "interface":       self.interface,
            "total_packets":   self.total_packets,
            "total_flows":     self.total_flows,
            "active_flows":    active_flows,
            "total_api_calls": self.total_api_calls,
            "total_alerts":    self.total_alerts,
            "running":         self._running,
        }
def main():
    parser = argparse.ArgumentParser(description="NIDS Network Sniffer")
    parser.add_argument(
        "--interface", "-i",
        default="auto",
        help="Network interface to listen on (default: auto-detect)"
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
    try:
        while True:
            time.sleep(ACTIVE_FLOW_LOG_INTERVAL)
            stats = sniffer.get_stats()
            log.info(
                f"Stats — Packets: {stats['total_packets']} | "
                f"Flows: {stats['total_flows']} | "
                f"Active: {stats['active_flows']} | "
                f"API calls: {stats['total_api_calls']} | "
                f"Alerts: {stats['total_alerts']}"
            )
    except KeyboardInterrupt:
        log.info("Stopping sniffer ...")
        sniffer.stop()
if __name__ == "__main__":
    main()
