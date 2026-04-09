"""
sim_portscan.py — Port Scan Simulator
========================================
Sends TCP SYN packets to many ports on a target IP.
Mimics an Nmap-style SYN scan (stealth scan).
Characteristic features the model will detect:
- Many SYN flags, zero FIN flags (no connections complete)
- Many different destination ports in sequence
- Short inter-arrival time between packets
- No backward traffic (target doesn't respond)
Run:
    sudo python src/simulation/sim_portscan.py
    sudo python src/simulation/sim_portscan.py --target 192.168.1.1 --start-port 20 --end-port 1024
"""
import time
import argparse
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s  [SCAN-SIM]  %(message)s")
log = logging.getLogger(__name__)
try:
    from scapy.all import IP, TCP, send, conf
    conf.verb = 0
    SCAPY_AVAILABLE = True
except ImportError:
    SCAPY_AVAILABLE = False
    log.error("Scapy is not installed. Run: pip install scapy")
def simulate_portscan(
    target_ip:  str = "127.0.0.1",
    start_port: int = 20,
    end_port:   int = 200,
    src_ip:     str = "10.0.0.99",
    delay:      float = 0.01,
):
    """
    Send SYN packets to every port in the range [start_port, end_port].
    Uses a fixed source IP so all packets group into related flows.
    Parameters
    ----------
    target_ip  : host being scanned
    start_port : first port to probe
    end_port   : last port to probe (inclusive)
    src_ip     : attacker's IP address
    delay      : seconds between each probe
    """
    if not SCAPY_AVAILABLE:
        return
    port_count = end_port - start_port + 1
    log.info(f"Starting port scan: {src_ip} → {target_ip}")
    log.info(f"Scanning ports {start_port}–{end_port} ({port_count} ports)")
    sent = 0
    for port in range(start_port, end_port + 1):
        packet = IP(src=src_ip, dst=target_ip) / TCP(
            sport=12345,   
            dport=port,
            flags="S",     
        )
        send(packet, verbose=False)
        sent += 1
        if sent % 50 == 0:
            log.info(f"  Scanned {sent}/{port_count} ports (current: {port})")
        time.sleep(delay)
    log.info(f"Port scan complete. {sent} SYN packets sent to {target_ip}.")
def main():
    parser = argparse.ArgumentParser(description="Simulate a TCP port scan")
    parser.add_argument("--target",     default="127.0.0.1", help="Target IP")
    parser.add_argument("--src-ip",     default="10.0.0.99", help="Attacker source IP")
    parser.add_argument("--start-port", type=int, default=20,  help="First port to scan")
    parser.add_argument("--end-port",   type=int, default=200, help="Last port to scan")
    parser.add_argument("--delay",      type=float, default=0.01, help="Delay between probes")
    args = parser.parse_args()
    simulate_portscan(args.target, args.start_port, args.end_port, args.src_ip, args.delay)
if __name__ == "__main__":
    main()
