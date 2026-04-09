"""
sim_bruteforce.py — SSH Brute Force Simulator
===============================================
Sends repeated TCP packets to port 22 from one source IP.
Mimics an automated SSH password brute-force attack.
Characteristic features:
- Many packets to port 22 from one source IP
- Repeated connect attempts (SYN packets) that quickly close (RST/FIN)
- High flow volume from a single source to a single destination port
Run:
    sudo python src/simulation/sim_bruteforce.py
    sudo python src/simulation/sim_bruteforce.py --target 192.168.1.1 --attempts 100
"""
import time
import argparse
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s  [BRUTE-SIM]  %(message)s")
log = logging.getLogger(__name__)
try:
    from scapy.all import IP, TCP, Raw, send, conf
    conf.verb = 0
    SCAPY_AVAILABLE = True
except ImportError:
    SCAPY_AVAILABLE = False
    log.error("Scapy is not installed. Run: pip install scapy")
def simulate_bruteforce(
    target_ip:  str = "127.0.0.1",
    ssh_port:   int = 22,
    src_ip:     str = "10.0.0.77",
    attempts:   int = 100,
    delay:      float = 0.05,
):
    """
    Simulate an SSH brute force by sending repeated SYN packets to port 22.
    Each 'attempt' is a SYN followed by a RST (mimic failed login).
    Parameters
    ----------
    target_ip : IP of the SSH server being attacked
    ssh_port  : port to target (default 22)
    src_ip    : attacker's fixed source IP
    attempts  : number of login attempts to simulate
    delay     : seconds between each attempt
    """
    if not SCAPY_AVAILABLE:
        return
    log.info(f"Starting SSH brute force: {src_ip} → {target_ip}:{ssh_port}")
    log.info(f"Simulating {attempts} login attempts")
    for i in range(1, attempts + 1):
        src_port = 10000 + i    
        syn = IP(src=src_ip, dst=target_ip) / TCP(
            sport=src_port, dport=ssh_port, flags="S", seq=1000
        )
        send(syn, verbose=False)
        time.sleep(0.005)
        data = IP(src=src_ip, dst=target_ip) / TCP(
            sport=src_port, dport=ssh_port, flags="PA", seq=1001
        ) / Raw(load=b"SSH-2.0-OpenSSH_8.0\r\n")
        send(data, verbose=False)
        time.sleep(0.005)
        rst = IP(src=src_ip, dst=target_ip) / TCP(
            sport=src_port, dport=ssh_port, flags="R", seq=1050
        )
        send(rst, verbose=False)
        if i % 20 == 0:
            log.info(f"  Attempt {i}/{attempts} ...")
        time.sleep(delay)
    log.info(f"Brute force simulation complete. {attempts} attempts sent to {target_ip}:{ssh_port}.")
def main():
    parser = argparse.ArgumentParser(description="Simulate SSH brute force attack")
    parser.add_argument("--target",   default="127.0.0.1", help="Target IP")
    parser.add_argument("--src-ip",   default="10.0.0.77", help="Attacker source IP")
    parser.add_argument("--port",     type=int, default=22,  help="Target port (default: 22)")
    parser.add_argument("--attempts", type=int, default=100, help="Number of attempts")
    parser.add_argument("--delay",    type=float, default=0.05, help="Delay between attempts")
    args = parser.parse_args()
    simulate_bruteforce(args.target, args.port, args.src_ip, args.attempts, args.delay)
if __name__ == "__main__":
    main()
