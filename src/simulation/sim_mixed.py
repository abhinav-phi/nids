"""
sim_mixed.py — Mixed Attack Simulator
========================================
Runs all three attack simulations in sequence with a pause between each.
This is the best script to run during live demos — it generates
visible alerts of multiple types on the dashboard within ~2 minutes.

Run:
    sudo python src/simulation/sim_mixed.py
    sudo python src/simulation/sim_mixed.py --target 127.0.0.1 --pause 10
"""

import time
import argparse
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [MIXED-SIM]  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

from sim_ddos       import simulate_ddos
from sim_portscan   import simulate_portscan
from sim_bruteforce import simulate_bruteforce


def run_mixed(target_ip: str = "127.0.0.1", pause: int = 8):
    """
    Run DDoS → PortScan → BruteForce in sequence.
    Each simulation is followed by a pause so the sniffer
    can process flows before the next attack starts.

    Parameters
    ----------
    target_ip : target for all three simulations
    pause     : seconds to wait between simulations
    """
    log.info("=" * 55)
    log.info("  NIDS Mixed Attack Demo")
    log.info(f"  Target: {target_ip}")
    log.info(f"  Pause between attacks: {pause}s")
    log.info("=" * 55)
    log.info("Make sure the sniffer and API are running before starting!")
    log.info("Press Ctrl+C at any time to stop.\n")

    time.sleep(3)   # brief pause before starting

    # ── Attack 1: DDoS ────────────────────────────────────────────────────────
    log.info("\n[1/3] Starting DDoS simulation ...")
    simulate_ddos(
        target_ip  = target_ip,
        target_port= 80,
        count      = 200,
        delay      = 0.002,
    )
    log.info(f"[1/3] DDoS done. Waiting {pause}s for sniffer to process flows ...\n")
    time.sleep(pause)

    # ── Attack 2: Port Scan ───────────────────────────────────────────────────
    log.info("[2/3] Starting port scan simulation ...")
    simulate_portscan(
        target_ip  = target_ip,
        start_port = 20,
        end_port   = 150,
        src_ip     = "10.0.0.99",
        delay      = 0.01,
    )
    log.info(f"[2/3] Port scan done. Waiting {pause}s ...\n")
    time.sleep(pause)

    # ── Attack 3: Brute Force ─────────────────────────────────────────────────
    log.info("[3/3] Starting SSH brute force simulation ...")
    simulate_bruteforce(
        target_ip = target_ip,
        ssh_port  = 22,
        src_ip    = "10.0.0.77",
        attempts  = 60,
        delay     = 0.03,
    )
    log.info(f"[3/3] Brute force done.\n")

    # ── Summary ───────────────────────────────────────────────────────────────
    log.info("=" * 55)
    log.info("  All simulations complete!")
    log.info("  Check your NIDS dashboard for alerts.")
    log.info("  You should see: DDoS, PortScan, and FTP-Patator/SSH-Patator alerts.")
    log.info("=" * 55)


def main():
    parser = argparse.ArgumentParser(
        description="Run all attack simulations in sequence (best for demos)"
    )
    parser.add_argument("--target", default="127.0.0.1", help="Target IP for all attacks")
    parser.add_argument("--pause",  type=int, default=8,  help="Seconds between attacks")
    args = parser.parse_args()

    try:
        run_mixed(args.target, args.pause)
    except KeyboardInterrupt:
        log.info("\nSimulation interrupted by user.")


if __name__ == "__main__":
    main()