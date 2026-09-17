# Security Policy

**Project:** The Sentinel — Network Intrusion Detection System (NIDS)

The Sentinel is an ML-powered security tool — which means we hold it to the same standard it applies to network traffic. Thank you for taking the time to report issues responsibly.

---

## Supported Versions

| Version | Supported | Notes |
|---------|-----------|-------|
| 1.0.x (Colab Edition) | ✅ | Current release line |
| < 1.0.0 (local backend / React frontend) | ❌ | Removed in the Colab-only cleanup; fixes land on the current edition only |

---

## How to Report a Vulnerability

**Please do NOT open a public GitHub issue for security vulnerabilities.**

1. **Preferred:** use GitHub's **[Private Vulnerability Reporting](https://github.com/abhinav-phi/nids/security/advisories/new)** — this keeps the report confidential end-to-end and lets us coordinate a fix and advisory.
2. **Alternative:** open a draft security advisory through the repository's *Security* tab.

### What to include

- Affected notebook / component (`01_EDA`, `02_Training_GPU`, `03_Inference_API`, `04_Dashboard`, `_build/*`)
- A minimal reproduction (cell, payload, or request body)
- Impact assessment — what could an attacker do?
- Any known workarounds

### What to expect

| Step | Target |
|------|--------|
| Acknowledgement | within **48 hours** |
| Triage & severity assessment | within **7 days** |
| Fix or mitigation | severity-dependent, coordinated with you |
| Public disclosure | after the fix ships, credit to you unless you prefer otherwise |

---

## Scope

### ✅ In scope

- The **embedded FastAPI API** in notebook 03 (input validation, metadata handling, WebSocket fan-out, response sanitization)
- **Artifact integrity** — anything that could make `model.pkl` / `scaler.pkl` / `label_encoder.pkl` load or behave incorrectly (e.g., bypassing the `n_features_in_` parity guards)
- **Notebook code paths** that execute or persist attacker-controlled data (feature vectors, `_source_ip` metadata, replayed CSV content)
- **Supply-chain issues** in the pinned Colab dependencies (`_build` install cells)

### ❌ Out of scope

- **Google Colab platform security** — report to [Google VRP](https://bughunters.google.com) instead
- **Denial-of-service against the demo API** — the embedded server is intentionally demo-grade: optional API key (unset by default), no TLS, per-IP rate limit 120/min, WS client cap 20
- **Missing authentication / TLS on the public tunnel URLs** — documented limitation of the Colab edition; reports on this alone will be closed as *documented behavior* unless accompanied by a concrete hardening proposal
- **Attacks requiring write access to your Google Drive or Colab VM** — that is platform compromise, not a project vulnerability
- **Findings from intentionally adversarial inputs in `_build/smoke_test.py`** — those are the test suite's own fixtures

---

## Known & Accepted Limitations (by design)

These are documented in `README.md` / `docs/NIDS_PRD.md` and are **not** vulnerabilities:

1. No user authentication / authorization — the optional `X-API-Key` + WS token is defence-in-depth, not access control.
2. No TLS termination on the Cloudflare quick tunnel / Gradio share URL.
3. Per-process rate limiting and WS counters (single-process demo deployment).
4. Up to 8 of 52 missing features are zero-filled by design and surfaced via `missing_features`.

## Safe Handling

If you discover an exploitable issue, please do not run it against public deployments you do not own. For local reproduction, use the synthetic-flow fixtures in `_build/smoke_test.py` rather than live traffic.

---

*This policy follows the standard responsible-disclosure practice. For general questions (not vulnerabilities), see [SUPPORT.md](SUPPORT.md).*
