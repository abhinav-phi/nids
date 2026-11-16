# %% [markdown]
# # 04 — Live Dashboard · Colab Edition · *The Sentinel* NIDS
#
# The repo ships a React + Vite command center (`nids-frontend/`) — a Node dev
# server can't meaningfully run inside a Colab VM, so this notebook rebuilds
# the **same dashboard over the same API** with Gradio:
#
# | React page | This notebook |
# |---|---|
# | `/` Dashboard — KPIs, attack pie, live alert feed | **📊 Dashboard** tab (auto-refresh) |
# | `/reports` + `/network` — distribution, leaderboard, timeline | **🕵 Threat intel** tab |
# | `/explain` — SHAP feature contributions per alert | **🧠 Explain** tab |
# | attack simulators + `send_attacks.py` replay | **🧪 Traffic lab** tab |
# | WebSocket live stream | 3-second polling of the same endpoints |
#
# The notebook boots the **same embedded API** as `03_Inference_API_Colab.ipynb`
# (predict → persist → broadcast on SQLite), so the dashboard consumes real
# alerts — nothing synthetic.
#
# **Before running:** artifacts from `02_Training_GPU_Colab.ipynb`
# (Drive sync or `nids_artifacts.zip` in `/content/`).

# %%
!pip install -q gradio shap

# %%include frag_discovery.py
# %%include frag_inference.py
# %%include frag_api.py
# %%include frag_server.py

# %% [markdown]
# ## 🖥️ Step 1 — Dashboard data layer
#
# Every figure below reads from the API exactly like the React `client.ts`
# does — `/api/stats`, `/api/alerts`, `/api/ip-leaderboard` — so this UI is a
# faithful (not simulated) view of the backend state.

# %%
import json
import random
from collections import Counter
from datetime import datetime

import matplotlib.pyplot as plt

SEV_ORDER  = ["CRITICAL", "HIGH", "MEDIUM", "LOW", "NONE"]
SEV_COLORS = {"CRITICAL": "#dc2626", "HIGH": "#ea580c", "MEDIUM": "#d97706",
              "LOW": "#2563eb", "NONE": "#6b7280"}
FEED_HEADERS = ["Time (UTC)", "Source", "Type", "Severity", "Conf"]

def _get(path, **params):
    try:
        return rq.get(f"{API_BASE}{path}", params=params or None, timeout=5).json()
    except Exception:
        return None

def kpi_markdown(stats):
    if not stats:
        return "**⚠️ Backend offline** — re-run the server cell above."
    up = stats.get("uptime_seconds", 0)
    return (
        f"| 🌊 Flows analyzed | 🚨 Attacks | ✅ Benign | ⏱ Uptime |\n"
        f"|---|---|---|---|\n"
        f"| **{stats['total_flows']:,}** | **{stats['total_attacks']:,}** "
        f"| **{stats['benign_count']:,}** | **{int(up // 60)}m {int(up % 60)}s** |"
    )

def attack_pie_fig(stats):
    plt.close("all")
    fig, ax = plt.subplots(figsize=(5.2, 4))
    by_type = (stats or {}).get("attacks_by_type") or {}
    if not by_type:
        ax.text(0.5, 0.5, "No attacks yet\n→ use the Traffic Lab tab",
                ha="center", va="center", fontsize=11)
        ax.axis("off")
    else:
        items = sorted(by_type.items(), key=lambda kv: -kv[1])
        ax.pie([v for _, v in items], labels=[k for k, _ in items],
               autopct=lambda p: f"{p:.0f}%" if p >= 5 else "",
               startangle=90, textprops={"fontsize": 9},
               colors=plt.cm.Set2.colors)
        ax.set_title("Attacks by type", fontweight="bold")
    plt.tight_layout()
    return fig

def severity_bar_fig(stats):
    plt.close("all")
    fig, ax = plt.subplots(figsize=(5.2, 4))
    by_sev = (stats or {}).get("attacks_by_severity") or {}
    order = [s for s in SEV_ORDER if s in by_sev]
    if order:
        bars = ax.bar(order, [by_sev[s] for s in order],
                      color=[SEV_COLORS[s] for s in order])
        ax.bar_label(bars, fontsize=9)
        ax.set_yscale("symlog")
    else:
        ax.text(0.5, 0.5, "No attacks yet", ha="center", va="center", fontsize=11)
        ax.axis("off")
    ax.set_title("Attacks by severity", fontweight="bold")
    plt.tight_layout()
    return fig

def feed_dataframe(limit=12):
    alerts = _get("/api/alerts", limit=limit) or []
    rows = [[
        (a["timestamp"][11:19] if a["timestamp"] else ""),
        f"{a['source_ip']}:{a['src_port']}",
        a["prediction"], a["severity"], f"{a['confidence'] * 100:.1f}%",
    ] for a in alerts]
    return pd.DataFrame(rows, columns=FEED_HEADERS)

def leaderboard_dataframe(limit=10):
    rows = _get("/api/ip-leaderboard", limit=limit) or []
    data = [[r["rank"], r["source_ip"], r["attack_count"],
             (r["last_seen"] or "")[11:19]] for r in rows]
    return pd.DataFrame(data, columns=["Rank", "Attacker IP", "Attacks", "Last seen"])

def timeline_fig():
    plt.close("all")
    fig, ax = plt.subplots(figsize=(7, 4))
    alerts = _get("/api/alerts", limit=500) or []
    hours = Counter()
    for a in alerts:
        try:
            hours[datetime.fromisoformat(a["timestamp"]).strftime("%H:00")] += 1
        except (TypeError, ValueError):
            continue
    if not hours:
        ax.text(0.5, 0.5, "No attacks in the last 500 alerts",
                ha="center", va="center", fontsize=11)
        ax.axis("off")
    else:
        labels = sorted(hours)
        ax.bar(labels, [hours[h] for h in labels], color="#dc2626", alpha=0.85)
        ax.set_xlabel("Hour (UTC)")
        ax.set_ylabel("Alerts")
        ax.tick_params(axis="x", rotation=45)
    ax.set_title("Attack timeline (last 500 alerts)", fontweight="bold")
    plt.tight_layout()
    return fig

def alert_choices(limit=50):
    alerts = _get("/api/alerts", limit=limit) or []
    return [f"#{a['id']} · {a['prediction']} · {a['severity']}" for a in alerts]

def explain_alert(choice):
    """SHAP top-5 bar plot for one alert — the /explain page."""
    if not choice:
        return None, "Pick an alert from the dropdown."
    alerts = _get("/api/alerts", limit=50) or []
    target_id = choice.split(" · ")[0].lstrip("#")
    alert = next((a for a in alerts if str(a["id"]) == target_id), None)
    if alert is None:
        return None, "Alert not found in the last 50."
    try:
        items = json.loads(alert["shap_json"]) if alert["shap_json"] else []
    except json.JSONDecodeError:
        items = []
    if not items:
        return None, f"Alert #{alert['id']} ({alert['prediction']}) has no SHAP payload."
    items = sorted(items, key=lambda s: s["value"])
    plt.close("all")
    fig, ax = plt.subplots(figsize=(7, max(1.6, 0.55 * len(items) + 1)))
    ax.barh([s["feature"][:30] for s in items], [s["value"] for s in items],
            color=["#dc2626" if s["value"] > 0 else "#2563eb" for s in items])
    ax.axvline(0, color="black", lw=0.8)
    ax.set_title(f"SHAP — {alert['prediction']} (confidence {alert['confidence'] * 100:.1f}%)",
                 fontweight="bold")
    ax.set_xlabel("SHAP value (pushes toward / away from this attack class)")
    plt.tight_layout()
    details = (f"**#{alert['id']}** · `{alert['source_ip']}` → `{alert['destination_ip']}` · "
               f"**{alert['prediction']}** · severity **{alert['severity']}** · "
               f"confidence **{alert['confidence'] * 100:.1f}%**")
    return fig, details

# %% [markdown]
# ## 🧪 Step 2 — Traffic injection (the "attack simulator")
#
# Equivalent of `send_attacks.py` / `src/simulation/*`: balanced CICIDS2017
# flows are replayed through `POST /api/predict` with synthetic attacker IPs.
# The dashboard tabs pick the new alerts up on their next refresh.

# %%
_POOL = None

def _replay_pool():
    """Lazily build a generous balanced pool (30/class) from the CSV once."""
    global _POOL
    if _POOL is not None:
        return _POOL
    if DATA_PATH is None:
        _POOL = pd.DataFrame()
        return _POOL
    from collections import Counter as _C
    got, batches = _C(), []
    want_default = 30
    for chunk in pd.read_csv(DATA_PATH, chunksize=250_000):
        chunk.columns = chunk.columns.str.strip()
        chunk = chunk.replace([np.inf, -np.inf], np.nan).dropna()
        for cls, group in chunk.groupby("Attack Type"):
            want = want_default * 2 if cls == "Normal Traffic" else want_default
            if got[cls] >= want:
                continue
            take = group.sample(n=min(want - got[cls], len(group)), random_state=42)
            batches.append(take)
            got[cls] += len(take)
        if all(got[c] >= (want_default * 2 if c == "Normal Traffic" else want_default)
               for c in got):
            break
    _POOL = pd.concat(batches).reset_index(drop=True) if batches else pd.DataFrame()
    print(f"Replay pool ready: {got}")
    return _POOL

def inject_traffic(per_type=5):
    pool = _replay_pool()
    if pool.empty:
        return ("**CSV not available** — upload `cicids2017_cleaned.csv` to "
                "`MyDrive/nids_data/` to enable traffic injection.")
    parts = []
    for cls, group in pool.groupby("Attack Type"):
        n = per_type * 2 if cls == "Normal Traffic" else per_type
        parts.append(group.sample(n=min(len(group), n),
                                  random_state=random.randint(0, 10 ** 6)))
    batch = pd.concat(parts)
    feature_cols = [c for c in pool.columns if c != "Attack Type"]
    sent = attacks = 0
    for _, row in batch.iterrows():
        payload = {c: float(row[c]) for c in feature_cols}
        payload["_source_ip"] = f"10.0.0.{1 + (sent % 200)}"
        payload["_destination_ip"] = "192.168.1.10"
        try:
            r = rq.post(f"{API_BASE}/api/predict", json=payload, timeout=15)
            if r.status_code == 200:
                sent += 1
                if r.json()["severity"] != "NONE":
                    attacks += 1
        except Exception:
            continue
    return (f"Injected **{sent}** flows → **{attacks}** produced attack alerts.\n\n"
            f"Switch to the **📊 Dashboard** tab — it auto-refreshes every 3 s.")

# %% [markdown]
# ## 🚀 Step 3 — Launch the command center
#
# Gradio serves the UI on a **public `*.gradio.live` URL** — open it on your
# laptop or phone. The Dashboard tab polls the API every 3 seconds (the
# notebook-equivalent of the WebSocket feed).

# %%
try:
    import gradio as gr
    HAVE_GRADIO = True
    print(f"Gradio {gr.__version__} ✔")
except Exception as e:
    HAVE_GRADIO = False
    print(f"(gradio unavailable: {e} — the static fallback dashboard will be used)")

# %%
if HAVE_GRADIO:
    def refresh_all():
        stats = _get("/api/stats")
        return (
            kpi_markdown(stats),
            attack_pie_fig(stats),
            severity_bar_fig(stats),
            feed_dataframe(),
            leaderboard_dataframe(),
            timeline_fig(),
            gr.update(choices=alert_choices()),
        )

    REFRESH_OUTPUTS = None  # set below

    with gr.Blocks(title="The Sentinel — NIDS Command Center") as demo:
        gr.Markdown(
            "# 🛡️ The Sentinel — NIDS Command Center (Colab)\n"
            "ML-powered intrusion detection · CICIDS2017 · live alerts · SHAP explainability")
        kpi_md = gr.Markdown()
        with gr.Tabs():
            with gr.Tab("📊 Dashboard"):
                with gr.Row():
                    pie = gr.Plot(label="Attacks by type")
                    sev = gr.Plot(label="Attacks by severity")
                gr.Markdown("### 🚨 Live alert feed")
                feed = gr.DataFrame(headers=FEED_HEADERS, interactive=False)
            with gr.Tab("🕵 Threat intel"):
                with gr.Row():
                    board = gr.DataFrame(label="Top attackers", interactive=False)
                    tl = gr.Plot(label="Attack timeline")
            with gr.Tab("🧠 Explain"):
                with gr.Row():
                    shap_dd = gr.Dropdown(label="Pick an alert (last 50)",
                                          choices=[], interactive=True)
                shap_info = gr.Markdown()
                shap_plot = gr.Plot(label="SHAP top-5 contributions")
                shap_dd.change(explain_alert, inputs=shap_dd,
                               outputs=[shap_plot, shap_info])
            with gr.Tab("🧪 Traffic lab"):
                gr.Markdown(
                    "Replay balanced CICIDS2017 flows through the API — the offline "
                    "equivalent of `send_attacks.py` / the attack simulators.")
                per_type = gr.Slider(1, 25, value=5, step=1,
                                     label="Flows per attack class")
                inject_btn = gr.Button("Inject traffic 🚀", variant="primary")
                inject_log = gr.Markdown()
                inject_btn.click(inject_traffic, inputs=per_type, outputs=inject_log)
        with gr.Row():
            refresh_btn = gr.Button("⟳ Refresh now")

        REFRESH_OUTPUTS = [kpi_md, pie, sev, feed, board, tl, shap_dd]
        refresh_btn.click(refresh_all, outputs=REFRESH_OUTPUTS)
        demo.load(refresh_all, outputs=REFRESH_OUTPUTS)   # initial paint on page open
        if hasattr(gr, "Timer"):
            # Gradio ≥ 5: auto-refresh via Timer.tick — `load(every=)` was removed
            # in Gradio 5 (TypeError on Colab's gradio 5.x).
            gr.Timer(3).tick(refresh_all, outputs=REFRESH_OUTPUTS)
        else:
            # Gradio 4.x fallback: repeated load event
            demo.load(refresh_all, outputs=REFRESH_OUTPUTS, every=3)

    demo.launch(share=True, show_error=True)
else:
    print("Gradio missing — running the static fallback snapshot instead.")

# %%
# Fallback: static dashboard snapshot (no Gradio) — one refresh per run.
if not HAVE_GRADIO:
    stats = _get("/api/stats")
    print(kpi_markdown(stats))
    fig = attack_pie_fig(stats); plt.show()
    fig = severity_bar_fig(stats); plt.show()
    fig = timeline_fig(); plt.show()
    print("\nTop attackers:")
    display(leaderboard_dataframe())
    print("\nLatest alerts:")
    display(feed_dataframe())
    print("\n(pip install gradio, then re-run, for the live auto-refreshing UI)")

# %% [markdown]
# ## 🖼️ Step 4 — Static dashboard preview (for your report)
# Gradio is the live UI; this cell renders the **same data** as static PNGs so
# you can embed a "command-center screenshot" in your project report even
# without opening the share URL.

# %%
stats = _get("/api/stats")

previews = [
    ("dashboard_attacks_by_type.png", attack_pie_fig(stats)),
    ("dashboard_attacks_by_severity.png", severity_bar_fig(stats)),
    ("dashboard_timeline.png", timeline_fig()),
]
for name, fig in previews:
    fig.savefig(ART_DIR / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved → {ART_DIR / name}")

print("\nTop attackers (last refresh):")
display(leaderboard_dataframe())
print("\nLatest alert feed (last refresh):")
display(feed_dataframe(10))

# %% [markdown]
# ## ✅ Wrap-up — what runs where
#
# | Capability | Repo (local machine) | Colab (these notebooks) |
# |---|---|---|
# | Model training | `src/model/train.py` | ✔ `02_Training_GPU_Colab.ipynb` (T4 GPU) |
# | Inference + SHAP + severity | `src/model/predict.py` | ✔ notebooks 03/04 |
# | Flow extraction from packets | `src/features/extractor.py` | ✔ notebook 03 (synthetic flows) |
# | REST API + WebSocket | `src/api/*` | ✔ notebooks 03/04 (embedded server) |
# | Dataset replay | `send_attacks.py` | ✔ notebooks 03/04 |
# | Live dashboard | React app (`npm run dev`) | ✔ this notebook (Gradio UI) |
# | **Scapy live capture** | `src/capture/sniffer.py` | ✖ needs raw sockets on a real NIC |
# | **Packet simulators** | `src/simulation/*` | ✖ need a real interface to attack |
# | Gemini chatbot endpoint | `/api/chat` (LangChain) | ✔ mini-port in notebook 03 (optional) |
#
# **Going back to the local project:** download `nids_artifacts.zip` from
# notebook 02, unzip into `nids-backend/`, and run
# `uvicorn src.api.main:app --port 8000` + `npm run dev` as usual — the
# artifacts are contract-compatible (`model.pkl` / `scaler.pkl` /
# `label_encoder.pkl` / `manifest.json`).
