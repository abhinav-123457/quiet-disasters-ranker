import streamlit as st
import subprocess
import pandas as pd
import os
import time
import json

st.set_page_config(page_title="Quiet Disasters · AI Ranker", layout="wide", page_icon="◆")

# ════════════════════════════════════════════════════════════
#  THEME — formal, modern, restrained (Inter + one calm accent)
# ════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap');

:root{
  --bg:#0a0c11; --surface:#11141c; --surface-2:#0d1016;
  --border:#1e2430; --border-hi:#2b3342;
  --text:#e8eaf0; --muted:#9aa3b5; --faint:#5b6478;
  --accent:#5b8cff; --accent-soft:rgba(91,140,255,.12);
  --ok:#3fb950; --ok-soft:rgba(63,185,80,.12);
}
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
html,body,[data-testid="stAppViewContainer"]{
  background:var(--bg)!important;color:var(--text)!important;
  font-family:'Inter',sans-serif!important;
  -webkit-font-smoothing:antialiased;
}
[data-testid="stHeader"],[data-testid="stToolbar"],[data-testid="stSidebar"],footer{display:none!important}
[data-testid="stAppViewContainer"] > .main{padding:0!important}

/* single, static, very faint top glow — no animation */
[data-testid="stAppViewContainer"]::before{
  content:'';position:fixed;inset:0;z-index:0;pointer-events:none;
  background:radial-gradient(60% 40% at 50% -8%, rgba(91,140,255,.08) 0%, transparent 70%);
}

.block-container{max-width:760px!important;margin:0 auto!important;padding:0 24px 72px!important;position:relative;z-index:1}
[data-testid="stVerticalBlock"]>div{gap:0!important}

/* ── HERO ── */
.hero{text-align:center;padding:80px 0 30px}
.eyebrow{
  font-family:'JetBrains Mono',monospace;font-size:11px;font-weight:500;
  letter-spacing:.34em;color:var(--faint);text-transform:uppercase;
  display:flex;align-items:center;justify-content:center;gap:9px;margin-bottom:26px;
}
.eyebrow .dot{width:5px;height:5px;border-radius:50%;background:var(--accent)}
.hero-title{
  font-size:clamp(2.1rem,4.6vw,3.1rem);font-weight:800;line-height:1.08;
  letter-spacing:-.03em;color:var(--text);margin-bottom:18px;
}
.hero-title em{font-style:normal;color:var(--accent)}
.hero-sub{font-size:15px;line-height:1.7;color:var(--muted);max-width:500px;margin:0 auto 30px}
.hero-sub code{
  font-family:'JetBrains Mono',monospace;font-size:12.5px;
  background:var(--surface);border:1px solid var(--border);
  border-radius:5px;padding:2px 7px;color:var(--accent);
}
.chip-row{display:flex;gap:9px;justify-content:center;flex-wrap:wrap}
.chip{
  font-family:'JetBrains Mono',monospace;font-size:11px;letter-spacing:.02em;
  color:var(--muted);background:var(--surface);border:1px solid var(--border);
  border-radius:7px;padding:7px 13px;white-space:nowrap;
}

/* ── SECTION LABELS (no boxes — they sit above controls) ── */
.section{margin:34px 0 12px}
.section-num{
  font-family:'JetBrains Mono',monospace;font-size:11px;font-weight:700;
  color:var(--accent);letter-spacing:.1em;
}
.section-title{font-size:16px;font-weight:700;color:var(--text);margin:4px 0 2px;letter-spacing:-.01em}
.section-hint{font-family:'JetBrains Mono',monospace;font-size:11px;color:var(--faint);letter-spacing:.03em}

/* ── UPLOADER styled as one clean card ── */
[data-testid="stFileUploader"]{
  background:var(--surface-2)!important;border:1.5px dashed var(--border-hi)!important;
  border-radius:12px!important;padding:0!important;min-height:120px!important;
  display:flex!important;align-items:center!important;justify-content:center!important;
  transition:border-color .2s,background .2s;
}
[data-testid="stFileUploader"]:hover{border-color:var(--accent)!important;background:var(--accent-soft)!important}
[data-testid="stFileUploader"] section{padding:22px!important;background:transparent!important}
[data-testid="stFileUploader"] label,[data-testid="stFileUploader"] span,[data-testid="stFileUploader"] div{color:var(--muted)!important}
[data-testid="stFileUploader"] small{font-family:'JetBrains Mono',monospace!important;color:var(--faint)!important}

/* ── BUTTONS — flat, formal ── */
[data-testid="stButton"]>button{
  font-family:'Inter',sans-serif!important;font-weight:600!important;font-size:14px!important;
  letter-spacing:.01em!important;border-radius:10px!important;width:100%!important;
  padding:13px 24px!important;transition:all .16s ease!important;
}
[data-testid="stButton"]>button[kind="primary"]{
  background:var(--accent)!important;color:#fff!important;border:1px solid var(--accent)!important;
  box-shadow:0 4px 16px -6px rgba(91,140,255,.5)!important;
}
[data-testid="stButton"]>button[kind="primary"]:hover{filter:brightness(1.08)!important;transform:translateY(-1px)!important}
[data-testid="stButton"]>button[kind="primary"]:disabled{
  background:var(--surface)!important;color:var(--faint)!important;border-color:var(--border)!important;box-shadow:none!important;
}
[data-testid="stButton"]>button[kind="secondary"]{
  background:transparent!important;color:var(--muted)!important;border:1px solid var(--border-hi)!important;
}
[data-testid="stButton"]>button[kind="secondary"]:hover{border-color:var(--accent)!important;color:var(--text)!important}
.btn-divider{font-family:'JetBrains Mono',monospace;font-size:11px;color:var(--faint);text-align:center;margin:8px 0;letter-spacing:.1em}

[data-testid="stAlert"]{border-radius:9px!important;font-size:13.5px!important;border:1px solid var(--border)!important;margin-top:12px!important}

/* ── STATUS / SPINNER ── */
[data-testid="stStatusWidget"],[data-testid="stStatus"]{
  background:var(--surface)!important;border:1px solid var(--border)!important;border-radius:10px!important;
}

/* ── METRICS ── */
.metric-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin:8px 0 18px}
@media(max-width:600px){.metric-grid{grid-template-columns:repeat(2,1fr)}}
.metric-card{background:var(--surface);border:1px solid var(--border);border-radius:11px;padding:14px;text-align:center}
.m-val{font-family:'JetBrains Mono',monospace;font-size:18px;font-weight:700;color:var(--text)}
.m-val.accent{color:var(--accent)}
.m-lab{font-family:'JetBrains Mono',monospace;font-size:9.5px;letter-spacing:.13em;color:var(--faint);margin-top:5px;text-transform:uppercase}

/* ── TOP-10 PODIUM ── */
.pod-card{display:flex;gap:14px;background:var(--surface);border:1px solid var(--border);border-radius:11px;padding:15px 16px;margin-bottom:10px;animation:fadeUp .35s ease both}
.pod-card:hover{border-color:var(--border-hi)}
.pod-rank{font-family:'JetBrains Mono',monospace;font-weight:700;font-size:15px;min-width:46px;height:46px;border-radius:10px;display:flex;align-items:center;justify-content:center;border:1px solid var(--border);color:var(--muted);background:var(--surface-2)}
.pod-1 .pod-rank{color:#e8c468;border-color:rgba(232,196,104,.4);background:rgba(232,196,104,.07)}
.pod-2 .pod-rank{color:#c3cad8;border-color:rgba(195,202,216,.32);background:rgba(195,202,216,.05)}
.pod-3 .pod-rank{color:#cf9b6b;border-color:rgba(207,155,107,.4);background:rgba(207,155,107,.07)}
.pod-body{flex:1;min-width:0}
.pod-top{display:flex;align-items:center;gap:10px;margin-bottom:7px}
.pod-cid{font-family:'JetBrains Mono',monospace;font-size:12.5px;color:var(--accent);font-weight:500}
.pod-score{font-family:'JetBrains Mono',monospace;font-size:11.5px;color:var(--ok);margin-left:auto}
.pod-bar{height:4px;border-radius:2px;background:var(--surface-2);overflow:hidden;margin-bottom:9px}
.pod-bar span{display:block;height:100%;border-radius:2px;background:var(--accent)}
.pod-text{font-size:13px;line-height:1.62;color:var(--muted)}

/* ── TABS / DATAFRAME / DOWNLOAD ── */
[data-testid="stTabs"] button{font-family:'Inter',sans-serif!important;font-size:13px!important;font-weight:600!important;color:var(--faint)!important}
[data-testid="stTabs"] button[aria-selected="true"]{color:var(--text)!important}
[data-testid="stTabs"] [data-baseweb="tab-highlight"]{background:var(--accent)!important}
[data-testid="stDataFrame"]{border-radius:10px!important;border:1px solid var(--border)!important;overflow:hidden!important}
[data-testid="stDownloadButton"]>button{
  font-family:'Inter',sans-serif!important;font-weight:600!important;font-size:13px!important;
  background:transparent!important;border:1px solid var(--border-hi)!important;color:var(--muted)!important;
  border-radius:9px!important;padding:10px 18px!important;margin-top:12px!important;transition:all .15s;
}
[data-testid="stDownloadButton"]>button:hover{border-color:var(--accent)!important;color:var(--text)!important}
.stCode,pre{border-radius:9px!important}

@keyframes fadeUp{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:translateY(0)}}
.hero,.section,.metric-grid{animation:fadeUp .5s ease both}
@media(prefers-reduced-motion:reduce){*{animation:none!important}}
</style>
""", unsafe_allow_html=True)

# ── HERO ──
st.markdown("""
<div class="hero">
  <div class="eyebrow"><span class="dot"></span>Quiet Disasters · Sandbox</div>
  <div class="hero-title">Candidate <em>Ranking</em> System</div>
  <div class="hero-sub">
    A multi-stage retrieval &amp; ranking pipeline that surfaces the strongest
    candidates for the role — running entirely on CPU, within the
    competition's 5-minute budget.
  </div>
  <div class="chip-row">
    <span class="chip">CPU only · no GPU</span>
    <span class="chip">bge-small + MiniLM-L-12 reranker</span>
    <span class="chip">Offline ranking · no API calls</span>
  </div>
</div>
""", unsafe_allow_html=True)


def count_candidates(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read().strip()
        if content.startswith("["):
            return len(json.loads(content))
        return sum(1 for l in content.split("\n") if l.strip())
    except Exception:
        return None


# ── SECTION 1: DATA ──
st.markdown("""
<div class="section">
  <div class="section-num">01</div>
  <div class="section-title">Provide candidates</div>
  <div class="section-hint">.jsonl or .json · up to 100 candidates</div>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload candidates", type=["jsonl", "json"],
                                 label_visibility="collapsed")

if uploaded_file is not None:
    with open("./candidates.jsonl", "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.session_state.sample_loaded = False
    n = count_candidates("./candidates.jsonl")
    st.success(f"**{uploaded_file.name}** · {uploaded_file.size:,} bytes"
               + (f" · **{n} candidates parsed**" if n else ""))

if uploaded_file is None and os.path.exists("sample_candidates.json"):
    if st.button("Load the bundled 50-candidate sample", type="secondary",
                 use_container_width=True):
        with open("sample_candidates.json", "rb") as src, open("./candidates.jsonl", "wb") as dst:
            dst.write(src.read())
        st.session_state.sample_loaded = True
    if st.session_state.get("sample_loaded"):
        n = count_candidates("./candidates.jsonl")
        st.success(f"Bundled sample loaded · **{n} candidates** ready")

data_ready = (uploaded_file is not None) or st.session_state.get("sample_loaded", False)

# ── SECTION 2: RUN ──
st.markdown("""
<div class="section">
  <div class="section-num">02</div>
  <div class="section-title">Execute pipeline</div>
  <div class="section-hint">sample ≈ 1 min · full 100K run may take several minutes on Space CPU</div>
</div>
""", unsafe_allow_html=True)

run_uploaded = st.button("Run on provided sample", type="primary",
                         disabled=(not data_ready), use_container_width=True)
st.markdown('<div class="btn-divider">OR</div>', unsafe_allow_html=True)
run_preloaded = st.button("Run on pre-loaded 100K dataset", type="secondary",
                          use_container_width=True)

# ── EXECUTION (live streamed) ──
if run_uploaded or run_preloaded:
    if not os.path.exists("./artifacts"):
        st.error("Artifacts folder not found. Ensure precomputed artifacts are available to the Space.")
    else:
        script_name = "rank_small.py" if run_uploaded else "rank.py"
        cmd = ["python", "-u", script_name,
               "--candidates", "./candidates.jsonl",
               "--artifacts", "./artifacts", "--out", "./submission.csv"]
        start = time.time()
        log_lines = []
        with st.status(f"Starting {script_name} …", expanded=True) as status:
            log_box = st.empty()
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True, bufsize=1)
            for line in proc.stdout:
                line = line.rstrip()
                if not line:
                    continue
                log_lines.append(line)
                if "STAGE 1" in line:
                    status.update(label="Stage 1 / 3 — Retrieval (FAISS + skills + behavioral)")
                elif "STAGE 2" in line:
                    status.update(label="Stage 2 / 3 — Cross-encoder re-ranking")
                elif "STAGE 3" in line:
                    status.update(label="Stage 3 / 3 — Feature scoring + JD-fit signals")
                elif "Writing CSV" in line:
                    status.update(label="Writing ranked results")
                log_box.code("\n".join(log_lines[-12:]), language="bash")
            proc.wait()
            duration = time.time() - start
            if proc.returncode == 0:
                status.update(label=f"Pipeline complete in {duration:.1f}s",
                              state="complete", expanded=False)
            else:
                status.update(label="Pipeline failed — see logs", state="error")
        st.session_state.run_success = proc.returncode == 0
        st.session_state.run_failed = proc.returncode != 0
        st.session_state.duration = duration
        st.session_state.stderr = "\n".join(log_lines)

# ── RESULTS ──
if st.session_state.get("run_success", False) and os.path.exists("submission.csv"):
    df = pd.read_csv("submission.csv")
    top_score = df["score"].max()

    st.markdown("""
    <div class="section"><div class="section-num">03</div>
    <div class="section-title">Results</div></div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="metric-grid">
      <div class="metric-card"><div class="m-val accent">{top_score:.4f}</div><div class="m-lab">Top score</div></div>
      <div class="metric-card"><div class="m-val">{df["score"].median():.4f}</div><div class="m-lab">Median</div></div>
      <div class="metric-card"><div class="m-val">{len(df):,}</div><div class="m-lab">Ranked</div></div>
      <div class="metric-card"><div class="m-val">{st.session_state.duration:.1f}s</div><div class="m-lab">Runtime</div></div>
    </div>
    """, unsafe_allow_html=True)

    t_table, t_top, t_chart, t_logs = st.tabs(
        ["Ranked table", "Top 10", "Scores", "Logs"])

    with t_table:
        st.dataframe(
            df,
            column_config={
                "rank": st.column_config.NumberColumn("Rank", format="%d"),
                "score": st.column_config.NumberColumn("Score", format="%.6f"),
                "reasoning": st.column_config.TextColumn("Reasoning", width="large"),
            },
            use_container_width=True, hide_index=True,
        )
        with open("submission.csv", "rb") as f:
            st.download_button("Download submission.csv", data=f,
                               file_name="submission.csv", mime="text/csv")

    with t_top:
        for _, row in df.head(10).iterrows():
            r = int(row["rank"])
            cls = f"pod-card pod-{r}" if r <= 3 else "pod-card"
            width = max(row["score"] / top_score * 100, 5)
            st.markdown(f"""
            <div class="{cls}">
              <div class="pod-rank">{r}</div>
              <div class="pod-body">
                <div class="pod-top">
                  <span class="pod-cid">{row["candidate_id"]}</span>
                  <span class="pod-score">{row["score"]:.6f}</span>
                </div>
                <div class="pod-bar"><span style="width:{width:.1f}%"></span></div>
                <div class="pod-text">{row["reasoning"]}</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

    with t_chart:
        st.bar_chart(df.set_index("rank")["score"], height=260, color="#5b8cff")
        st.caption("Score by rank — non-increasing by construction.")

    with t_logs:
        st.code(st.session_state.stderr, language="bash")

elif st.session_state.get("run_failed", False):
    st.error("Execution failed. Logs below.")
    st.code(st.session_state.stderr, language="bash")