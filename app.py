"""Emotion AI - Streamlit GUI (icon-based UI).   Run:  streamlit run app.py"""
import html
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Emotion AI", page_icon=":material/psychology:", layout="wide")

# ------------------------------------------------------------------ icons (inline SVG, stroke style)
_I = {
    "network": '<circle cx="5" cy="6" r="2"/><circle cx="5" cy="18" r="2"/><circle cx="12" cy="12" r="2.4"/>'
               '<circle cx="19" cy="7" r="2"/><circle cx="19" cy="17" r="2"/>'
               '<path d="M6.8 6.8 10 10.8M6.8 17.2 10 13.2M14.2 11 17.3 8M14.2 13 17.3 16"/>',
    "target": '<circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/>',
    "database": '<ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M3 5v14c0 1.7 4 3 9 3s9-1.3 9-3V5"/>'
                '<path d="M3 12c0 1.7 4 3 9 3s9-1.3 9-3"/>',
    "cpu": '<rect x="4" y="4" width="16" height="16" rx="2"/><rect x="9" y="9" width="6" height="6"/>'
           '<path d="M9 2v2M15 2v2M9 20v2M15 20v2M2 9h2M2 15h2M20 9h2M20 15h2"/>',
    "layers": '<path d="M12 2 2 7l10 5 10-5-10-5Z"/><path d="M2 17l10 5 10-5"/><path d="M2 12l10 5 10-5"/>',
    "search": '<circle cx="11" cy="11" r="7"/><path d="m21 21-4.3-4.3"/>',
    "bulb": '<path d="M9 18h6M10 22h4"/><path d="M12 2a7 7 0 0 0-4 12.7c.6.5 1 1.3 1 2.3h6c0-1 .4-1.8 1-2.3A7 7 0 0 0 12 2Z"/>',
    "list": '<path d="M8 6h13M8 12h13M8 18h13M3 6h.01M3 12h.01M3 18h.01"/>',
    "bars": '<path d="M3 3v18h18"/><path d="M7 16v-5M12 16V7M17 16v-8"/>',
    "clock": '<circle cx="12" cy="12" r="10"/><path d="M12 6v6l4 2"/>',
    "gauge": '<path d="M12 14l4-4"/><path d="M3.3 19a10 10 0 1 1 17.4 0"/>',
    "file": '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8Z"/><path d="M14 2v6h6M8 13h8M8 17h8"/>',
    "alert": '<path d="M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0Z"/>'
             '<path d="M12 9v4M12 17h.01"/>',
    "check": '<circle cx="12" cy="12" r="10"/><path d="m8 12 3 3 5-6"/>',
    "scale": '<path d="M12 3v18M5 21h14"/><path d="m5 8-3 7a4 4 0 0 0 6 0Zm14 0-3 7a4 4 0 0 0 6 0Z"/><path d="M5 8h14"/>',
    "flask": '<path d="M9 2h6M10 2v6L4 19a2 2 0 0 0 1.8 3h12.4A2 2 0 0 0 20 19l-6-11V2"/><path d="M7 15h10"/>',
}
# emotion faces drawn as icons
_FACE = '<circle cx="12" cy="12" r="10"/>'
_E = {
    "sadness": _FACE + '<path d="M8 17c1-1.6 2.4-2.2 4-2.2s3 .6 4 2.2"/><path d="M9 10h.01M15 10h.01" stroke-width="3"/>',
    "joy": _FACE + '<path d="M8 14c1 1.8 2.4 2.6 4 2.6s3-.8 4-2.6"/><path d="M9 9.5h.01M15 9.5h.01" stroke-width="3"/>',
    "love": '<path d="M19 14c1.5-1.5 3-3.2 3-5.5A5.5 5.5 0 0 0 16.5 3c-1.8 0-3 .5-4.5 2-1.5-1.5-2.7-2-4.5-2A5.5 5.5 0 0 0 2 8.5c0 2.3 1.5 4 3 5.5l7 7Z"/>',
    "anger": _FACE + '<path d="m7 8 4 2M17 8l-4 2"/><path d="M8.5 17c1-1.4 2.2-2 3.5-2s2.5.6 3.5 2"/><path d="M9 12.5h.01M15 12.5h.01" stroke-width="3"/>',
    "fear": _FACE + '<path d="m7 7 3 1M17 7l-3 1"/><path d="M8 16.5c1-1 2-1 4 0s3 1 4 0"/><circle cx="9" cy="11.5" r="1.3"/><circle cx="15" cy="11.5" r="1.3"/>',
    "surprise": _FACE + '<circle cx="12" cy="16" r="2.2"/><path d="M9 9.5h.01M15 9.5h.01" stroke-width="3"/><path d="M8 6.5 9.5 7.5M16 6.5 14.5 7.5"/>',
}
COLOR = {"sadness": "#4F8CFF", "joy": "#F5A524", "love": "#F0568C", "anger": "#EF4444", "fear": "#8B5CF6", "surprise": "#14B8A6"}


def svg(inner, size=20, color="currentColor", sw=2):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
            f'stroke="{color}" stroke-width="{sw}" stroke-linecap="round" stroke-linejoin="round" '
            f'style="flex:none">{inner}</svg>')


def ico(name, size=20, color="currentColor", sw=2):
    return svg(_I[name], size, color, sw)


def face(emo, size=20, color="currentColor", sw=2):
    return svg(_E[emo], size, color, sw)


def head(icon, text, sub=None):
    s = f'<div class="sub2">{sub}</div>' if sub else ""
    st.markdown(f'<div class="sh">{ico(icon, 22, "var(--acc)")}<div><div class="t">{text}</div>{s}</div></div>',
                unsafe_allow_html=True)


def kpi(col, icon, label, value, color="var(--acc)"):
    col.markdown(f'<div class="kpi"><div class="ki" style="color:{color}">{ico(icon, 22)}</div>'
                 f'<div><div class="kl">{label}</div><div class="kv">{value}</div></div></div>', unsafe_allow_html=True)


EXAMPLES = [
    ("Happy", "i feel so happy and proud of what i achieved today", ":material/sentiment_very_satisfied:"),
    ("Sad", "i am alone and feel awful about everything", ":material/sentiment_dissatisfied:"),
    ("Angry", "i am furious at how they treated me", ":material/sentiment_very_dissatisfied:"),
    ("Scared", "i am terrified about what might happen tomorrow", ":material/warning:"),
    ("Loving", "i adore my family and feel so close to them", ":material/favorite:"),
    ("Shocked", "i was stunned by the sudden unexpected news", ":material/bolt:"),
    ("Negation", "i do not feel happy at all", ":material/block:"),
]

# ------------------------------------------------------------------ style
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
:root {--acc:#6d5efc; --card:rgba(128,128,128,.07); --line:rgba(128,128,128,.22);}
html, body, [class*="css"] {font-family: 'Inter', system-ui, sans-serif;}
#MainMenu, footer {visibility: hidden;}
.block-container {padding-top: 1.4rem; max-width: 1280px;}
.hero {position:relative; overflow:hidden; display:flex; gap:22px; align-items:center;
  background: linear-gradient(120deg,#17153b 0%,#2d2a7a 50%,#6d5efc 100%);
  padding: 28px 34px; border-radius: 22px; color:#fff; margin-bottom: 22px;
  box-shadow: 0 18px 40px -18px rgba(109,94,252,.65);}
.hero:after {content:""; position:absolute; right:-60px; top:-60px; width:260px; height:260px; border-radius:50%;
  background: radial-gradient(circle, rgba(255,255,255,.18), transparent 70%);}
.hero .logo {width:68px; height:68px; border-radius:18px; background:rgba(255,255,255,.14);
  display:flex; align-items:center; justify-content:center; flex:none; backdrop-filter: blur(6px);}
.hero h1 {margin:0; font-size:2rem; font-weight:700; letter-spacing:-.02em; color:#fff;}
.hero p {margin:4px 0 0; opacity:.85; color:#fff;}
.badges {display:flex; flex-wrap:wrap; gap:8px; margin-top:12px;}
.badge {display:inline-flex; gap:7px; align-items:center; padding:5px 12px; border-radius:999px;
  background:rgba(255,255,255,.15); font-size:.8rem; color:#fff;}
.sh {display:flex; gap:10px; align-items:center; margin: 18px 0 10px;}
.sh .t {font-size:1.15rem; font-weight:600;}
.sh .sub2 {font-size:.82rem; opacity:.65;}
.kpi {display:flex; gap:14px; align-items:center; padding:16px 18px; border-radius:16px;
  background:var(--card); border:1px solid var(--line); margin-bottom:10px;}
.kpi .ki {width:42px; height:42px; border-radius:12px; background:rgba(109,94,252,.13);
  display:flex; align-items:center; justify-content:center; flex:none;}
.kpi .kl {font-size:.78rem; opacity:.65; text-transform:uppercase; letter-spacing:.05em;}
.kpi .kv {font-size:1.45rem; font-weight:700; line-height:1.2;}
.result {display:flex; gap:20px; align-items:center; border-radius:20px; padding:24px 28px; color:#fff;
  box-shadow: 0 16px 34px -18px rgba(0,0,0,.5);}
.result .ricon {width:84px; height:84px; border-radius:50%; background:rgba(255,255,255,.22);
  display:flex; align-items:center; justify-content:center; flex:none;}
.result .big {font-size:2.5rem; font-weight:700; line-height:1.1; letter-spacing:-.02em;}
.result .sub {opacity:.92; font-size:.95rem;}
.tok {display:inline-block; padding:5px 11px; margin:3px; border-radius:9px; font-size:1rem;}
.stepbox {display:flex; gap:14px; border:1px solid var(--line); background:var(--card);
  border-radius:14px; padding:13px 16px; margin-bottom:10px;}
.stepbox .n {width:28px; height:28px; border-radius:50%; background:var(--acc); color:#fff; font-weight:600;
  display:flex; align-items:center; justify-content:center; flex:none; font-size:.85rem;}
.stepbox .lbl {font-size:.75rem; text-transform:uppercase; letter-spacing:.06em; opacity:.62; margin-bottom:3px;}
.chip {display:inline-block; padding:2px 10px; margin:2px; border-radius:999px; border:1px solid var(--line); font-size:.88rem;}
.chip.rm {text-decoration:line-through; opacity:.5;}
.legend {display:flex; gap:16px; font-size:.82rem; opacity:.8; margin: 2px 0 8px;}
.legend i {display:inline-block; width:12px; height:12px; border-radius:3px; margin-right:6px; vertical-align:-1px;}
button[kind="primary"] {border-radius:12px !important; font-weight:600 !important;}
button[data-baseweb="tab"] {font-weight:600;}
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading model...")
def get_engine():
    from backend import EmotionEngine
    return EmotionEngine()


try:
    eng = get_engine()
except FileNotFoundError as e:
    st.error(str(e), icon=":material/error:")
    st.code("python train_save.py", language="bash")
    st.stop()

M = eng.metrics
acc = M["report"]["accuracy"]

st.markdown(
    f"""
<div class="hero">
  <div class="logo">{ico("network", 38, "#fff", 1.8)}</div>
  <div>
    <h1>Emotion AI</h1>
    <p>Text emotion classifier: NLTK preprocessing, TF-IDF features and an Artificial Neural Network</p>
    <div class="badges">
      <span class="badge">{ico("layers", 15, "#fff")}6 emotions</span>
      <span class="badge">{ico("cpu", 15, "#fff")}{M['params']:,} parameters</span>
      <span class="badge">{ico("target", 15, "#fff")}{acc:.1%} test accuracy</span>
      <span class="badge">{ico("database", 15, "#fff")}{M['rows_after_dedupe']:,} training texts</span>
    </div>
  </div>
</div>""",
    unsafe_allow_html=True,
)

if "history" not in st.session_state:
    st.session_state.history = []
if "text" not in st.session_state:
    st.session_state.text = ""

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    ":material/search: Analyze", ":material/science: Pipeline Lab", ":material/folder_open: Batch",
    ":material/monitoring: Model Insights", ":material/history: History"])


def prob_chart(probs: dict):
    order = sorted(probs.items(), key=lambda kv: kv[1])
    fig = go.Figure(go.Bar(
        x=[v for _, v in order], y=[k.title() for k, _ in order], orientation="h",
        marker_color=[COLOR[k] for k, _ in order], text=[f"{v:.1%}" for _, v in order], textposition="outside"))
    fig.update_layout(height=300, margin=dict(l=0, r=30, t=10, b=0), xaxis=dict(range=[0, 1.15], tickformat=".0%"),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig


def gauge(conf, color):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=conf * 100, number={"suffix": "%", "valueformat": ".1f"},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": color, "thickness": .28}, "borderwidth": 0},
        title={"text": "Confidence"}))
    fig.update_layout(height=260, margin=dict(l=10, r=10, t=50, b=0), paper_bgcolor="rgba(0,0,0,0)")
    return fig


def transparent(fig, h):
    fig.update_layout(height=h, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig


# ================================================================ Analyze
with tab1:
    head("bulb", "Quick examples", "Click one to fill the text box")
    cols = st.columns(len(EXAMPLES))
    for c, (name, txt, icon) in zip(cols, EXAMPLES):
        if c.button(name, icon=icon, use_container_width=True, key=f"ex_{name}"):
            st.session_state.text = txt

    head("search", "Your text")
    text = st.text_area("Type or paste text", key="text", height=110, label_visibility="collapsed",
                        placeholder="e.g. i feel so happy and proud today")
    go_btn = st.button("Analyze emotion", type="primary", icon=":material/play_arrow:")

    if go_btn:
        if not text.strip():
            st.warning("Enter some text first.", icon=":material/warning:")
        else:
            r = eng.predict(text)
            emo, color = r["emotion"], COLOR[r["emotion"]]
            st.session_state.history.insert(0, {
                "time": datetime.now().strftime("%H:%M:%S"), "text": text,
                "emotion": emo, "confidence": round(r["confidence"], 4)})

            left, right = st.columns([1.15, 1])
            with left:
                st.markdown(
                    f"""<div class="result" style="background:linear-gradient(135deg,{color},{color}b0)">
                    <div class="ricon">{face(emo, 52, "#fff", 1.7)}</div>
                    <div><div class="sub">Predicted emotion</div>
                    <div class="big">{emo.title()}</div>
                    <div class="sub">{r['confidence']:.1%} confidence &nbsp;|&nbsp; cleaned text: <i>{html.escape(r['clean_text']) or '(empty)'}</i></div>
                    </div></div>""", unsafe_allow_html=True)
                head("bars", "Probability by emotion")
                st.plotly_chart(prob_chart(r["probs"]), use_container_width=True)
            with right:
                st.plotly_chart(gauge(r["confidence"], color), use_container_width=True)
                if r["known_features"] == 0:
                    st.warning("None of these words are in the model's 5000-feature vocabulary, so this prediction is unreliable.",
                               icon=":material/warning:")
                elif r["confidence"] < 0.6:
                    st.warning("Low confidence: the text is ambiguous, has no clear emotion word, or mixes emotions.",
                               icon=":material/warning:")
                elif r["confidence"] < 0.85:
                    st.info("Moderate confidence: check the word impact chart below. The model may be relying on a weak word.",
                            icon=":material/info:")
                else:
                    st.success("High confidence prediction.", icon=":material/check_circle:")

            head("bulb", "Why this prediction?",
                 "Each word is removed in turn and the change in the predicted emotion's probability is measured")
            imp = eng.word_importance(text)
            if imp:
                st.markdown('<div class="legend"><span><i style="background:rgba(34,197,94,.8)"></i>supports the prediction</span>'
                            '<span><i style="background:rgba(239,68,68,.8)"></i>works against it</span></div>',
                            unsafe_allow_html=True)
                mx = max(abs(i["impact"]) for i in imp) or 1
                spans = []
                for i in imp:
                    a = min(abs(i["impact"]) / mx, 1) * 0.7 + 0.12
                    rgb = "34,197,94" if i["impact"] >= 0 else "239,68,68"
                    spans.append(f'<span class="tok" style="background:rgba({rgb},{a:.2f})" '
                                 f'title="{i["impact"]:+.3f}">{html.escape(i["word"])}</span>')
                st.markdown("".join(spans), unsafe_allow_html=True)
                idf = pd.DataFrame(imp).sort_values("impact")
                fig = px.bar(idf, x="impact", y="word", orientation="h", color="impact",
                             color_continuous_scale=["#ef4444", "#d1d5db", "#22c55e"], color_continuous_midpoint=0)
                fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), coloraxis_showscale=False, yaxis_title=None,
                                  xaxis_title="Change in probability when the word is removed")
                st.plotly_chart(transparent(fig, max(220, 28 * len(idf))), use_container_width=True)

            tf = eng.top_features(text)
            if tf:
                head("list", "Top TF-IDF features of this input")
                st.dataframe(pd.DataFrame(tf), use_container_width=True, hide_index=True)

# ================================================================ Pipeline Lab
with tab2:
    head("flask", "Pipeline Lab", "See exactly how raw text becomes model input")
    lab = st.text_input("Text", "I'm NOT feeling great, the days were really awful!!", key="lab")
    if lab.strip():
        s = eng.pipeline_steps(lab)

        def box(n, label, body):
            st.markdown(f'<div class="stepbox"><div class="n">{n}</div><div><div class="lbl">{label}</div>{body}</div></div>',
                        unsafe_allow_html=True)

        box(1, "Raw input", html.escape(s["raw"]))
        box(2, "Lowercase", html.escape(s["lowercase"]))
        box(3, "Remove non-letters and collapse spaces", html.escape(s["letters_only"]))
        box(4, "Tokenize", "".join(f'<span class="chip">{html.escape(t)}</span>' for t in s["tokens"]))
        rm = set(s["stopwords_removed"])
        box(5, "Stopword removal (struck through; negations not/no/never/nor are kept)",
            "".join(f'<span class="chip {"rm" if t in rm else ""}">{html.escape(t)}</span>' for t in s["tokens"]))
        box(6, "Lemmatize",
            "".join(f'<span class="chip">{html.escape(a)} &rarr; <b>{html.escape(b)}</b></span>'
                    for a, b in zip(s["after_stopwords"], s["lemmatized"])))
        box(7, "Final clean_text", f"<b>{html.escape(s['clean_text']) or '(empty)'}</b>")
        tf = eng.top_features(lab, k=15)
        if tf:
            head("list", "TF-IDF vector (non-zero entries)",
                 "The vector has 5000 positions; only these are non-zero. Bigrams such as 'feel awful' are included.")
            st.dataframe(pd.DataFrame(tf), use_container_width=True, hide_index=True)

# ================================================================ Batch
with tab3:
    head("file", "Batch classification", "Upload a CSV or paste one text per line")
    up = st.file_uploader("CSV file", type="csv")
    pasted = st.text_area("Or paste texts (one per line)", height=130, key="paste")
    texts, src = [], None
    if up is not None:
        d = pd.read_csv(up)
        col = st.selectbox("Text column", d.columns, index=list(d.columns).index("text") if "text" in d.columns else 0)
        texts, src = d[col].astype(str).tolist(), d
    elif pasted.strip():
        texts = [t for t in pasted.splitlines() if t.strip()]
        src = pd.DataFrame({"text": texts})

    if st.button("Run batch", type="primary", icon=":material/play_arrow:") and texts:
        with st.spinner(f"Classifying {len(texts):,} texts..."):
            cleaned, P = eng.predict_many(texts)
        out = src.copy()
        out["clean_text"] = cleaned
        out["emotion"] = [M["emotions"][i] for i in P.argmax(1)]
        out["confidence"] = P.max(1).round(4)
        for j, e in enumerate(M["emotions"]):
            out[f"p_{e}"] = P[:, j].round(4)
        st.session_state.batch = out

    if "batch" in st.session_state:
        out = st.session_state.batch
        c1, c2 = st.columns([1, 1.3])
        cnt = out["emotion"].value_counts()
        fig = px.pie(values=cnt.values, names=cnt.index, hole=.55, color=cnt.index, color_discrete_map=COLOR)
        fig.update_layout(margin=dict(l=0, r=0, t=10, b=0))
        c1.plotly_chart(transparent(fig, 320), use_container_width=True)
        kpi(c2, "file", "Texts classified", f"{len(out):,}")
        kpi(c2, "gauge", "Average confidence", f"{out['confidence'].mean():.1%}")
        kpi(c2, "alert", "Low confidence (under 60%)", f"{(out['confidence'] < .6).sum():,}", "#F5A524")
        st.dataframe(out, use_container_width=True, height=330)
        st.download_button("Download results CSV", out.to_csv(index=False).encode(), "emotion_results.csv",
                           "text/csv", icon=":material/download:")

# ================================================================ Insights
with tab4:
    rep = M["report"]
    k = st.columns(4)
    kpi(k[0], "target", "Accuracy", f"{acc:.2%}")
    kpi(k[1], "scale", "Macro F1", f"{rep['macro avg']['f1-score']:.3f}")
    kpi(k[2], "database", "Train / Test", f"{M['train_size']:,} / {M['test_size']:,}")
    kpi(k[3], "cpu", "Parameters", f"{M['params']:,}")

    a, b = st.columns(2)
    with a:
        head("bars", "Per-class metrics")
        f1 = pd.DataFrame({"emotion": [e.title() for e in M["emotions"]],
                           "Precision": [rep[e]["precision"] for e in M["emotions"]],
                           "Recall": [rep[e]["recall"] for e in M["emotions"]],
                           "F1": [rep[e]["f1-score"] for e in M["emotions"]]}).melt("emotion", var_name="metric")
        fig = px.bar(f1, x="emotion", y="value", color="metric", barmode="group",
                     color_discrete_sequence=["#6d5efc", "#14B8A6", "#F5A524"])
        fig.update_layout(yaxis_range=[0.6, 1], xaxis_title=None, yaxis_title=None, margin=dict(t=10))
        st.plotly_chart(transparent(fig, 360), use_container_width=True)
    with b:
        head("layers", "Confusion matrix", "Row-normalised: how each true emotion was predicted")
        cm = np.array(M["confusion_matrix"])
        cmn = cm / cm.sum(1, keepdims=True)
        lbl = [e.title() for e in M["emotions"]]
        fig = px.imshow(cmn, x=lbl, y=lbl, text_auto=".0%", color_continuous_scale="Purples",
                        labels=dict(x="Predicted", y="True"))
        fig.update_layout(margin=dict(t=10), coloraxis_showscale=False)
        st.plotly_chart(transparent(fig, 360), use_container_width=True)

    h = M["history"]
    ep = list(range(1, len(h["loss"]) + 1))
    c, d = st.columns(2)
    with c:
        head("clock", "Training loss")
        fig = go.Figure([go.Scatter(x=ep, y=h["loss"], name="train", line=dict(color="#6d5efc", width=3)),
                         go.Scatter(x=ep, y=h["val_loss"], name="validation", line=dict(color="#F5A524", width=3))])
        fig.update_layout(xaxis_title="epoch", margin=dict(t=10))
        st.plotly_chart(transparent(fig, 300), use_container_width=True)
    with d:
        head("clock", "Training accuracy")
        fig = go.Figure([go.Scatter(x=ep, y=h["accuracy"], name="train", line=dict(color="#6d5efc", width=3)),
                         go.Scatter(x=ep, y=h["val_accuracy"], name="validation", line=dict(color="#F5A524", width=3))])
        fig.update_layout(xaxis_title="epoch", margin=dict(t=10))
        st.plotly_chart(transparent(fig, 300), use_container_width=True)

    head("database", "Class distribution")
    cc = M["class_counts"]
    fig = px.bar(x=[k.title() for k in cc], y=list(cc.values()), color=list(cc), color_discrete_map=COLOR,
                 labels=dict(x="", y="texts"))
    fig.update_layout(showlegend=False, margin=dict(t=10))
    st.plotly_chart(transparent(fig, 300), use_container_width=True)

    st.info("Limitations: TF-IDF ignores word order; the test set was also used for early stopping; "
            "love and surprise are the weakest classes because they have the fewest samples.", icon=":material/info:")

# ================================================================ History
with tab5:
    head("clock", "Session history", "Every analysis from the Analyze tab")
    if not st.session_state.history:
        st.info("No analyses yet. Run one in the Analyze tab.", icon=":material/info:")
    else:
        hist = pd.DataFrame(st.session_state.history)
        st.dataframe(hist, use_container_width=True, hide_index=True)
        b1, b2, _ = st.columns([1, 1, 4])
        b1.download_button("Download CSV", hist.to_csv(index=False).encode(), "history.csv", "text/csv",
                           icon=":material/download:")
        if b2.button("Clear", icon=":material/delete:"):
            st.session_state.history = []
            st.rerun()