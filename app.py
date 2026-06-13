"""Dialectal Arabic Sentiment Analyzer — Streamlit dashboard."""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.model import (
    DEFAULT_THRESHOLD, LABEL_AR, LABEL_EMOJI, MODEL_NAME, SentimentAnalyzer,
)
from src.preprocess import is_arabic
from src import dialect as dialect_mod

st.set_page_config(page_title="Dialectal Arabic Sentiment Analyzer",
                   page_icon="\U0001f4ac", layout="wide",
                   initial_sidebar_state="expanded")

PRIMARY = "#1e3a8a"; PRIMARY_LIGHT = "#3b82f6"; ACCENT = "#06b6d4"
SENTIMENT_COLORS = {"positive": "#16a34a", "negative": "#dc2626",
                    "neutral": "#64748b", "uncertain": "#f59e0b"}

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
.stApp {{ background:
    radial-gradient(1200px 500px at 100% -10%, #dbeafe55 0%, transparent 60%),
    linear-gradient(180deg,#f8fafc 0%,#eef2f7 100%); }}
html, body, [class*="css"] {{ font-family:'Inter',sans-serif; }}
.hero {{ position:relative; overflow:hidden;
    background: linear-gradient(125deg,{PRIMARY} 0%,{PRIMARY_LIGHT} 55%,{ACCENT} 130%);
    padding:2rem 2.2rem; border-radius:22px; color:white; margin-bottom:1rem;
    box-shadow:0 18px 40px rgba(30,58,138,.28); }}
.hero::after {{ content:""; position:absolute; right:-60px; top:-60px;
    width:230px; height:230px; border-radius:50%;
    background:rgba(255,255,255,.10); }}
.hero::before {{ content:""; position:absolute; right:60px; bottom:-90px;
    width:170px; height:170px; border-radius:50%;
    background:rgba(255,255,255,.07); }}
.hero h1 {{ color:white; margin:0 0 .4rem 0; font-size:2.05rem; font-weight:800;
    letter-spacing:-.5px; }}
.hero p {{ color:#dbeafe; margin:0; font-size:1.02rem; line-height:1.7;
    direction:rtl; max-width:900px; }}
.statbar {{ display:flex; gap:.8rem; flex-wrap:wrap; margin:-.2rem 0 1.2rem 0; }}
.stat {{ flex:1; min-width:150px; background:white; border:1px solid #e2e8f0;
    border-radius:16px; padding:.95rem 1.1rem;
    box-shadow:0 4px 14px rgba(15,23,42,.05);
    border-top:4px solid {PRIMARY_LIGHT}; }}
.stat .n {{ font-size:1.6rem; font-weight:800; color:{PRIMARY}; line-height:1; }}
.stat .t {{ font-size:.78rem; color:#64748b; font-weight:600; margin-top:.35rem;
    text-transform:uppercase; letter-spacing:.03em; }}
.stat .a {{ font-size:.74rem; color:#94a3b8; direction:rtl; }}
.card {{ background:white; border:1px solid #e2e8f0; border-radius:16px;
    padding:1.15rem 1.3rem; box-shadow:0 4px 14px rgba(15,23,42,.05); height:100%; }}
.card .label {{ color:#64748b; font-size:.8rem; font-weight:700;
    text-transform:uppercase; letter-spacing:.04em; }}
.card .value {{ color:{PRIMARY}; font-size:1.75rem; font-weight:800; margin-top:.3rem; }}
.card .sub {{ color:#94a3b8; font-size:.8rem; margin-top:.2rem; direction:rtl; }}
.pill {{ display:inline-block; padding:.4rem 1rem; border-radius:999px;
    font-weight:800; font-size:1rem; color:white; }}
.dialect-banner {{ background:linear-gradient(90deg,#eff6ff,#e0f2fe);
    border:1px solid #bfdbfe; border-left:5px solid {ACCENT}; border-radius:14px;
    padding:.9rem 1.2rem; color:{PRIMARY}; font-weight:700; direction:rtl;
    text-align:right; font-size:1.05rem; }}
.stTabs [data-baseweb="tab-list"] {{ gap:6px; }}
.stTabs [data-baseweb="tab"] {{ border-radius:12px 12px 0 0; padding:9px 20px;
    font-weight:700; background:#f1f5f9; }}
.stTabs [aria-selected="true"] {{ background:white; color:{PRIMARY}!important; }}
div.stButton > button {{ background:linear-gradient(90deg,{PRIMARY},{PRIMARY_LIGHT});
    color:white; border:none; border-radius:12px; padding:.55rem 1.6rem;
    font-weight:700; box-shadow:0 6px 16px rgba(59,130,246,.35); }}
div.stButton > button:hover {{ filter:brightness(1.08); }}
section[data-testid="stSidebar"] {{ background:#0f172a; }}
section[data-testid="stSidebar"] * {{ color:#e2e8f0; }}
section[data-testid="stSidebar"] code {{ color:#7dd3fc; background:#1e293b; }}
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading CAMeLBERT-DA model (first run downloads ~500MB)...")
def load_analyzer(): return SentimentAnalyzer()


@st.cache_resource(show_spinner=False)
def load_dialect(): return dialect_mod.load()


def card(label, value, sub=""):
    sub_html = f'<div class="sub">{sub}</div>' if sub else ""
    return (f'<div class="card"><div class="label">{label}</div>'
            f'<div class="value">{value}</div>{sub_html}</div>')


def gauge(score):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=score*100,
        number={"suffix": "%", "font": {"size": 40, "color": PRIMARY}},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": PRIMARY},
               "steps": [{"range": [0, 40], "color": "#fee2e2"},
                         {"range": [40, 70], "color": "#fef3c7"},
                         {"range": [70, 100], "color": "#dcfce7"}],
               "threshold": {"line": {"color": ACCENT, "width": 5},
                             "thickness": 0.85, "value": score*100}}))
    fig.update_layout(height=250, margin=dict(t=20, b=10, l=20, r=20))
    return fig


def render_single_result(result, threshold):
    emoji = LABEL_EMOJI.get(result.label, "\u2753")
    color = SENTIMENT_COLORS.get(result.display_label, "#64748b")
    c1, c2, c3 = st.columns(3)
    c1.markdown(card("Sentiment \u00b7 \u0627\u0644\u0645\u0634\u0627\u0639\u0631",
        f'<span class="pill" style="background:{color}">{emoji} {result.label.title()}</span>',
        LABEL_AR.get(result.label, "")), unsafe_allow_html=True)
    c2.markdown(card("Confidence \u00b7 \u0627\u0644\u062b\u0642\u0629", f"{result.confidence:.0%}"), unsafe_allow_html=True)
    c3.markdown(card("Status \u00b7 \u0627\u0644\u062d\u0627\u0644\u0629",
        "\u2705 Confident" if result.is_confident else "\u26a0\ufe0f Uncertain",
        "\u0645\u0648\u062b\u0648\u0642" if result.is_confident else "\u064a\u062d\u062a\u0627\u062c \u0645\u0631\u0627\u062c\u0639\u0629"), unsafe_allow_html=True)
    if not result.is_confident:
        st.warning(f"Confidence {result.confidence:.0%} < {threshold:.0%} \u2014 \u063a\u064a\u0631 \u0645\u0624\u0643\u062f.")
    st.write("")
    prob_df = pd.DataFrame({"Sentiment": list(result.probabilities.keys()),
                            "Probability": list(result.probabilities.values())}
                           ).sort_values("Probability")
    fig = px.bar(prob_df, x="Probability", y="Sentiment", orientation="h",
                 color="Sentiment", color_discrete_map=SENTIMENT_COLORS,
                 range_x=[0, 1], text_auto=".0%")
    fig.update_layout(showlegend=False, height=240, plot_bgcolor="white",
                      title="Class probabilities \u00b7 \u062a\u0648\u0632\u064a\u0639 \u0627\u0644\u0627\u062d\u062a\u0645\u0627\u0644\u0627\u062a",
                      margin=dict(t=40, b=10))
    st.plotly_chart(fig, use_container_width=True)


def main():
    st.markdown("""
    <div class="hero">
      <h1>\U0001f4ac Dialectal Arabic Sentiment Analyzer</h1>
      <p>\u062a\u062d\u0644\u064a\u0644 \u0627\u0644\u0645\u0634\u0627\u0639\u0631 \u0644\u0644\u0647\u062c\u0627\u062a \u0627\u0644\u0639\u0631\u0628\u064a\u0629 \u0627\u0644\u0639\u0627\u0645\u064a\u0629 + \u0643\u0634\u0641 \u0627\u0644\u0644\u0647\u062c\u0629\u060c \u0645\u0628\u0646\u064a \u0639\u0644\u0649 \u0646\u0645\u0648\u0630\u062c CAMeLBERT \u062a\u0645 \u062a\u062f\u0631\u064a\u0628\u0647 \u0639\u0644\u0649 ArSarcasm-v2 \u0645\u0639 \u062a\u062d\u0644\u064a\u0644 \u0644\u0643\u0644 \u0644\u0647\u062c\u0629 \u0648\u0644\u0648\u062d\u0629 \u0631\u0636\u0627 \u062a\u0641\u0627\u0639\u0644\u064a\u0629.</p>
    </div>""", unsafe_allow_html=True)

    ft = "finetuned" in MODEL_NAME
    st.markdown(f"""
    <div class="statbar">
      <div class="stat"><div class="n">15.5K</div><div class="t">Labeled tweets</div><div class="a">\u062a\u063a\u0631\u064a\u062f\u0629 \u0645\u0635\u0646\u0651\u0641\u0629</div></div>
      <div class="stat"><div class="n">CAMeLBERT</div><div class="t">Fine-tuned</div><div class="a">\u0646\u0645\u0648\u0630\u062c \u0645\u062f\u0631\u0651\u0628</div></div>
      <div class="stat"><div class="n">+2.2</div><div class="t">F1 gain (fine-tune)</div><div class="a">\u062a\u062d\u0633\u0651\u0646 \u0628\u0639\u062f \u0627\u0644\u062a\u062f\u0631\u064a\u0628</div></div>
      <div class="stat"><div class="n">4</div><div class="t">Dialects</div><div class="a">\u0644\u0647\u062c\u0627\u062a \u0645\u062f\u0639\u0648\u0645\u0629</div></div>
    </div>""", unsafe_allow_html=True)

    with st.sidebar:
        st.header("\u2699\ufe0f Settings \u00b7 \u0627\u0644\u0625\u0639\u062f\u0627\u062f\u0627\u062a")
        threshold = st.slider("Confidence threshold \u00b7 \u0639\u062a\u0628\u0629 \u0627\u0644\u062b\u0642\u0629",
                              0.50, 0.95, DEFAULT_THRESHOLD, 0.05)
        st.divider()
        st.subheader("Model \u00b7 \u0627\u0644\u0646\u0645\u0648\u0630\u062c")
        st.code(MODEL_NAME, language=None)
        if ft:
            st.success("\u2705 \u0627\u0644\u0646\u0645\u0648\u0630\u062c **\u0627\u0644\u0645\u062f\u0631\u064e\u0651\u0628** (ArSarcasm-v2)")
        else:
            st.info("\u0627\u0644\u0646\u0645\u0648\u0630\u062c \u0627\u0644\u0623\u0633\u0627\u0633\u064a. \u0634\u063a\u0651\u0644\u064a train.py.")
        st.caption("3-class \u00b7 positive / negative / neutral")

    analyzer = load_analyzer()
    dialect_model = load_dialect()

    t1, t2, t3 = st.tabs(["\U0001f50d Single Text \u00b7 \u0646\u0635 \u0648\u0627\u062d\u062f",
                          "\U0001f4ca Batch Analysis \u00b7 \u062a\u062d\u0644\u064a\u0644 \u062c\u0645\u0627\u0639\u064a",
                          "\u2139\ufe0f About \u00b7 \u0639\u0646 \u0627\u0644\u0645\u0634\u0631\u0648\u0639"])

    with t1:
        examples = {
            "\u2014 pick an example \u00b7 \u0627\u062e\u062a\u0631 \u0645\u062b\u0627\u0644 \u2014": "",
            "\U0001f334 Saudi (positive)": "\u0627\u0644\u062a\u0637\u0628\u064a\u0642 \u0645\u0631\u0647 \u062d\u0644\u0648 \u0648\u064a\u062c\u0646\u0646\u060c \u0633\u0647\u0651\u0644 \u0639\u0644\u064a \u0643\u0644 \u0634\u064a \U0001f44c",
            "\U0001f334 Saudi (negative)": "\u0627\u0644\u062e\u062f\u0645\u0629 \u0632\u0641\u062a \u0648\u0627\u0644\u062a\u0648\u0635\u064a\u0644 \u062a\u0623\u062e\u0631 \u0627\u0633\u0628\u0648\u0639 \u0643\u0627\u0645\u0644 \U0001f621",
            "\U0001f334 Gulf (neutral)": "\u0627\u0633\u062a\u0644\u0645\u062a \u0627\u0644\u0637\u0644\u0628 \u0627\u0644\u064a\u0648\u0645 \u0648\u0628\u0627\u062c\u0631\u0628\u0647 \u0648\u0627\u0634\u0648\u0641",
            "\U0001f1ea\U0001f1ec Egyptian (positive)": "\u0627\u0644\u0641\u064a\u0644\u0645 \u062f\u0647 \u062a\u062d\u0641\u0629 \u0628\u062c\u062f \u0648\u0636\u062d\u0643\u0646\u064a \u0645\u0646 \u0642\u0644\u0628\u064a",
        }
        choice = st.selectbox("Try an example \u00b7 \u062c\u0631\u0651\u0628\u064a \u0645\u062b\u0627\u0644:", list(examples.keys()))
        text = st.text_area("Arabic text \u00b7 \u0627\u0644\u0646\u0635 \u0627\u0644\u0639\u0631\u0628\u064a", value=examples[choice],
                            height=120, placeholder="\u0627\u0643\u062a\u0628\u064a \u0646\u0635 \u0628\u0627\u0644\u0639\u0627\u0645\u064a\u0629 \u0647\u0646\u0627...")
        if st.button("Analyze \u00b7 \u062d\u0644\u0651\u0644", type="primary", disabled=not text.strip()):
            if not is_arabic(text):
                st.error("\u0627\u0644\u0646\u0635 \u063a\u064a\u0631 \u0639\u0631\u0628\u064a \u0628\u0645\u0639\u0638\u0645\u0647.")
            else:
                if dialect_model is not None:
                    d, dconf = dialect_mod.predict(dialect_model, text)
                    d_ar = dialect_mod.DIALECT_AR.get(d, d)
                    d_em = dialect_mod.DIALECT_EMOJI.get(d, "\U0001f5e3\ufe0f")
                    st.markdown(f'<div class="dialect-banner">{d_em} \u0627\u0644\u0644\u0647\u062c\u0629 \u0627\u0644\u0645\u0643\u062a\u0634\u0641\u0629 \u00b7 Detected dialect: '
                                f'<b>{d.title()} / {d_ar}</b> \u2014 {dconf:.0%}</div>', unsafe_allow_html=True)
                    st.write("")
                render_single_result(analyzer.predict(text, threshold=threshold), threshold)

    with t2:
        st.markdown("##### Upload reviews \u2192 satisfaction analytics")
        st.caption("\u0627\u0631\u0641\u0639\u064a CSV \u0641\u064a\u0647 \u0639\u0645\u0648\u062f text \u2014 \u062c\u0631\u0651\u0628\u064a data/restaurant_reviews.csv")
        up = st.file_uploader("Upload CSV \u00b7 \u0627\u0631\u0641\u0639 \u0627\u0644\u0645\u0644\u0641", type=["csv"])
        if up is not None:
            try:
                df = pd.read_csv(up)
            except Exception as e:
                st.error(f"\u062a\u0639\u0630\u0631\u062a \u0627\u0644\u0642\u0631\u0627\u0621\u0629: {e}"); st.stop()
            if "text" not in df.columns:
                st.error(f"\u0644\u0627\u0632\u0645 \u0639\u0645\u0648\u062f text. \u0627\u0644\u0623\u0639\u0645\u062f\u0629: {list(df.columns)}"); st.stop()
            df = df.dropna(subset=["text"]).reset_index(drop=True)
            with st.spinner(f"Classifying {len(df)} rows..."):
                res = analyzer.predict_batch(df["text"].tolist(), threshold=threshold)
            df["sentiment"] = [r.label for r in res]
            df["confidence"] = [round(r.confidence, 4) for r in res]
            df["flag"] = ["ok" if r.is_confident else "uncertain" for r in res]
            df["display"] = [r.display_label for r in res]
            n_pos = int((df["display"] == "positive").sum())
            n_neg = int((df["display"] == "negative").sum())
            n_neu = int((df["display"] == "neutral").sum())
            n_unc = int((df["flag"] == "uncertain").sum())
            opin = n_pos + n_neg
            sat = (n_pos/opin) if opin else None
            k1, k2, k3, k4 = st.columns(4)
            k1.markdown(card("Reviews \u00b7 \u062a\u0642\u064a\u064a\u0645\u0627\u062a", str(len(df))), unsafe_allow_html=True)
            k2.markdown(card("Positive \u00b7 \u0625\u064a\u062c\u0627\u0628\u064a", f'<span style="color:{SENTIMENT_COLORS["positive"]}">\U0001f60a {n_pos}</span>'), unsafe_allow_html=True)
            k3.markdown(card("Negative \u00b7 \u0633\u0644\u0628\u064a", f'<span style="color:{SENTIMENT_COLORS["negative"]}">\U0001f620 {n_neg}</span>'), unsafe_allow_html=True)
            k4.markdown(card("Uncertain \u00b7 \u063a\u064a\u0631 \u0645\u0624\u0643\u062f", f'<span style="color:{SENTIMENT_COLORS["uncertain"]}">\u26a0\ufe0f {n_unc}</span>'), unsafe_allow_html=True)
            st.write("")
            lft, rgt = st.columns(2)
            with lft:
                st.markdown("###### \u2b50 Customer Satisfaction \u00b7 \u0646\u0633\u0628\u0629 \u0627\u0644\u0631\u0636\u0627")
                if sat is not None:
                    st.plotly_chart(gauge(sat), use_container_width=True)
                    st.caption("positive \u00f7 (positive+negative)")
                else:
                    st.info("\u0644\u0627 \u062a\u0648\u062c\u062f \u0622\u0631\u0627\u0621 \u0648\u0627\u0636\u062d\u0629.")
            with rgt:
                st.markdown("###### Distribution \u00b7 \u0627\u0644\u062a\u0648\u0632\u064a\u0639")
                dist = pd.DataFrame({"Sentiment": ["positive", "negative", "neutral", "uncertain"],
                                     "Count": [n_pos, n_neg, n_neu, n_unc]})
                dist = dist[dist["Count"] > 0]
                pie = px.pie(dist, names="Sentiment", values="Count", color="Sentiment",
                             color_discrete_map=SENTIMENT_COLORS, hole=0.55)
                pie.update_layout(height=250, margin=dict(t=10, b=10),
                                  legend=dict(orientation="h", y=-0.1))
                st.plotly_chart(pie, use_container_width=True)
            st.markdown("###### Confidence distribution \u00b7 \u062a\u0648\u0632\u064a\u0639 \u0627\u0644\u062b\u0642\u0629")
            h = px.histogram(df, x="confidence", color="sentiment", nbins=20,
                             color_discrete_map=SENTIMENT_COLORS)
            h.add_vline(x=threshold, line_dash="dash", annotation_text="threshold")
            h.update_layout(height=250, margin=dict(t=10, b=10), plot_bgcolor="white")
            st.plotly_chart(h, use_container_width=True)
            st.markdown("###### Results \u00b7 \u0627\u0644\u0646\u062a\u0627\u0626\u062c")
            only_u = st.toggle("Show uncertain only \u00b7 \u063a\u064a\u0631 \u0627\u0644\u0645\u0624\u0643\u062f\u0629 \u0641\u0642\u0637")
            view = df[df["flag"] == "uncertain"] if only_u else df
            st.dataframe(view.drop(columns=["display"]), use_container_width=True, height=320)
            st.download_button("\u2b07\ufe0f Download CSV \u00b7 \u062d\u0645\u0651\u0644 \u0627\u0644\u0646\u062a\u0627\u0626\u062c",
                               df.drop(columns=["display"]).to_csv(index=False).encode("utf-8-sig"),
                               file_name="sentiment_results.csv", mime="text/csv")

    with t3:
        st.markdown("""
##### Why \u00b7 \u0644\u0645\u0627\u0630\u0627
\u0623\u062f\u0648\u0627\u062a \u0627\u0644\u0645\u0634\u0627\u0639\u0631 \u0627\u0644\u0639\u0631\u0628\u064a\u0629 \u0645\u062f\u0631\u0628\u0629 \u0639\u0644\u0649 \u0627\u0644\u0641\u0635\u062d\u0649\u060c \u0644\u0643\u0646 \u0627\u0644\u0646\u0627\u0633 \u064a\u0643\u062a\u0628\u0648\u0646 \u0628\u0627\u0644\u0639\u0627\u0645\u064a\u0629.

##### Pipeline \u00b7 \u0627\u0644\u0645\u0639\u0645\u0627\u0631\u064a\u0629
1. Light preprocessing \u2014 \u062a\u0646\u0638\u064a\u0641 \u062e\u0641\u064a\u0641.
2. CAMeLBERT-DA \u2014 \u0645\u0636\u0628\u0648\u0637 \u0639\u0644\u0649 ArSarcasm.
3. Dialect detector \u2014 TF-IDF.
4. Confidence thresholding.

##### Stack
PyTorch \u00b7 Transformers \u00b7 Streamlit \u00b7 Plotly \u00b7 scikit-learn \u00b7 FastAPI
""")


if __name__ == "__main__":
    main()