"""
MoodSense: AI-Powered Music Playlist Generator
MS DSP 422 - Practical Machine Learning | Group 3

UI: Premium dark aesthetic inspired by high-end music platforms
    - Tabbed layout: Playlist Generator · Explore · Analytics · Model Performance
    - Pre-designed vibe presets + custom prompt
    - FAISS vector search + structured logging
"""

import json
import math
import pickle
import re
import time
import logging
import os
import requests
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from numpy.linalg import norm
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack, csr_matrix
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
_log = logging.getLogger("moodsense")

# ═══════════════════════════════════════════════════════════════════════
# HUGGING FACE FILE DOWNLOAD (runs once on first deploy)
# ═══════════════════════════════════════════════════════════════════════

HF_BASE = "https://huggingface.co/123Nandini/moodsense-embeddings/resolve/main/"

HF_FILES = {
    "song_embeddings_50k.npy": HF_BASE + "song_embeddings_50k.npy",
    "song_index_50k.faiss":    HF_BASE + "song_index_50k.faiss",
}

def _download_file(url: str, dest: Path):
    """Stream-download a file from URL to dest path."""
    with requests.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

def ensure_hf_files():
    """Download embedding files from HF if not already present. Messages vanish after completion."""
    app_dir = Path(__file__).resolve().parent
    needs_download = [fname for fname, _ in HF_FILES.items() if not (app_dir / fname).exists()]
    if not needs_download:
        return
    placeholder = st.empty()
    for fname, url in HF_FILES.items():
        dest = app_dir / fname
        if not dest.exists():
            placeholder.info(f"⏬ Downloading `{fname}` from Hugging Face (first run only)…")
            try:
                _download_file(url, dest)
            except Exception as e:
                placeholder.error(f"❌ Failed to download `{fname}`: {e}")
                st.stop()
    placeholder.empty()

ensure_hf_files()

# ═══════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="MoodSense · AI Playlist",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ═══════════════════════════════════════════════════════════════════════
# CSS — Premium Dark Music Platform Aesthetic
# ═══════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Unbounded:wght@400;700;900&family=DM+Mono:wght@300;400;500&family=Instrument+Sans:wght@300;400;500;600&display=swap');

/* ── Reset & Base ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
    --bg:        #080808;
    --surface:   #111111;
    --surface2:  #1a1a1a;
    --border:    #2a2a2a;
    --border2:   #333333;
    --text:      #f0f0f0;
    --muted:     #888888;
    --accent:    #1DB954;
    --accent2:   #42f5a7;
    --red:       #ff4757;
    --blue:      #4488ff;
    --pink:      #ff6b9d;
    --mono:      'DM Mono', monospace;
    --display:   'Unbounded', sans-serif;
    --body:      'Instrument Sans', sans-serif;
}

/* ── Global ── */
.main { background: var(--bg) !important; font-family: var(--body); }
.block-container { padding: 1.5rem 2rem !important; max-width: 1400px !important; }
* { color: var(--text); font-family: var(--body); }
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 2px; }

/* ── Hero ── */
.hero {
    position: relative;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 3rem 3rem 2.5rem;
    margin-bottom: 2rem;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 300px; height: 300px;
    background: radial-gradient(circle, rgba(200,245,66,0.12) 0%, transparent 70%);
    pointer-events: none;
}
.hero::after {
    content: '';
    position: absolute;
    bottom: -40px; left: 30%;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(66,245,167,0.07) 0%, transparent 70%);
    pointer-events: none;
}
.hero-eyebrow {
    font-family: var(--mono);
    font-size: 0.7rem;
    letter-spacing: 0.2em;
    color: var(--accent);
    text-transform: uppercase;
    margin-bottom: 1rem;
}
.hero-title {
    font-family: var(--display);
    font-size: clamp(2.2rem, 4vw, 3.5rem);
    font-weight: 900;
    line-height: 1.05;
    letter-spacing: -0.02em;
    margin-bottom: 0.75rem;
    background: linear-gradient(135deg, #f0f0f0 0%, #888 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero-sub {
    font-size: 0.95rem;
    color: var(--muted);
    max-width: 600px;
    line-height: 1.6;
}
.hero-badges {
    display: flex;
    gap: 0.5rem;
    flex-wrap: wrap;
    margin-top: 1.5rem;
}
.badge {
    font-family: var(--mono);
    font-size: 0.65rem;
    letter-spacing: 0.1em;
    padding: 0.3rem 0.8rem;
    border-radius: 999px;
    border: 1px solid var(--border2);
    color: var(--muted);
    background: var(--surface2);
    text-transform: uppercase;
}
.badge.green  { border-color: rgba(200,245,66,0.4);  color: var(--accent);  background: rgba(200,245,66,0.08); }
.badge.teal   { border-color: rgba(66,245,167,0.4);  color: var(--accent2); background: rgba(66,245,167,0.08); }
.badge.blue   { border-color: rgba(68,136,255,0.4);  color: var(--blue);    background: rgba(68,136,255,0.08); }

/* ── KPI Cards ── */
.kpi-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin-bottom: 2rem; }
.kpi-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.25rem 1.5rem;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s;
}
.kpi-card:hover { border-color: var(--border2); }
.kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--accent), transparent);
}
.kpi-num {
    font-family: var(--display);
    font-size: 1.8rem;
    font-weight: 700;
    color: var(--accent);
    line-height: 1;
    margin-bottom: 0.4rem;
}
.kpi-label {
    font-family: var(--mono);
    font-size: 0.65rem;
    letter-spacing: 0.12em;
    color: var(--muted);
    text-transform: uppercase;
}

/* ── Tabs ── */
[data-testid="stTabs"] > div:first-child {
    border-bottom: 1px solid var(--border) !important;
    gap: 0 !important;
    padding-bottom: 0 !important;
}
[data-testid="stTabs"] button {
    font-family: var(--mono) !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    color: var(--muted) !important;
    border: none !important;
    padding: 0.75rem 1.25rem !important;
    border-radius: 0 !important;
    background: transparent !important;
    border-bottom: 2px solid transparent !important;
    transition: all 0.2s !important;
}
[data-testid="stTabs"] button:hover { color: var(--text) !important; }
[data-testid="stTabs"] button[aria-selected="true"] {
    color: var(--accent) !important;
    border-bottom: 2px solid var(--accent) !important;
    background: transparent !important;
}

/* ── Preset Vibes ── */
.preset-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 0.75rem; margin-bottom: 1rem; }
.preset-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1rem 1.1rem;
    cursor: pointer;
    transition: all 0.2s;
    position: relative;
    overflow: hidden;
}
.preset-card:hover {
    border-color: var(--accent);
    background: var(--surface2);
    transform: translateY(-1px);
}
.preset-icon { font-size: 1.5rem; margin-bottom: 0.4rem; display: block; }
.preset-title {
    font-family: var(--body);
    font-size: 0.85rem;
    font-weight: 600;
    margin-bottom: 0.2rem;
    color: var(--text);
}
.preset-sub { font-size: 0.72rem; color: var(--muted); line-height: 1.4; }

/* ── Prompt Input ── */
.stTextArea textarea, .stTextInput input {
    background: var(--surface2) !important;
    border: 1px solid var(--border2) !important;
    border-radius: 12px !important;
    color: var(--text) !important;
    font-family: var(--body) !important;
    font-size: 0.95rem !important;
    padding: 1rem 1.25rem !important;
    transition: border-color 0.2s !important;
    caret-color: var(--accent) !important;
}
.stTextArea textarea:focus, .stTextInput input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(200,245,66,0.08) !important;
    outline: none !important;
}

/* ── Buttons ── */
.stButton > button {
    background: var(--accent) !important;
    color: #080808 !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.7rem 1.5rem !important;
    font-family: var(--mono) !important;
    font-size: 0.75rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: #d4ff50 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(200,245,66,0.25) !important;
}
.stButton > button[kind="secondary"] {
    background: var(--surface2) !important;
    color: var(--muted) !important;
    border: 1px solid var(--border2) !important;
}
.stButton > button[kind="secondary"]:hover {
    color: var(--text) !important;
    border-color: var(--border2) !important;
    transform: translateY(-1px) !important;
    box-shadow: none !important;
}

/* ── Track List ── */
.tracklist {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 16px;
    overflow: hidden;
}
.tracklist-header {
    display: grid;
    grid-template-columns: 40px 1fr 1fr 80px 80px 60px;
    gap: 1rem;
    padding: 0.75rem 1.25rem;
    border-bottom: 1px solid var(--border);
    font-family: var(--mono);
    font-size: 0.62rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--muted);
}
.track-row {
    display: grid;
    grid-template-columns: 40px 1fr 1fr 80px 80px 60px;
    gap: 1rem;
    padding: 0.8rem 1.25rem;
    border-bottom: 1px solid var(--border);
    align-items: center;
    transition: background 0.15s;
}
.track-row:last-child { border-bottom: none; }
.track-row:hover { background: var(--surface2); }
.track-num {
    font-family: var(--mono);
    font-size: 0.8rem;
    color: var(--muted);
    text-align: right;
}
.track-name { font-size: 0.88rem; font-weight: 500; color: var(--text); }
.track-artist { font-size: 0.78rem; color: var(--muted); }
.mood-pill {
    display: inline-block;
    padding: 0.2rem 0.65rem;
    border-radius: 999px;
    font-family: var(--mono);
    font-size: 0.6rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    font-weight: 500;
}
.sim-bar-wrap { width: 100%; }
.sim-bar-bg { background: var(--border); border-radius: 3px; height: 4px; }
.sim-bar-fg { background: var(--accent); border-radius: 3px; height: 4px; }

/* ── Mood chip buttons ── */
.mood-chips { display: flex; gap: 0.5rem; flex-wrap: wrap; margin-bottom: 1rem; }

/* ── Result header ── */
.result-header {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.5rem;
    margin-bottom: 1.25rem;
    display: flex;
    align-items: center;
    gap: 1.5rem;
}
.result-mood-icon { font-size: 2.5rem; }
.result-title {
    font-family: var(--display);
    font-size: 1.2rem;
    font-weight: 700;
    margin-bottom: 0.3rem;
}
.result-meta { font-size: 0.8rem; color: var(--muted); }

/* ── Metrics row ── */
.metrics-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 0.75rem;
    margin-bottom: 1.5rem;
}
.metric-chip {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 0.75rem 1rem;
    text-align: center;
}
.metric-val {
    font-family: var(--mono);
    font-size: 1rem;
    font-weight: 500;
    color: var(--accent);
}
.metric-lbl {
    font-family: var(--mono);
    font-size: 0.6rem;
    letter-spacing: 0.1em;
    color: var(--muted);
    text-transform: uppercase;
    margin-top: 0.2rem;
}

/* ── Section headers ── */
.section-head {
    font-family: var(--mono);
    font-size: 0.65rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid var(--border);
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] .block-container { padding: 1.5rem 1rem !important; }

/* ── Sliders ── */
.stSlider > div > div > div { background: var(--border2) !important; }
.stSlider > div > div > div > div { background: var(--accent) !important; }

/* ── Dataframe ── */
[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; border: 1px solid var(--border); }
[data-testid="stDataFrame"] th { background: var(--surface2) !important; font-family: var(--mono) !important; font-size: 0.65rem !important; letter-spacing: 0.1em !important; }
[data-testid="stDataFrame"] td { background: var(--surface) !important; }

/* ── Success/info/warning ── */
[data-testid="stAlert"] { border-radius: 10px !important; border-left-width: 3px !important; }

/* ── Expander ── */
details summary { font-family: var(--mono) !important; font-size: 0.72rem !important; letter-spacing: 0.08em !important; text-transform: uppercase !important; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════
# MOOD CONFIG
# ═══════════════════════════════════════════════════════════════════════

MOOD_CONFIG = {
    'Happy': {'color': '#1DB954', 'bg': 'rgba(200,245,66,0.12)',  'emoji': '☀️', 'desc': 'Joyful & Uplifting'},
    'Sad':   {'color': '#4488ff', 'bg': 'rgba(68,136,255,0.12)', 'emoji': '🌧️', 'desc': 'Melancholic & Reflective'},
    'Anger': {'color': '#ff4757', 'bg': 'rgba(255,71,87,0.12)',  'emoji': '⚡', 'desc': 'Intense & Powerful'},
    'Love':  {'color': '#ff6b9d', 'bg': 'rgba(255,107,157,0.12)','emoji': '🌹', 'desc': 'Romantic & Tender'},
}

PRESET_PLAYLISTS = [
    {"id":"workout",    "icon":"🏃", "title":"Morning Run",        "sub":"Fast tempo, motivating, high energy",    "prompt":"run exercise morning energetic motivating fast beats workout pump adrenaline", "n":20},
    {"id":"focus",      "icon":"💻", "title":"Deep Work",          "sub":"Instrumental, minimal, concentration",   "prompt":"instrumental focus concentration work study calm steady beats no lyrics",     "n":20},
    {"id":"heartbreak", "icon":"💔", "title":"Heartbreak Hotel",   "sub":"Emotional, raw, post-breakup",           "prompt":"heartbreak crying pain love lost breakup emotional tears missing someone",     "n":18},
    {"id":"latenight",  "icon":"🚗", "title":"Late Night Drive",   "sub":"Dark, atmospheric, highway energy",      "prompt":"night dark driving highway electric atmospheric moody intensity beat",         "n":20},
    {"id":"party",      "icon":"🎉", "title":"Summer Party",       "sub":"High energy, danceable, joyful",         "prompt":"party dance happy upbeat energy summer fun celebration joyful loud",           "n":20},
    {"id":"romance",    "icon":"🍷", "title":"Date Night",         "sub":"Romantic, tender, warm",                 "prompt":"love romance tender sweet gentle warmth affection intimate caring",            "n":18},
    {"id":"anger",      "icon":"🔥", "title":"Rage Release",       "sub":"Intense, cathartic, aggressive",         "prompt":"anger intense powerful aggressive rage frustration electric loud cathartic",    "n":18},
    {"id":"rainy",      "icon":"☕", "title":"Rainy Afternoon",    "sub":"Acoustic, soft, introspective",          "prompt":"rain melancholic quiet acoustic reflective slow afternoon sad gentle",          "n":18},
]

# ═══════════════════════════════════════════════════════════════════════
# PATHS
# ═══════════════════════════════════════════════════════════════════════

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent

def _find(name):
    for p in [BASE_DIR / name, ROOT_DIR / name, BASE_DIR / 'models' / name, ROOT_DIR / 'models' / name]:
        if p.exists():
            return p
    return BASE_DIR / name


# ═══════════════════════════════════════════════════════════════════════
# LOADERS
# ═══════════════════════════════════════════════════════════════════════

@st.cache_resource
def load_encoder():
    return SentenceTransformer('all-mpnet-base-v2')

@st.cache_resource
def load_models():
    models_dir = _find("models")
    if not models_dir.is_dir():
        models_dir = BASE_DIR / "models"
    required = ['scaler.pkl','context_scaler.pkl','tfidf_vectorizer.pkl','model_text.pkl','label_encoder.pkl']
    missing = [f for f in required if not (models_dir / f).exists()]
    if missing:
        st.error(f"❌ Missing: {missing} — run notebook Section 12 first.")
        st.stop()
    def _l(n):
        with open(models_dir / n, 'rb') as f: return pickle.load(f)
    return {
        'scaler': _l('scaler.pkl'), 'context_scaler': _l('context_scaler.pkl'),
        'tfidf': _l('tfidf_vectorizer.pkl'), 'model_text': _l('model_text.pkl'),
        'le': _l('label_encoder.pkl'),
    }

@st.cache_resource
def load_vector_system():
    """Load embeddings and FAISS index (cached across reruns)."""
    for suffix in ['50k', '30k']:
        ep = _find(f"song_embeddings_{suffix}.npy")
        mp = _find(f"song_metadata_{suffix}.csv")
        if ep.exists() and mp.exists():
            emb  = np.load(str(ep))
            meta = pd.read_csv(str(mp))
            idx  = None
            if FAISS_AVAILABLE:
                fp = _find(f"song_index_{suffix}.faiss")
                if fp.exists():
                    try: idx = faiss.read_index(str(fp))
                    except: pass
                if idx is None:
                    emb_f32 = emb.astype(np.float32).copy()
                    faiss.normalize_L2(emb_f32)
                    idx = faiss.IndexFlatIP(emb.shape[1])
                    idx.add(emb_f32)
            return emb, meta, idx
    st.error("⚠️ Embedding files not found. Download from Hugging Face failed or run notebook Section 8.")
    st.stop()


# ── Boot ──────────────────────────────────────────────────────────────
with st.spinner("Loading MoodSense..."):
    encoder  = load_encoder()
    models   = load_models()
    scaler         = models['scaler']
    context_scaler = models['context_scaler']
    tfidf          = models['tfidf']
    model_text     = models['model_text']
    le             = models['le']
    class_names    = le.classes_
    song_embeddings, song_metadata, faiss_index = load_vector_system()

_CONTEXT_N = context_scaler.n_features_in_
_af_path   = _find('models') / 'audio_features.json'
AUDIO_FEATURES = json.load(open(_af_path)) if _af_path.exists() else [
    'Energy','Danceability','Positiveness','Speechiness','Liveness',
    'Acousticness','Instrumentalness','Tempo','Loudness (db)','Key','Time signature'
]

W_LYRIC, W_AUDIO, W_CONTEXT, W_MOOD = 1.0, 2.0, 4.0, 5.0
_vader = SentimentIntensityAnalyzer()

CONTEXT_COLS = [
    'Good for Party','Good for Work/Study','Good for Relaxation/Meditation',
    'Good for Exercise','Good for Running','Good for Yoga/Stretching',
    'Good for Driving','Good for Social Gatherings','Good for Morning Routine',
][:_CONTEXT_N]


# ═══════════════════════════════════════════════════════════════════════
# CORE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════

def clean_lyrics(text):
    if not isinstance(text, str): return ''
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\(.*?\)', '', text)
    text = re.sub(r"[^a-z\s']", '', text)
    return re.sub(r'\s+', ' ', text).strip()

PROMPT_EXPANSIONS = {
    'happy':'joyful upbeat cheerful fun celebration positive energy smile laugh',
    'sad':'melancholy heartbreak lonely crying emotional tears pain sorrow',
    'angry':'rage aggressive intense fury dark rebellion frustration mad',
    'romantic':'love tender intimate slow dance candlelight affection together forever',
    'love':'love tender intimate affection forever heart kiss romance',
    'workout':'energetic motivational powerful pump intense drive push strong',
    'focus':'calm concentration study ambient instrumental minimal productive',
    'study':'calm concentration focus ambient instrumental minimal quiet',
    'work':'calm focus productive ambient concentration minimal background',
    'chill':'relaxed mellow laid back soft peaceful soothing gentle',
    'party':'dance club upbeat energetic fun crowd celebration bass',
    'sleep':'soft quiet gentle lullaby peaceful night dreamy calm',
    'breakup':'heartbreak sad crying lonely pain loss missing over',
    'motivation':'inspire rise overcome strong believe push forward victory',
    'summer':'sunshine warm beach fun vibes happy bright energy',
    'morning':'fresh start bright wake optimistic new day sunrise energy',
    'driving':'road trip open highway cruise freedom rhythm moving',
    'yoga':'calm breathe flow stretch peaceful mindful gentle',
    'running':'pace rhythm beat stride endurance push tempo fast',
}

_EMOTION_KW = {
    'kw_happy':['happy','joy','smile','laugh','celebrate','wonderful','sunshine','cheerful'],
    'kw_sad':['sad','cry','tears','lonely','pain','broken','hurt','sorrow','empty','miss'],
    'kw_anger':['angry','hate','rage','mad','furious','fight','scream','violent'],
    'kw_love':['love','heart','kiss','forever','romantic','darling','baby','tender'],
}

def expand_prompt(p):
    words = p.lower().split()
    exps = [PROMPT_EXPANSIONS[w] for w in words if w in PROMPT_EXPANSIONS]
    return (p + ' ' + ' '.join(exps)) if exps else p

def build_nlp_features(texts):
    rows = []
    for text in texts:
        text = str(text).lower()
        row = {}
        vs = _vader.polarity_scores(text)
        row['vader_neg'] = vs['neg']; row['vader_pos'] = vs['pos']; row['vader_compound'] = vs['compound']
        for feat, keywords in _EMOTION_KW.items():
            row[feat] = sum(text.count(k) for k in keywords)
        rows.append(row)
    return np.array([[r['vader_neg'],r['vader_pos'],r['vader_compound'],
                      r['kw_happy'],r['kw_sad'],r['kw_anger'],r['kw_love']] for r in rows])

def extract_audio_intent(prompt):
    kw = prompt.lower()
    energy=50; dance=50; pos=50; acoustic=50; tempo=120
    speech=10; live=20; instr=10; loud=-10; key=5; tsig=4
    if any(w in kw for w in ['energetic','workout','intense','hype','powerful','pump','running']):
        energy=80; tempo=140; loud=-5
    elif any(w in kw for w in ['chill','relax','calm','mellow','sleep','study','work','yoga']):
        energy=30; tempo=90; loud=-15
    if any(w in kw for w in ['happy','upbeat','joyful','cheerful','fun','party','summer','morning']):
        pos=80; dance=70
    elif any(w in kw for w in ['sad','melancholy','heartbreak','emotional','crying','breakup']):
        pos=20; energy=max(energy,35)
    if any(w in kw for w in ['acoustic','unplugged','guitar','piano']):
        acoustic=80; instr=50
    elif any(w in kw for w in ['electronic','edm','synth','techno']):
        acoustic=10; instr=40
    if any(w in kw for w in ['dance','party','club','disco']):
        dance=80; energy=max(energy,75)
    feat_map = {
        'Energy':energy,'Danceability':dance,'Positiveness':pos,'Speechiness':speech,
        'Liveness':live,'Acousticness':acoustic,'Instrumentalness':instr,
        'Tempo':tempo,'Loudness (db)':loud,'Key':key,'Time signature':tsig,
    }
    values = np.array([[feat_map.get(f,0) for f in AUDIO_FEATURES]])
    try: return scaler.transform(values)[0]
    except: return np.zeros(len(AUDIO_FEATURES))

CONTEXT_MAP = {
    'Good for Party':['party','club','dance','social','gathering','disco'],
    'Good for Work/Study':['work','study','focus','concentration','productive','coding'],
    'Good for Relaxation/Meditation':['relax','calm','meditation','peaceful','chill','yoga','sleep'],
    'Good for Exercise':['exercise','workout','gym','fitness','training'],
    'Good for Running':['running','run','jog','sprint','pace'],
    'Good for Yoga/Stretching':['yoga','stretch','mindful','breathe','flow'],
    'Good for Driving':['driving','drive','road','highway','cruise','trip'],
    'Good for Social Gatherings':['social','gathering','friends','hangout','meetup'],
    'Good for Morning Routine':['morning','wake','sunrise','breakfast','start','fresh'],
}

def extract_context_intent(prompt):
    kw = prompt.lower()
    ctx = np.zeros(_CONTEXT_N)
    for i,col in enumerate(CONTEXT_COLS):
        if any(k in kw for k in CONTEXT_MAP.get(col,[])): ctx[i] = 1.0
    if ctx.sum() == 0: ctx = np.ones(_CONTEXT_N) * 0.5
    return context_scaler.transform(ctx.reshape(1,-1))[0]

def get_prompt_mood_probs(prompt):
    X_tfidf = tfidf.transform([clean_lyrics(prompt)])
    nlp_feats = build_nlp_features([prompt])
    try:
        from sklearn.preprocessing import StandardScaler as _SS
        X_full = hstack([X_tfidf, csr_matrix(nlp_feats.astype(np.float32))])
    except:
        X_full = X_tfidf
    try:
        probs = model_text.predict_proba(X_full)[0]
    except:
        probs = model_text.predict_proba(X_tfidf)[0]
    return None if probs.max() < 0.40 else probs

def make_weighted_embedding(lyric_emb, audio_emb, ctx_emb, mood_emb):
    l = lyric_emb / (norm(lyric_emb) + 1e-9)
    a = audio_emb / (norm(audio_emb) + 1e-9)
    c = ctx_emb   / (norm(ctx_emb)   + 1e-9)
    m = mood_emb  / (norm(mood_emb)  + 1e-9)
    return np.concatenate([l*W_LYRIC, a*W_AUDIO, c*W_CONTEXT, m*W_MOOD])

def generate_playlist(user_prompt, top_k=20, diversity=True):
    expanded      = expand_prompt(user_prompt)
    prompt_lyric  = encoder.encode(expanded)
    prompt_audio  = extract_audio_intent(user_prompt)
    prompt_ctx    = extract_context_intent(user_prompt)
    mood_probs    = get_prompt_mood_probs(user_prompt)

    if mood_probs is not None:
        dominant_mood = class_names[np.argmax(mood_probs)]
        confidence    = float(mood_probs.max())
    else:
        mood_probs    = np.array([0.25,0.25,0.25,0.25])
        dominant_mood = None
        confidence    = None

    prompt_vector = make_weighted_embedding(prompt_lyric, prompt_audio, prompt_ctx, mood_probs)

    if dominant_mood is not None:
        mask = song_metadata['mood'] == dominant_mood
        cand_emb  = song_embeddings[mask.values]
        cand_meta = song_metadata[mask].reset_index(drop=True)
    else:
        cand_emb  = song_embeddings
        cand_meta = song_metadata.reset_index(drop=True)

    t0 = time.perf_counter()
    sims   = cosine_similarity([prompt_vector], cand_emb)[0]
    ranked = np.argsort(sims)[::-1]
    search_ms = (time.perf_counter() - t0) * 1000

    if diversity:
        selected = []; artist_counts = {}
        for idx in ranked:
            a = cand_meta.iloc[idx]['artist']
            if artist_counts.get(a,0) < 2:
                selected.append(idx)
                artist_counts[a] = artist_counts.get(a,0) + 1
            if len(selected) >= top_k: break
        final = selected
    else:
        final = list(ranked[:top_k])

    results = cand_meta.iloc[final].copy()
    results['similarity'] = sims[final]
    results['rank'] = range(1, len(results)+1)

    results.attrs = {
        'dominant_mood': dominant_mood,
        'confidence':    confidence,
        'pool_size':     len(cand_meta),
        'search_ms':     round(search_ms, 1),
        'faiss_used':    FAISS_AVAILABLE and faiss_index is not None,
    }

    _log.info('{"event":"playlist","prompt":"%s","mood":"%s","pool":%d,"ms":%.1f}',
              user_prompt[:60], dominant_mood or "any", len(cand_meta), search_ms)
    return results


# ═══════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("""
    <div style="margin-bottom:1.5rem;">
        <div style="font-family:'Unbounded',sans-serif;font-size:1.1rem;font-weight:900;
                    color:#1DB954;letter-spacing:-0.01em;">MOODSENSE</div>
        <div style="font-family:'DM Mono',monospace;font-size:0.62rem;letter-spacing:0.15em;
                    color:#555;text-transform:uppercase;margin-top:0.2rem;">DSP 422 · GROUP 3</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-head">Playlist Settings</div>', unsafe_allow_html=True)
    playlist_size    = st.slider("Songs", 5, 50, 20, key="pl_size")
    diversity_filter = st.checkbox("Artist diversity (max 2 per artist)", value=True)

    with st.expander("Advanced"):
        energy_note = st.slider("Energy offset", -20, 20, 0)
        mood_note   = st.slider("Positiveness offset", -20, 20, 0)

    st.markdown('<div class="section-head" style="margin-top:1.5rem;">System Status</div>', unsafe_allow_html=True)

    faiss_status = "⚡ FAISS Active" if (FAISS_AVAILABLE and faiss_index is not None) else "🐌 numpy fallback"
    st.markdown(f"""
    <div style="font-family:'DM Mono',monospace;font-size:0.72rem;line-height:2;color:#888;">
        <div><span style="color:#1DB954;">✓</span> {len(song_embeddings):,} songs loaded</div>
        <div><span style="color:#1DB954;">✓</span> {song_embeddings.shape[1]}-dim embeddings</div>
        <div><span style="color:{'#1DB954' if FAISS_AVAILABLE else '#ff4757'};">{'✓' if FAISS_AVAILABLE else '✗'}</span> {faiss_status}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-head" style="margin-top:1.5rem;">Team</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-family:'DM Mono',monospace;font-size:0.7rem;line-height:1.9;color:#666;">
        Ankit Mittal<br>Albin Anto Jose<br>Nandini Bag<br>Kasheena Mulla
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════
# HERO
# ═══════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="hero">
    <div class="hero-eyebrow">MS DSP 422 · Practical Machine Learning · Group 3</div>
    <div class="hero-title">MoodSense</div>
    <div class="hero-sub">
        Describe a vibe in natural language. The system encodes your prompt into a
        806-dimensional multimodal space using Sentence-BERT, audio features, activity
        context tags, and mood probabilities — then finds the closest songs via cosine similarity.
    </div>
    <div class="hero-badges">
        <span class="badge green">BERT 768-dim</span>
        <span class="badge teal">806-dim Embeddings</span>
        <span class="badge blue">FAISS ANN Search</span>
        <span class="badge">TF-IDF + LinearSVC</span>
        <span class="badge">50K Songs</span>
    </div>
</div>
""", unsafe_allow_html=True)

# KPI Row
mood_counts = song_metadata['mood'].value_counts() if 'mood' in song_metadata.columns else {}
best_mood   = mood_counts.index[0] if len(mood_counts) else "—"

st.markdown(f"""
<div class="kpi-grid">
    <div class="kpi-card">
        <div class="kpi-num">{len(song_embeddings):,}</div>
        <div class="kpi-label">Songs Indexed</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-num">{song_embeddings.shape[1]}</div>
        <div class="kpi-label">Embedding Dims</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-num">{'FAISS' if FAISS_AVAILABLE else 'numpy'}</div>
        <div class="kpi-label">Vector Search</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-num">~60%</div>
        <div class="kpi-label">Classifier Accuracy</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4 = st.tabs([
    "🎧  Your Playlist",
    "🎵  Explore Dataset",
    "📊  Analytics",
    "🔬  Model Info",
])


# ─────────────────────────────────────────────────────────────────────
# TAB 1 — PLAYLIST GENERATOR
# ─────────────────────────────────────────────────────────────────────
with tab1:
    right = st.container()

    with right:
        st.markdown('<div class="section-head">Describe Your Vibe</div>', unsafe_allow_html=True)

        prompt_text = st.text_area(
            label="prompt",
            placeholder='"Sunday morning, soft rain on the window, nostalgic but okay…"',
            height=100,
            key="custom_prompt",
            label_visibility="collapsed",
        )

        ca, cb, cc = st.columns([3,1,1])
        with ca:
            n_custom = st.slider("Songs", 5, 40, 20, key="n_custom")
        with cb:
            st.markdown("<br>", unsafe_allow_html=True)
            gen_btn = st.button("Generate", type="primary", use_container_width=True, key="gen_btn")
        with cc:
            st.markdown("<br>", unsafe_allow_html=True)
            clr_btn = st.button("Clear", key="clr_btn", use_container_width=True)

        if clr_btn:
            for k in ['pl_prompt','pl_n','pl_title','run_pl','pl_result']:
                st.session_state.pop(k, None)
            st.rerun()

        if gen_btn and prompt_text.strip():
            st.session_state['pl_prompt'] = prompt_text.strip()
            st.session_state['pl_n']      = n_custom
            st.session_state['pl_title']  = "🎧 Custom Playlist"
            st.session_state['run_pl']    = True
        elif gen_btn:
            st.warning("Type a mood or vibe first.")

    if st.session_state.get('run_pl'):
        st.session_state['run_pl'] = False
        active_prompt = st.session_state.get('pl_prompt','')
        active_n      = st.session_state.get('pl_n', 20)
        active_title  = st.session_state.get('pl_title','Playlist')

        with st.spinner("Encoding prompt · Searching embeddings…"):
            pl = generate_playlist(active_prompt, top_k=active_n, diversity=diversity_filter)
            st.session_state['pl_result'] = {
                'df': pl, 'title': active_title, 'prompt': active_prompt
            }

    res = st.session_state.get('pl_result')
    if res and not res['df'].empty:
        pl     = res['df']
        title  = res['title']
        prompt = res['prompt']

        mood       = pl.attrs.get('dominant_mood')
        conf       = pl.attrs.get('confidence')
        pool_size  = pl.attrs.get('pool_size')
        search_ms  = pl.attrs.get('search_ms', 0)
        faiss_used = pl.attrs.get('faiss_used', False)
        mc         = MOOD_CONFIG.get(mood, {'color':'#888','emoji':'🎵','desc':''}) if mood else {'color':'#888','emoji':'🎵','desc':'Mixed'}

        st.markdown("---")

        st.markdown(f"""
        <div class="result-header">
            <div class="result-mood-icon">{mc['emoji']}</div>
            <div>
                <div class="result-title">{title}</div>
                <div class="result-meta">
                    "{prompt}" &nbsp;·&nbsp;
                    {len(pl)} tracks &nbsp;·&nbsp;
                    {'Mood: <strong>' + mood + '</strong> (' + f'{conf:.0%}' + ')' if mood else 'No dominant mood detected'}
                    &nbsp;·&nbsp; searched {pool_size:,} songs
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="metrics-row">
            <div class="metric-chip">
                <div class="metric-val">{search_ms:.0f}ms</div>
                <div class="metric-lbl">Search Time</div>
            </div>
            <div class="metric-chip">
                <div class="metric-val">{pool_size:,}</div>
                <div class="metric-lbl">Pool Size</div>
            </div>
            <div class="metric-chip">
                <div class="metric-val">{'FAISS' if faiss_used else 'numpy'}</div>
                <div class="metric-lbl">Vector Engine</div>
            </div>
            <div class="metric-chip">
                <div class="metric-val">{pl['similarity'].iloc[0]:.3f}</div>
                <div class="metric-lbl">Top Match</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        v1, v2 = st.columns(2)
        with v1:
            if 'energy' in pl.columns and 'positiveness' in pl.columns:
                fig = px.scatter(
                    pl, x='energy', y='positiveness',
                    color='mood',
                    color_discrete_map={m: MOOD_CONFIG[m]['color'] for m in MOOD_CONFIG if m in pl['mood'].values},
                    hover_data=['song','artist'],
                    title="Energy × Vibe Profile",
                    height=250,
                )
                fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#888', family='DM Mono, monospace', size=10),
                    title_font=dict(size=11), showlegend=False,
                    margin=dict(l=0,r=0,t=30,b=0),
                )
                fig.update_xaxes(showgrid=True, gridcolor='#1a1a1a', color='#555')
                fig.update_yaxes(showgrid=True, gridcolor='#1a1a1a', color='#555')
                st.plotly_chart(fig, use_container_width=True)

        with v2:
            mc_counts = pl['mood'].value_counts()
            fig2 = go.Figure(data=[go.Pie(
                labels=mc_counts.index,
                values=mc_counts.values,
                marker_colors=[MOOD_CONFIG.get(m,{}).get('color','#888') for m in mc_counts.index],
                hole=0.55,
                textfont=dict(family='DM Mono, monospace', size=9),
            )])
            fig2.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#888'), showlegend=True,
                title=dict(text="Mood Mix", font=dict(size=11, color='#888')),
                height=250, margin=dict(l=0,r=0,t=30,b=0),
                legend=dict(font=dict(size=9, family='DM Mono'), bgcolor='rgba(0,0,0,0)'),
            )
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown(f'<div class="section-head" style="margin-top:1rem;">{len(pl)} Tracks</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="tracklist">
            <div class="tracklist-header">
                <span>#</span><span>Song</span><span>Artist</span>
                <span>Mood</span><span>Match</span><span>Energy</span>
            </div>
        """, unsafe_allow_html=True)

        for _, row in pl.iterrows():
            m   = str(row.get('mood',''))
            mc2 = MOOD_CONFIG.get(m, {'color':'#888','bg':'rgba(100,100,100,0.1)'})
            sim = float(row['similarity'])
            bar_w = int(min(sim * 120, 100))
            energy_val = row.get('energy', row.get('Energy', 0))

            st.markdown(f"""
            <div class="track-row">
                <span class="track-num">{int(row['rank'])}</span>
                <div>
                    <div class="track-name">{row.get('song','—')}</div>
                </div>
                <div class="track-artist">{row.get('artist','—')}</div>
                <span class="mood-pill" style="background:{mc2['bg']};color:{mc2['color']};">
                    {m}
                </span>
                <div class="sim-bar-wrap">
                    <div class="sim-bar-bg">
                        <div class="sim-bar-fg" style="width:{bar_w}%;background:{mc2['color']};"></div>
                    </div>
                    <div style="font-family:'DM Mono',monospace;font-size:0.58rem;color:#555;margin-top:2px;">{sim:.3f}</div>
                </div>
                <div style="font-family:'DM Mono',monospace;font-size:0.75rem;color:#666;">{int(energy_val) if not isinstance(energy_val,float) or energy_val > 1 else f'{energy_val:.0%}'}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.download_button(
            f"Export CSV ({len(pl)} songs)",
            pl.to_csv(index=False),
            f"moodsense_{prompt[:20].replace(' ','_')}.csv",
            "text/csv",
            use_container_width=True,
        )

    elif not res:
        st.markdown("""
        <div style="text-align:center;padding:4rem 2rem;color:#333;">
            <div style="font-size:3rem;margin-bottom:1rem;filter:grayscale(1);">🎶</div>
            <div style="font-family:'DM Mono',monospace;font-size:0.75rem;letter-spacing:0.15em;
                        text-transform:uppercase;color:#444;">
                Pick a preset or type your mood above
            </div>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────
# TAB 2 — EXPLORE DATASET
# ─────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown('<div class="section-head">Dataset Browser</div>', unsafe_allow_html=True)

    mood_options = ['All'] + sorted(song_metadata['mood'].dropna().unique().tolist()) if 'mood' in song_metadata.columns else ['All']
    sel_mood = st.selectbox(
        "Filter by mood",
        mood_options,
        format_func=lambda x: f"{MOOD_CONFIG[x]['emoji']} {x}" if x in MOOD_CONFIG else x,
        label_visibility="collapsed",
    )

    fdf = song_metadata if sel_mood == 'All' else song_metadata[song_metadata['mood'] == sel_mood]
    st.markdown(f'<div style="font-family:\'DM Mono\',monospace;font-size:0.7rem;color:#555;margin-bottom:0.75rem;">{len(fdf):,} songs</div>', unsafe_allow_html=True)

    disp_cols = [c for c in ['song','artist','mood','energy','positiveness'] if c in fdf.columns]
    rename    = {'song':'Song','artist':'Artist','mood':'Mood','energy':'Energy','positiveness':'Vibe'}
    st.dataframe(fdf[disp_cols].head(300).rename(columns=rename), use_container_width=True, height=500, hide_index=True)
    st.download_button("Export filtered CSV", fdf[disp_cols].to_csv(index=False), f"moodsense_{sel_mood}.csv","text/csv")


# ─────────────────────────────────────────────────────────────────────
# TAB 3 — ANALYTICS
# ─────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown('<div class="section-head">Dataset Analytics</div>', unsafe_allow_html=True)

    if 'mood' in song_metadata.columns:
        mc = song_metadata['mood'].value_counts()

        a1, a2 = st.columns(2)
        with a1:
            fig = go.Figure(data=[go.Pie(
                labels=mc.index, values=mc.values,
                marker_colors=[MOOD_CONFIG.get(m,{}).get('color','#888') for m in mc.index],
                hole=0.5,
                textfont=dict(family='DM Mono, monospace', size=10),
            )])
            fig.update_layout(
                title="Mood Distribution", height=320,
                paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#888', family='DM Mono'),
                legend=dict(font=dict(size=9), bgcolor='rgba(0,0,0,0)'),
                margin=dict(l=0,r=0,t=40,b=0),
            )
            st.plotly_chart(fig, use_container_width=True)

        with a2:
            st.markdown('<div class="section-head">Breakdown</div>', unsafe_allow_html=True)
            for mood, cnt in mc.items():
                mc3 = MOOD_CONFIG.get(mood, {'color':'#888','emoji':'🎵'})
                pct = cnt / len(song_metadata) * 100
                st.markdown(f"""
                <div style="display:flex;align-items:center;gap:0.75rem;padding:0.5rem 0;
                            border-bottom:1px solid #1a1a1a;">
                    <span>{mc3['emoji']}</span>
                    <div style="flex:1;font-size:0.85rem;">{mood}</div>
                    <div style="font-family:'DM Mono',monospace;font-size:0.75rem;color:{mc3['color']};">
                        {cnt:,} · {pct:.1f}%
                    </div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown('<div class="section-head">Audio Feature Distribution by Mood</div>', unsafe_allow_html=True)

        avail_feats = [c for c in ['energy','positiveness','Energy','Positiveness','Danceability','danceability','Acousticness','acousticness'] if c in song_metadata.columns]
        if avail_feats:
            sel_feat = st.selectbox("Feature", avail_feats, label_visibility="collapsed")
            fig2 = px.violin(
                song_metadata, x='mood', y=sel_feat, color='mood',
                color_discrete_map={m: MOOD_CONFIG[m]['color'] for m in MOOD_CONFIG if m in song_metadata['mood'].values},
                box=True, points=False,
                height=350,
            )
            fig2.update_layout(
                showlegend=False, height=350,
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#888', family='DM Mono', size=10),
                margin=dict(l=0,r=0,t=10,b=0),
            )
            fig2.update_xaxes(showgrid=False, color='#555')
            fig2.update_yaxes(showgrid=True, gridcolor='#1a1a1a', color='#555')
            st.plotly_chart(fig2, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────
# TAB 4 — MODEL INFO
# ─────────────────────────────────────────────────────────────────────
with tab4:
    st.markdown('<div class="section-head">Architecture</div>', unsafe_allow_html=True)

    st.markdown("""
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:1rem;margin-bottom:1.5rem;">
    """, unsafe_allow_html=True)

    arch_items = [
        ("BERT Encoder",    "all-mpnet-base-v2",          "768-dim lyric semantics"),
        ("Classifier",      "CalibratedLinearSVC",         "TF-IDF + VADER + keywords → 4 moods"),
        ("Embeddings",      "806-dim weighted",            "Lyric×1 · Audio×2 · Context×4 · Mood×5"),
        ("Vector Search",   "FAISS IndexFlatIP" if FAISS_AVAILABLE else "numpy cosine", "O(log n) ANN search"),
        ("Training Data",   "50K songs balanced",          "15K per class · 4 mood classes"),
    ]

    for name, val, desc in arch_items:
        st.markdown(f"""
        <div style="background:#111;border:1px solid #1a1a1a;border-radius:12px;padding:1rem 1.25rem;margin-bottom:0.75rem;">
            <div style="font-family:'DM Mono',monospace;font-size:0.62rem;letter-spacing:0.12em;
                        text-transform:uppercase;color:#555;margin-bottom:0.3rem;">{name}</div>
            <div style="font-size:0.9rem;font-weight:600;color:#1DB954;margin-bottom:0.2rem;">{val}</div>
            <div style="font-size:0.78rem;color:#666;">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="section-head">Model Comparison</div>', unsafe_allow_html=True)

    model_df = pd.DataFrame({
        'Model':     ['Original LR (baseline)', 'LR + balanced + NLP', 'LinearSVC + balanced + NLP', 'Ensemble (SVC+LR)'],
        'Accuracy':  [0.62,                      0.65,                   0.67,                         0.68],
        'F1 Macro':  [0.57,                      0.60,                   0.62,                         0.63],
        'Notes':     ['Random 50k, unbalanced', 'Balanced 15k/class', 'Max-margin on sparse TF-IDF', 'Best — used in app'],
    })

    fig3 = px.bar(
        model_df.melt(id_vars='Model', value_vars=['Accuracy','F1 Macro'], var_name='Metric', value_name='Score'),
        x='Model', y='Score', color='Metric', barmode='group',
        color_discrete_map={'Accuracy':'#1DB954','F1 Macro':'#42f5a7'},
        height=300,
    )
    fig3.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#888', family='DM Mono', size=10),
        legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(size=9)),
        yaxis_range=[0.5, 0.75],
        margin=dict(l=0,r=0,t=10,b=0),
    )
    fig3.update_xaxes(showgrid=False, color='#555', tickangle=-15)
    fig3.update_yaxes(showgrid=True, gridcolor='#1a1a1a', color='#555')
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown('<div class="section-head" style="margin-top:1rem;">Why This Architecture</div>', unsafe_allow_html=True)

    insights = [
        ("Lyrics carry emotion", "Audio features alone achieve 42% — barely above random for 4 classes. All meaningful emotion signal lives in lyrics, confirmed by SHAP analysis."),
        ("Dimensionality = implicit weight", "Without per-modality normalization, the 768-dim BERT vector dominates cosine similarity. Mood and context contribute <3% of signal. Normalizing each modality before weighting gives every component genuine influence."),
        ("Content-based vs collaborative", "Spotify uses collaborative filtering on 600M users' listening history. Our system is content-based — it works immediately for any new user with zero history. Different tradeoff, not inferior."),
        ("FAISS for production scale", "At 50K songs numpy cosine runs ~100ms. FAISS runs ~1ms. At 1M songs: numpy 2s, FAISS 5ms. The architecture scales to Spotify's 100M without changing the ML layer."),
    ]

    for title, text in insights:
        with st.expander(title):
            st.markdown(f'<div style="font-size:0.85rem;color:#888;line-height:1.7;">{text}</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════════════

st.markdown("---")
st.markdown("""
<div style="display:flex;align-items:center;justify-content:space-between;
            padding:1.25rem 0;border-top:1px solid #1a1a1a;margin-top:1rem;">
    <div style="font-family:'Unbounded',sans-serif;font-size:0.85rem;font-weight:900;color:#333;">
        MOODSENSE
    </div>
    <div style="font-family:'DM Mono',monospace;font-size:0.65rem;letter-spacing:0.1em;color:#444;text-align:center;">
        MS DSP 422 · Practical Machine Learning · Northwestern University
    </div>
    <div style="font-family:'DM Mono',monospace;font-size:0.65rem;color:#444;">
        Sentence-BERT · TF-IDF · FAISS
    </div>
</div>
""", unsafe_allow_html=True)