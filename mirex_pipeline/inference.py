import pickle, requests, re, base64, os
import numpy as np, pandas as pd
from scipy.sparse import hstack, csr_matrix
from dotenv import dotenv_values

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR  = os.path.dirname(BASE_DIR)   # repo root
MODEL_DIR = os.path.join(BASE_DIR, 'models')

# ── Config (local .env or Streamlit Cloud secrets) ─────────────────────────────
cfg = dotenv_values(os.path.join(ROOT_DIR, '.env'))
if not cfg.get('SPOTIFY_CLIENT_ID'):
    try:
        import streamlit as st
        cfg = st.secrets
    except Exception:
        pass

# ── Load artifacts ─────────────────────────────────────────────────────────────
# LightGBM classifier trained on TF-IDF (20K bigram features) + 12 Spotify audio
# features against MIREX human-anchored emotion labels (70K songs, 4 classes)
with open(os.path.join(MODEL_DIR, 'lgbm_tfidf_audio.pkl'), 'rb') as f:
    lgbm_model = pickle.load(f)

# TF-IDF vectorizer: 20K bigram features, fitted on training split only
with open(os.path.join(MODEL_DIR, 'tfidf_mirex.pkl'), 'rb') as f:
    tfidf = pickle.load(f)

# StandardScaler for 12 audio features, fitted on training split only
with open(os.path.join(MODEL_DIR, 'audio_scaler_mirex.pkl'), 'rb') as f:
    scaler = pickle.load(f)

# 12 Spotify audio features — must match the order used during training
AUDIO_COLS = [
    'danceability', 'energy', 'key', 'loudness', 'mode',
    'speechiness', 'acousticness', 'instrumentalness',
    'liveness', 'valence', 'tempo', 'duration_ms'
]

# ── Text cleaning ──────────────────────────────────────────────────────────────
# Mirrors the cleaning applied during training (Section 1 of training notebook)
def clean_lyrics(text):
    if not isinstance(text, str): return ''
    text = re.sub(r'\[.*?\]', '', text)   # remove [Chorus], [Verse], etc.
    text = re.sub(r'\b(la|na|oh|ah|uh|yeah|ya|ooh|ayy|aye|hee|duh)\b', '', text, flags=re.IGNORECASE)
    return re.sub(r'\s+', ' ', text).strip()

# ── Helpers ────────────────────────────────────────────────────────────────────
def _spotify_token():
    creds = base64.b64encode(
        f'{cfg["SPOTIFY_CLIENT_ID"]}:{cfg["SPOTIFY_CLIENT_SECRET"]}'.encode()
    ).decode()
    r = requests.post(
        'https://accounts.spotify.com/api/token',
        headers={'Authorization': f'Basic {creds}'},
        data={'grant_type': 'client_credentials'}
    )
    return r.json()['access_token']

def _spotify_search(song, artist=None):
    token = _spotify_token()
    query = f'track:{song} artist:{artist}' if artist else song
    r = requests.get(
        'https://api.spotify.com/v1/search',
        headers={'Authorization': f'Bearer {token}'},
        params={'q': query, 'type': 'track', 'limit': 1}
    )
    items = r.json().get('tracks', {}).get('items', [])
    if not items:
        return None
    t = items[0]
    return {
        'id':        t['id'],
        'name':      t['name'],
        'artist':    t['artists'][0]['name'],
        'embed_url': f"https://open.spotify.com/embed/track/{t['id']}?utm_source=generator"
    }

def _reccobeats_features(spotify_track_id):
    r = requests.get('https://api.reccobeats.com/v1/track',
                     params={'ids': spotify_track_id})
    r.raise_for_status()
    content = r.json().get('content', [])
    if not content:
        print(f"Warning: No audio features for {spotify_track_id}, using zeros.")
        return {f: 0 for f in AUDIO_COLS}
    rb_id = content[0]['id']
    r2 = requests.get(f'https://api.reccobeats.com/v1/track/{rb_id}/audio-features')
    r2.raise_for_status()
    data = r2.json()
    return {
        'danceability':     data.get('danceability', 0),
        'energy':           data.get('energy', 0),
        'key':              data.get('key', 0),
        'loudness':         data.get('loudness', 0),
        'mode':             data.get('mode', 0),
        'speechiness':      data.get('speechiness', 0),
        'acousticness':     data.get('acousticness', 0),
        'instrumentalness': data.get('instrumentalness', 0),
        'liveness':         data.get('liveness', 0),
        'valence':          data.get('valence', 0),
        'tempo':            data.get('tempo', 0),
        'duration_ms':      data.get('duration_ms', 0),
    }

def _clean_track_name(name):
    name = re.sub(r'\s*\(From[^)]*\)',   '', name, flags=re.IGNORECASE)
    name = re.sub(r'\s*\(feat\.[^)]*\)', '', name, flags=re.IGNORECASE)
    name = re.sub(r'\s*\(ft\.[^)]*\)',   '', name, flags=re.IGNORECASE)
    name = re.sub(r'\s*\(with[^)]*\)',   '', name, flags=re.IGNORECASE)
    return name.strip()

def _lyrics_ovh(song, artist):
    """Fetch lyrics from lyrics.ovh — free, no API key, no Cloudflare."""
    try:
        clean_song   = _clean_track_name(song)
        r = requests.get(
            f'https://api.lyrics.ovh/v1/{requests.utils.quote(artist)}/{requests.utils.quote(clean_song)}',
            timeout=8
        )
        if r.status_code == 200:
            return r.json().get('lyrics', None)
    except Exception:
        pass
    return None

# ── Public API ─────────────────────────────────────────────────────────────────
def classify_song(song: str, artist: str = None) -> dict:
    """
    Classify a song's emotion using TF-IDF + audio features.

    Pipeline:
        1. Spotify Search API       → track metadata + embed URL
        2. ReccoBeats API           → 12 audio features
        3. lyrics.ovh API           → full lyrics (free, no key, cloud-safe)
        4. clean_lyrics()           → remove filler words/markers
        5. tfidf.transform()        → 20K-dim sparse text vector
        6. scaler.transform()       → 12-dim scaled audio vector
        7. hstack([text, audio])    → 20012-dim combined feature vector
        8. lgbm_model.predict_proba → emotion probabilities

    Args:
        song:   Song title
        artist: Artist name (optional but improves Spotify search accuracy)

    Returns:
        {
            'emotion':       'Happy' | 'Sad' | 'Anger' | 'Love',
            'confidence':    float,
            'probabilities': {'Anger': float, 'Happy': float, 'Love': float, 'Sad': float},
            'track':         {'id', 'name', 'artist', 'embed_url'},
            'lyrics_found':  bool
        }
    """
    # 1. Spotify search
    track = _spotify_search(song, artist)
    if not track:
        raise ValueError(f"Track not found on Spotify: '{song}'")

    # 2. Audio features via ReccoBeats
    audio_features = _reccobeats_features(track['id'])

    # 3. Lyrics via lyrics.ovh
    lyrics       = _lyrics_ovh(track['name'], track['artist']) or ''
    lyrics_found = bool(lyrics)

    # 4 & 5. Clean and vectorise lyrics with TF-IDF
    # Falls back to song title if lyrics not found
    text_input = clean_lyrics(lyrics) if lyrics else track['name']
    X_text = tfidf.transform([text_input])

    # 6. Scale audio features
    audio_vals = [[audio_features[col] for col in AUDIO_COLS]]
    X_audio    = csr_matrix(scaler.transform(audio_vals))

    # 7. Combine: sparse TF-IDF (20000d) + scaled audio (12d) = 20012d
    X = hstack([X_text, X_audio])

    # 8. Predict
    proba   = lgbm_model.predict_proba(X)[0]
    labels  = lgbm_model.classes_
    top_idx = int(np.argmax(proba))

    return {
        'emotion':       labels[top_idx],
        'confidence':    round(float(proba[top_idx]), 4),
        'probabilities': {l: round(float(p), 4) for l, p in zip(labels, proba)},
        'track':         track,
        'lyrics_found':  lyrics_found
    }
