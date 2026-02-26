import pickle, requests, re, base64, os
import numpy as np, pandas as pd
from openai import OpenAI
from dotenv import dotenv_values
import lyricsgenius

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models')

# ── Config ─────────────────────────────────────────────────────────────────────
cfg = dotenv_values(os.path.join(BASE_DIR, '.env'))

# ── Load artifacts ─────────────────────────────────────────────────────────────
with open(os.path.join(MODEL_DIR, 'lgbm_mirex_1548d.pkl'), 'rb') as f:
    lgbm_1548 = pickle.load(f)

with open(os.path.join(MODEL_DIR, 'audio_scaler.pkl'), 'rb') as f:
    scaler = pickle.load(f)

client  = OpenAI(api_key=cfg['OPENAI_API_KEY'])
_genius = lyricsgenius.Genius(cfg['GENIUS_ACCESS_TOKEN'],
                               verbose=False,
                               remove_section_headers=True)

AUDIO_FEATURE_ORDER = [
    'danceability', 'energy', 'key', 'loudness', 'mode',
    'speechiness', 'acousticness', 'instrumentalness',
    'liveness', 'valence', 'tempo', 'duration_ms'
]

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
        return {f: 0 for f in AUDIO_FEATURE_ORDER}
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

def _genius_lyrics(song, artist):
    result = _genius.search_song(_clean_track_name(song), artist)
    return result.lyrics if result else None

# ── Public API ─────────────────────────────────────────────────────────────────
def classify_song(song: str, artist: str = None) -> dict:
    """
    Classify a song's emotion from its name and optional artist.

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

    # 3. Lyrics via Genius
    lyrics       = _genius_lyrics(track['name'], track['artist']) or ''
    lyrics_found = bool(lyrics)

    # 4. Embed lyrics → 1536d
    resp = client.embeddings.create(
        model='text-embedding-3-small',
        input=lyrics[:8000] if lyrics else track['name'],
        dimensions=1536
    )
    text_vec = np.array(resp.data[0].embedding, dtype=np.float32).flatten()

    # 5. Scale audio → 12d
    audio_vec    = pd.DataFrame(
        [[audio_features[f] for f in scaler.feature_names_in_]],
        columns=scaler.feature_names_in_,
        dtype=np.float32
    )
    audio_scaled = scaler.transform(audio_vec).flatten()

    # 6. Concatenate → 1548d
    x = pd.DataFrame(
        np.concatenate([text_vec, audio_scaled]).reshape(1, -1),
        columns=lgbm_1548.feature_names_in_
    )

    # 7. Predict
    proba   = lgbm_1548.predict_proba(x)[0]
    labels  = lgbm_1548.classes_
    top_idx = int(np.argmax(proba))

    return {
        'emotion':       labels[top_idx],
        'confidence':    round(float(proba[top_idx]), 4),
        'probabilities': {l: round(float(p), 4) for l, p in zip(labels, proba)},
        'track':         track,
        'lyrics_found':  lyrics_found
    }
