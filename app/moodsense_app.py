"""
MoodSense: AI-Powered Music Playlist Generator
A production-ready Streamlit demo showcasing semantic music retrieval

Author: [Your Team Names]
Course: MS DSP 422 - Practical Machine Learning
"""

import streamlit as st
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go

# ═══════════════════════════════════════════════════════════════════════
# PAGE CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="MoodSense - AI Playlist Generator",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ═══════════════════════════════════════════════════════════════════════
# CUSTOM CSS (Spotify-Inspired Theme)
# ═══════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
    /* Main container */
    .main {
        background: linear-gradient(135deg, #1e1e1e 0%, #121212 100%);
        color: #FFFFFF;
    }
    
    /* Header styling */
    h1 {
        color: #1DB954;
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 700;
    }
    
    h2, h3 {
        color: #FFFFFF;
        font-family: 'Helvetica Neue', sans-serif;
    }
    
    /* Input box styling */
    .stTextInput > div > div > input {
        background-color: #282828;
        color: #FFFFFF;
        border: 2px solid #1DB954;
        border-radius: 20px;
        padding: 12px 20px;
        font-size: 16px;
    }
    
    /* Button styling */
    .stButton > button {
        background-color: #1DB954;
        color: #FFFFFF;
        border: none;
        border-radius: 25px;
        padding: 12px 40px;
        font-size: 16px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background-color: #1ED760;
        transform: scale(1.05);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #000000;
    }
    
    /* Success box */
    .element-container .stAlert {
        background-color: #282828;
        border-left: 4px solid #1DB954;
    }
    
    /* Dataframe styling */
    .dataframe {
        background-color: #282828 !important;
        color: #FFFFFF !important;
    }
    
    /* Example prompt pills */
    .example-pill {
        display: inline-block;
        background-color: #282828;
        color: #1DB954;
        padding: 8px 16px;
        margin: 5px;
        border-radius: 20px;
        border: 1px solid #1DB954;
        cursor: pointer;
        transition: all 0.2s;
    }
    
    .example-pill:hover {
        background-color: #1DB954;
        color: #000000;
    }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════
# LOAD MODELS & DATA (Cached for Performance)
# ═══════════════════════════════════════════════════════════════════════

@st.cache_resource
def load_encoder():
    """Load Sentence-BERT encoder (runs once)"""
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource
def load_scaler():
    """Create and return a fitted StandardScaler"""
    # In production, load your pre-fitted scaler from training
    scaler = StandardScaler()
    # For demo, create dummy fit (replace with your actual scaler)
    dummy_data = np.random.randn(100, 8)
    scaler.fit(dummy_data)
    return scaler

@st.cache_data
def load_embeddings():
    """Load pre-computed song embeddings"""
    try:
        embeddings = np.load('../data/processed/song_embeddings_30k.npy')
        metadata = pd.read_csv('../data/processed/song_metadata_30k.csv')
        return embeddings, metadata
    except FileNotFoundError:
        st.error("⚠️ Embedding files not found. Please upload song_embeddings_30k.npy and song_metadata_30k.csv")
        st.stop()

# Initialize
encoder = load_encoder()
scaler = load_scaler()

# Load data with error handling
try:
    song_embeddings, song_metadata = load_embeddings()
    st.sidebar.success(f"✅ Loaded {len(song_embeddings):,} songs")
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# ═══════════════════════════════════════════════════════════════════════
# AUDIO INTENT EXTRACTION
# ═══════════════════════════════════════════════════════════════════════

def extract_audio_intent(prompt, energy_boost=0, positiveness_boost=0):
    """Map prompt keywords to audio feature values"""
    keywords = prompt.lower()
    
    # Defaults
    energy = 50 + energy_boost
    danceability = 50
    positiveness = 50 + positiveness_boost
    acousticness = 50
    tempo = 120
    speechiness = 10
    liveness = 20
    instrumentalness = 10
    
    # Energy & Tempo
    if any(w in keywords for w in ['energetic', 'workout', 'intense', 'hype', 'powerful']):
        energy = 80 + energy_boost
        tempo = 140
    elif any(w in keywords for w in ['chill', 'relax', 'calm', 'mellow', 'slow']):
        energy = 30
        tempo = 90
    
    # Mood
    if any(w in keywords for w in ['happy', 'upbeat', 'joyful', 'cheerful', 'fun']):
        positiveness = 80 + positiveness_boost
        danceability = 70
    elif any(w in keywords for w in ['sad', 'melancholy', 'heartbreak', 'emotional']):
        positiveness = 20
        energy = 35
    
    # Acoustic / Electronic
    if any(w in keywords for w in ['acoustic', 'unplugged', 'guitar', 'piano']):
        acousticness = 80
        instrumentalness = 50
    elif any(w in keywords for w in ['electronic', 'edm', 'synth', 'techno']):
        acousticness = 10
        instrumentalness = 40
    
    # Dance / Party
    if any(w in keywords for w in ['dance', 'party', 'club', 'disco']):
        danceability = 80
        energy = 75
    
    # Lyrics focus
    if any(w in keywords for w in ['rap', 'hip hop', 'spoken']):
        speechiness = 30
    elif 'instrumental' in keywords:
        instrumentalness = 80
        speechiness = 5
    
    features = np.array([[energy, danceability, positiveness, 
                         acousticness, tempo, speechiness,
                         liveness, instrumentalness]])
    return scaler.transform(features)[0]

def extract_mood_intent(prompt):
    """Map prompt keywords to mood probabilities"""
    keywords = prompt.lower()
    mood_probs = np.array([0.25, 0.25, 0.25, 0.25])  # [Anger, Happy, Love, Sad]
    
    if any(w in keywords for w in ['happy', 'joyful', 'upbeat', 'cheerful', 'fun']):
        mood_probs = np.array([0.05, 0.8, 0.1, 0.05])
    elif any(w in keywords for w in ['sad', 'heartbreak', 'melancholy', 'lonely', 'crying']):
        mood_probs = np.array([0.05, 0.05, 0.1, 0.8])
    elif any(w in keywords for w in ['angry', 'aggressive', 'intense', 'rage', 'metal']):
        mood_probs = np.array([0.8, 0.05, 0.05, 0.1])
    elif any(w in keywords for w in ['love', 'romantic', 'romance', 'valentine', 'wedding']):
        mood_probs = np.array([0.05, 0.1, 0.8, 0.05])
    
    return mood_probs

# ═══════════════════════════════════════════════════════════════════════
# PLAYLIST GENERATION
# ═══════════════════════════════════════════════════════════════════════

def generate_playlist(user_prompt, top_k=20, energy_adjust=0, mood_adjust=0, 
                     diversity=True):
    """Generate playlist from natural language prompt"""
    
    # Encode prompt
    prompt_lyric = encoder.encode(user_prompt)
    prompt_audio = extract_audio_intent(user_prompt, energy_adjust, mood_adjust)
    prompt_mood = extract_mood_intent(user_prompt)
    
    # Combine
    prompt_vector = np.concatenate([prompt_lyric, prompt_audio, prompt_mood])
    
    # Compute similarities
    similarities = cosine_similarity([prompt_vector], song_embeddings)[0]
    ranked_indices = np.argsort(similarities)[::-1]
    
    # Apply diversity filter
    if diversity:
        selected = []
        artist_counts = {}
        max_per_artist = 2
        
        for idx in ranked_indices:
            artist = song_metadata.iloc[idx]['artist']
            if artist_counts.get(artist, 0) < max_per_artist:
                selected.append(idx)
                artist_counts[artist] = artist_counts.get(artist, 0) + 1
            if len(selected) >= top_k:
                break
        final_indices = selected
    else:
        final_indices = ranked_indices[:top_k]
    
    # Build results
    results = song_metadata.iloc[final_indices].copy()
    results['similarity'] = similarities[final_indices]
    results['rank'] = range(1, len(results) + 1)
    
    return results

# ═══════════════════════════════════════════════════════════════════════
# MAIN APP LAYOUT
# ═══════════════════════════════════════════════════════════════════════

# Header
st.markdown("""
<div style='text-align: center; padding: 20px 0;'>
    <h1 style='font-size: 48px; margin-bottom: 10px;'>🎵 MoodSense</h1>
    <p style='font-size: 20px; color: #B3B3B3;'>AI-Powered Music Playlist Generator</p>
    <p style='color: #1DB954; font-size: 14px;'>Using Semantic Search & NLP</p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════
# SIDEBAR - Controls
# ═══════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.header("⚙️ Controls")
    
    # Playlist size
    playlist_size = st.slider("Playlist Size", 5, 50, 20)
    
    # Advanced controls
    with st.expander("🎛️ Advanced Audio Adjustments"):
        energy_boost = st.slider("Energy Boost", -20, 20, 0)
        mood_boost = st.slider("Positiveness Boost", -20, 20, 0)
        diversity_filter = st.checkbox("Diversity Filter (2 songs/artist)", value=True)
    
    st.markdown("---")
    
    # Project info
    st.header("📊 Project Info")
    st.markdown("""
    **Team:** Group 3
    - Ankit Mittal
    - Albin Anto Jose
    - Nandini Bag
    - Kasheena Mulla
    
    **Course:** MS DSP 422  
    **Model:** Sentence-BERT + SVM  
    **Dataset:** 30,000 songs  
    **Embedding Dim:** 396
    """)

# ═══════════════════════════════════════════════════════════════════════
# MAIN CONTENT
# ═══════════════════════════════════════════════════════════════════════

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("💬 Enter Your Music Prompt")
    
    # Main input
    user_prompt = st.text_input(
        "",
        placeholder="E.g., 'energetic workout songs with motivational lyrics'",
        label_visibility="collapsed"
    )
    
    # Example prompts
    st.markdown("**💡 Try these examples:**")
    example_prompts = [
        "energetic workout songs",
        "sad heartbreak songs for crying",
        "romantic love songs for wedding",
        "chill acoustic guitar for studying",
        "aggressive metal for stress release"
    ]
    
    cols = st.columns(len(example_prompts))
    for i, prompt in enumerate(example_prompts):
        if cols[i].button(f"🎵 {prompt.split()[0].title()}", key=f"ex_{i}"):
            user_prompt = prompt
            st.rerun()
    
    st.markdown("---")
    
    # Generate button
    if st.button("🎵 Generate Playlist", type="primary", use_container_width=True):
        if not user_prompt:
            st.warning("⚠️ Please enter a prompt first!")
        else:
            with st.spinner("🎵 Finding perfect songs for you..."):
                playlist = generate_playlist(
                    user_prompt, 
                    top_k=playlist_size,
                    energy_adjust=energy_boost,
                    mood_adjust=mood_boost,
                    diversity=diversity_filter
                )
                
                st.success(f"✅ Found {len(playlist)} songs matching: **'{user_prompt}'**")
                
                # Display playlist
                st.subheader("🎧 Your Playlist")
                
                # Format display
                display_df = playlist[['rank', 'song', 'artist', 'mood', 'similarity', 'energy', 'positiveness']].copy()
                display_df['similarity'] = display_df['similarity'].apply(lambda x: f"{x:.3f}")
                display_df.columns = ['#', 'Song', 'Artist', 'Mood', 'Match', 'Energy', 'Vibe']
                
                st.dataframe(
                    display_df,
                    use_container_width=True,
                    hide_index=True,
                    height=600
                )
                
                # Download button
                csv = playlist.to_csv(index=False)
                st.download_button(
                    label="📥 Download Playlist CSV",
                    data=csv,
                    file_name=f"moodsense_playlist_{user_prompt[:20]}.csv",
                    mime="text/csv"
                )

with col2:
    st.subheader("📊 Insights")
    
    if user_prompt and st.session_state.get('playlist') is not None:
        playlist = st.session_state['playlist']
        
        # Mood distribution
        mood_counts = playlist['mood'].value_counts()
        fig = px.pie(
            values=mood_counts.values,
            names=mood_counts.index,
            title="Mood Distribution",
            color_discrete_sequence=px.colors.sequential.Greens
        )
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='white'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Average stats
        st.metric("Avg Energy", f"{playlist['energy'].mean():.0f}/100")
        st.metric("Avg Vibe", f"{playlist['positiveness'].mean():.0f}/100")
        st.metric("Avg Match Score", f"{playlist['similarity'].mean():.3f}")
    else:
        st.info("💡 Generate a playlist to see insights here!")

# ═══════════════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════════════

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #B3B3B3; padding: 20px;'>
    <p>Built with ❤️ using Sentence-BERT, TF-IDF, and Linear SVM</p>
    <p style='font-size: 12px;'>MS DSP 422: Practical Machine Learning | Group 3</p>
</div>
""", unsafe_allow_html=True)
