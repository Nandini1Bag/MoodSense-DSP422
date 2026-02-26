# 🎵 MoodSense: AI-Powered Music Playlist System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.31-FF4B4B.svg)](https://streamlit.io/)
[![FAISS](https://img.shields.io/badge/FAISS-Vector_Search-orange.svg)](https://github.com/facebookresearch/faiss)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Course:** MS DSP 422 – Practical Machine Learning  
**Team:** Group 3 (Ankit Mittal, Albin Anto Jose, Nandini Bag, Kasheena Mulla)

> An intelligent music recommendation system that generates personalized playlists from natural language prompts using semantic search, multimodal machine learning, and production-grade vector indexing.

---

## Problem Statement

Traditional music search relies on exact keyword matching and manual tagging. Users often struggle to find music that matches their mood or context without knowing specific song titles or artists.

**MoodSense** solves this by understanding natural language descriptions (e.g., "energetic workout songs with motivational lyrics") and using AI to find matching songs from a database of 50,000+ tracks — in under 5ms using FAISS approximate nearest neighbor search.

---

## Key Features

- **Natural Language Search**: "sad heartbreak songs for crying" → relevant playlist
- **Production Vector Search**: FAISS IndexFlatIP replaces brute-force cosine — scales to 100M songs
- **Multimodal Embeddings**: 806-dim vector combining BERT semantics, audio features, activity context, and mood
- **Mood Classification**: 4-class ensemble classifier (Happy, Sad, Anger, Love) ~68% accuracy
- **Structured Logging**: JSON request logs capturing latency, mood, pool size, top match
- **Interactive Demo**: Streamlit web app with real-time search metrics displayed per query

---

## Architecture

```
User Prompt
    ↓
Prompt Expansion         "happy" → "joyful upbeat cheerful energetic..." (19 trigger words)
    ↓
BERT Encoder             all-mpnet-base-v2 → 768-dim semantic vector
    ↓
Audio Intent Extraction  11-dim keyword-rule vector → StandardScaler
    ↓
Context Mapping          9 activity tags → binary flags (e.g., "driving" → Good_for_Driving=1)
    ↓
Mood Classifier          TF-IDF + Ensemble (LinearSVC 60% + LR 40%) → 4-dim probability vector
    ↓
806-dim Weighted Vector  Lyric×1 · Audio×2 · Context×4 · Mood×5 · L2 normalized
    ↓
FAISS ANN Search         IndexFlatIP · O(log n) · ~1ms for 50K songs
    ↓
Mood Filter              Hard restrict pool to detected mood class
    ↓
Diversity Filter         Max 2 songs per artist
    ↓
Top-K Playlist
```

---

## Results

### Mood Classification — Model Comparison

| Model | Accuracy | F1 Macro |
|---|---|---|
| Original LR (baseline) | ~62% | 0.57 |
| LR + balanced sampling + NLP features | ~65% | 0.60 |
| LinearSVC + balanced + NLP features | ~67% | 0.62 |
| **Ensemble (LinearSVC 60% + LR 40%)** | **~68%** | **0.63** |
| Audio-Only (Random Forest) | 42% | 0.30 |

**Key Finding:** Lyrics carry the emotion signal; audio features alone cannot reliably predict mood. The ensemble gain comes from balanced per-class sampling (15K/mood) + VADER sentiment + keyword counts added as training features.

### Vector Search — Scaling Performance

| Approach | 50K songs | 1M songs | 10M songs |
|---|---|---|---|
| numpy cosine (O(n)) | ~100ms | ~2,000ms | ~20,000ms ❌ |
| **FAISS IndexFlatIP (O(log n))** | **~1ms** | **~5ms** | **~20ms ✅** |

---

## Quick Start

### Run Locally

```bash
# 1. Clone repository
git clone https://github.com/YOUR_USERNAME/MoodSense-DSP422.git
cd MoodSense-DSP422

# 2. Create and activate virtual environment
python3 -m venv venv311_clean
source venv311_clean/bin/activate   # Windows: venv311_clean\Scripts\activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Run notebook to generate models + FAISS index
#    Execute all cells in: notebooks/MoodSense_Complete_Pipeline.ipynb
#    Sections 6.5 (accuracy) and 8.5 (FAISS) are required

# 5. Run the Streamlit app
streamlit run app/moodsense_app.py
```

- Streamlit app: `http://localhost:8501`

---

## Deploying to Streamlit Cloud

The two large embedding files (`song_embeddings_50k.npy` at ~316 MB and `song_index_50k.faiss`) exceed GitHub's file size limits and cannot be committed to the repo. They are hosted on Hugging Face and downloaded automatically on first deploy.

### Step 1 — Install Hugging Face CLI

```bash
pip install huggingface_hub
```

### Step 2 — Create a Hugging Face Repository

1. Go to [https://huggingface.co](https://huggingface.co) and sign in
2. Click **New Model** (top right)
3. Name it `moodsense-embeddings`
4. Set visibility to **Public**
5. Click **Create Model**

### Step 3 — Login via CLI

```bash
huggingface-cli login
```

When prompted, paste your token. To get one:
1. Go to [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. Click **New token** → set role to **Write** → click **Create**
3. Copy and paste into the terminal

>  Never share your token — treat it like a password.

### Step 4 — Upload Embedding Files

Navigate to the `app/` folder first, then upload both files:

```bash
cd app/

hf upload YOUR_HF_USERNAME/moodsense-embeddings song_embeddings_50k.npy
hf upload YOUR_HF_USERNAME/moodsense-embeddings song_index_50k.faiss
```

Replace `YOUR_HF_USERNAME` with your actual Hugging Face username (e.g. `123Nandini`).

Each upload shows a progress bar. When complete, the CLI prints a URL like:
```
https://huggingface.co/YOUR_HF_USERNAME/moodsense-embeddings/blob/main/song_embeddings_50k.npy
```

### Step 5 — Update the App Download URL

In `app/moodsense_app.py`, confirm the `HF_BASE` constant matches your username:

```python
HF_BASE = "https://huggingface.co/YOUR_HF_USERNAME/moodsense-embeddings/resolve/main/"
```

### Step 6 — Confirm requirements.txt includes

```
requests
faiss-cpu
huggingface_hub
```

### Step 7 — Push to GitHub and Deploy

```bash
git add app/moodsense_app.py requirements.txt
git commit -m "Add HuggingFace auto-download for embeddings"
git push origin main
```

Then connect the repo to [https://share.streamlit.io](https://share.streamlit.io).

**What happens on first deploy:**
1. Streamlit installs dependencies
2. App starts and detects embedding files are missing
3. Downloads both files from Hugging Face (~316 MB + FAISS index)
4. Progress message shows, then disappears automatically
5. App loads normally

On every subsequent run the files are already present — no download, instant startup.

---

## Re-generating Embeddings (After Notebook Changes)

If you retrain models or change the embedding dimensions, regenerate and re-upload:

```bash
# Delete old files
rm app/song_embeddings_50k.npy
rm app/song_index_50k.faiss

# Re-run notebook Sections 8 and 8.5 to regenerate

# Re-upload to Hugging Face
cd app/
hf upload YOUR_HF_USERNAME/moodsense-embeddings song_embeddings_50k.npy
hf upload YOUR_HF_USERNAME/moodsense-embeddings song_index_50k.faiss
```

> **Note:** The embedding dimension must match the saved `context_scaler.pkl`. If you see a `ValueError: Incompatible dimension` error, it means the scaler and embeddings were generated in different notebook runs. Re-run Section 8 fully from the top before Section 12 to keep them in sync.

---

## Project Structure

```
MoodSense-DSP422/
├── README.md
├── requirements.txt
├── .streamlit/
│   └── config.toml
├── data/
│   ├── raw/                           # Original Kaggle dataset
│   └── processed/
├── notebooks/
│   └── MoodSense_Complete_Pipeline.ipynb   # 12 sections + 3 new sections
│       ├── Section 6.5 — Accuracy upgrades (balanced sampling, LinearSVC, ensemble, VADER)
│       └── Section 8.5 — FAISS index build + benchmark
├── models/
│   ├── model_text.pkl                 # Ensemble mood classifier
│   ├── tfidf_vectorizer.pkl
│   ├── scaler.pkl
│   ├── context_scaler.pkl
│   ├── label_encoder.pkl
│   └── audio_features.json            # Exact feature list used by scaler
├── app/
│   ├── moodsense_app.py               # Streamlit app with FAISS + structured logging
│   ├── song_embeddings_50k.npy        #  NOT in repo — hosted on Hugging Face
│   ├── song_index_50k.faiss           #️ NOT in repo — hosted on Hugging Face
│   ├── song_metadata_50k.csv          # Song metadata (title, artist, mood, features)
│   └── audio_features.json            # Feature column list for train-serve consistency
└── reports/
    ├── final_report.pdf
    └── presentation.pptx
```

---

## Tech Stack

| Component | Technology |
|---|---|
| **NLP Encoding** | Sentence-BERT (all-mpnet-base-v2, 768-dim) |
| **Mood Classification** | Ensemble: LinearSVC + Logistic Regression (TF-IDF, 20K features) |
| **Sentiment Features** | VADER SentimentIntensityAnalyzer |
| **Vector Search** | FAISS IndexFlatIP (ANN, O(log n)) |
| **Explainability** | SHAP (SHapley Additive exPlanations) |
| **Large File Hosting** | Hugging Face Model Hub |
| **Deployment** | Streamlit Cloud |
| **Version Control** | Git + GitHub |

---

## Dataset

- **Source:** Kaggle — 550K Spotify Songs Dataset
- **Processed Size:** 50,000 songs with 806-dim embeddings
- **Audio Features:** Energy, Danceability, Positiveness, Tempo, Acousticness, Speechiness, Liveness, Instrumentalness
- **Mood Labels:** Mapped from BERT-generated emotions (joy/surprise → Happy, sadness/fear → Sad, anger → Anger, love → Love)
- **Training Balance:** 15,000 songs per mood class (stratified sampling)

---

## Embedding Design

The 806-dim weighted embedding combines four modalities:

| Component | Dims | Weight | Rationale |
|---|---|---|---|
| BERT lyric semantics | 768 | ×1 | Captures meaning; normalized to prevent dimension domination |
| Audio features | 11 | ×2 | Energy, tempo, danceability — core acoustic signal |
| Activity context | 9 | ×4 | "Driving", "workout", "sleep" — highest intent signal |
| Mood probability | 4 | ×5 | Classifier confidence — strongest retrieval filter |

Without per-modality normalization, the 768-dim BERT vector would dominate cosine similarity by sheer size, reducing mood and context to <3% of the signal regardless of their values.

---

## Performance Insights

**From SHAP Analysis:**

| Mood | Top Predictive Words |
|---|---|
| **Happy** | love, smile, happy, celebrate, joy |
| **Sad** | cry, tears, lonely, broken, hurt |
| **Anger** | hate, kill, fight, rage, mad |
| **Love** | love, heart, forever, kiss, baby |

**Most discriminative audio features:** Energy, Positiveness, Acousticness  
**Least useful for mood:** Liveness, Speechiness

---

## Limitations & Future Work

### Current Limitations
- Mood labels are BERT-generated, not human-annotated
- No re-ranking layer (LambdaMART for trending/personalization is a known gap)
- No Redis caching for repeated popular prompts
- No user feedback loop — playlist quality is not measured post-delivery

### Production Gaps (Known)
In production we would add: Kubeflow retraining pipeline, Evidently drift monitoring, a feature store (Feast/Vertex AI) for train-serve consistency, and Redis caching for high-frequency prompts. We focused on validating retrieval quality and embedding design first.

### Future Enhancements
- [ ] User feedback loop (thumbs up/down → personalization layer)
- [ ] Fine-tuned DistilBERT for mood classification (+6-8% accuracy)
- [ ] A/B testing infrastructure for retrieval strategies
- [ ] Spotify OAuth + playlist export
- [ ] Scale to full 550K dataset with Pinecone persistent index
- [ ] Multi-label mood classification

---

## Team

- **Ankit Mittal** —
- **Albin Anto Jose** — 
- **Nandini Bag** —
- **Kasheena Mulla** — 

---

## Acknowledgements

- **Course:** MS DSP 422 — Practical Machine Learning, Northwestern University
- **Dataset:** Kaggle Spotify Tracks Dataset
- **Models:** Sentence-Transformers (Hugging Face), Facebook FAISS

**Built with ❤️ for MS DSP 422 | Northwestern University**
