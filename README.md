# 🎵 MoodSense: AI-Powered Music Playlist System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.31-FF4B4B.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Course:** MS DSP 422 – Practical Machine Learning  
**Team:** Group 3 (Ankit Mittal, Albin Anto Jose, Nandini Bag, Kasheena Mulla)

> An intelligent music recommendation system that generates personalized playlists from natural language prompts using semantic search and multimodal machine learning.

---

##  Problem Statement

Traditional music search relies on exact keyword matching and manual tagging. Users often struggle to find music that matches their mood or context without knowing specific song titles or artists.

**MoodSense** solves this by understanding natural language descriptions (e.g., "energetic workout songs with motivational lyrics") and using AI to find matching songs from a database of 30,000+ tracks.

---

##  Key Features

-  **Natural Language Search**: "sad heartbreak songs for crying" → relevant playlist
-  **AI-Powered Matching**: Sentence-BERT semantic understanding + audio features
-  **Mood Classification**: 4-class classifier (Happy, Sad, Anger, Love) with 60% accuracy
-  **Interactive Demo**: Streamlit web app with Spotify-inspired UI
-  **Explainable AI**: SHAP analysis showing which words influence predictions
-  **Real-time Inference**: <2 second response for 30K song database

---

##  Architecture

```
User Prompt → Sentence-BERT (384-dim) 
              ↓
Audio Intent Extraction (8-dim)
              ↓
Mood Prediction (4-dim)
              ↓
Combined Vector (396-dim)
              ↓
Cosine Similarity with 30K Songs
              ↓
Top 20 Ranked Results
```

---
Steps:

1.Make sure your virtual environment is activated:
python3 -m venv venv311_clean
source venv311_clean/bin/activate

2.Install dependencies:

pip install --upgrade pip
pip install -r requirements.txt

3.Run the notebook(run by cell or anything)

4.Run Streamlit app

streamlit run app/moodsense_app.py


##  Results

### Phase 1: Mood Classification

| Model | Accuracy | F1 Macro |
|---|---|---|
| **Text-Only (TF-IDF + LinearSVC)** | **60.45%** | 0.55 |
| Combined (Text + Audio) | 60.62% | 0.56 |
| Audio-Only (Random Forest) | 42.03% | 0.30 |

**Key Finding:** Lyrics carry all emotion signal; audio features alone cannot predict lyric-based moods.

### Phase 2: AI-Prompted Playlists

-  Generated 396-dim embeddings for 30,000 songs
- Built semantic retrieval system with keyword-based intent extraction
- Demonstrated playlist generation from 5 diverse test prompts
- Implemented artist diversity filtering

---

##  Quick Start

### Try the Live Demo

 **[Launch MoodSense App](https://moodsense-demo.streamlit.app)** *(deploy first)*

### Run Locally

```bash
# 1. Clone repository
git clone https://github.com/YOUR_USERNAME/MoodSense-DSP422.git
cd MoodSense-DSP422

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download data files (from Google Drive)
# Place in app/:
#   - song_embeddings_30k.npy
#   - song_metadata_30k.csv

# 4. Run Streamlit app
cd app
streamlit run moodsense_app.py
```

App opens at `http://localhost:8501`

---

## Project Structure

```
MoodSense-DSP422/
├── README.md 
├── requirements.txt   
├── .streamlit/ 
│   ├──config.toml                    
├── data/
│   ├── raw/                          # Original datasets
│   └── processed/                    
├── notebooks/
│   ├── MoodSense_Complete_Pipeline.ipynb  # Main notebook
│   └── ..                      # Saved ML models
│   ├── model_text.pkl
│   ├── scaler.pkl
│   └── tfidf_vectorizer.pkl
├── app/                              # Streamlit demo
│   ├── moodsense_app.py
|.  |--- # Embeddings & metadata
└── reports/                          # Project deliverables
    ├── final_report.pdf
    └── presentation.pptx.
├── models/     
```

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| **NLP** | Sentence-BERT (all-MiniLM-L6-v2) |
| **Classification** | LinearSVC, Random Forest, Gradient Boosting, MLP |
| **Feature Extraction** | TF-IDF, StandardScaler, TruncatedSVD |
| **Explainability** | SHAP (SHapley Additive exPlanations) |
| **Frontend** | Streamlit with custom CSS |
| **Deployment** | Streamlit Cloud (free tier) |
| **Version Control** | Git + GitHub (with Git LFS) |

---

##  Dataset

- **Source:** Kaggle - 550K Spotify Songs Dataset
- **Size:** 551,443 songs with audio features + BERT-generated emotions
- **Processed:** 30,000 songs with 396-dim embeddings
- **Features:** Energy, Danceability, Positiveness, Tempo, Acousticness, Speechiness, Liveness, Instrumentalness

**Note:** Original emotions (joy, sadness, anger, fear, love, surprise) were mapped to 4 moods (Happy, Sad, Anger, Love) for better classification.

---

##  Methodology

### Phase 1: Mood Classification

1. **Data Cleaning:** Remove noise labels, merge duplicates, 6→4 mood mapping
2. **Feature Engineering:** TF-IDF (20K features, bigrams), audio feature scaling
3. **Model Training:** 5 classical ML models with stratified train/test split
4. **Evaluation:** Classification reports, confusion matrices, SHAP analysis

### Phase 2: Semantic Playlist Generation

1. **Embedding Generation:** Sentence-BERT + audio + mood predictions (396-dim)
2. **Prompt Encoding:** Keyword-based audio/mood intent extraction
3. **Similarity Matching:** Cosine similarity ranking
4. **Post-processing:** Artist diversity filtering (max 2 songs/artist)

---

## Performance Insights

**From SHAP Analysis:**

| Mood | Top Predictive Words |
|---|---|
| **Happy** | love, smile, happy, celebrate, joy |
| **Sad** | cry, tears, lonely, broken, hurt |
| **Anger** | hate, kill, fight, rage, mad |
| **Love** | love, heart, forever, kiss, baby |

**Audio Features:**
- Most discriminative: `Energy`, `Positiveness`, `Acousticness`
- Least useful: `Liveness`, `Speechiness` (for mood classification)

---

## Educational Value

This project demonstrates:

-  End-to-end ML pipeline (data → model → deployment)
-  Multimodal learning (text + audio features)
-  Semantic search with neural embeddings
-  Classical ML model comparison
-  Explainable AI techniques (SHAP)
-  Production deployment (Streamlit Cloud)
-  Git workflow with large files (Git LFS)

---

##  Limitations & Future Work

### Current Limitations
- Mood labels are BERT-generated, not human-annotated
- Audio features alone cannot predict text-based moods (42% accuracy)
- Keyword-based intent extraction is simplistic
- No real-time Spotify API integration

### Future Enhancements
- [ ] Train prompt→mood classifier (replace keyword extraction)
- [ ] Add user feedback loop to improve rankings
- [ ] Scale to full 550K dataset
- [ ] Implement A/B testing for ranking algorithms
- [ ] Add Spotify OAuth + playlist export
- [ ] Multi-label mood classification
- [ ] Personalization based on listening history

---

##  Citation

If you use this project in your research or work, please cite:

```bibtex
@misc{moodsense2024,
  title={MoodSense: AI-Powered Music Playlist Generation},
  author={Mittal, Ankit and Jose, Albin Anto and Bag, Nandini and Mulla, Kasheena},
  year={2024},
  course={MS DSP 422 - Practical Machine Learning}
}
```

---

##  Team

- **Ankit Mittal** - [LinkedIn](#) | [GitHub](#)
- **Albin Anto Jose** - [LinkedIn](#) | [GitHub](#)
- **Nandini Bag** - [LinkedIn](#) | [GitHub](#)
- **Kasheena Mulla** - [LinkedIn](#) | [GitHub](#)

---

##  License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

---

##  Acknowledgements

- **Course:** MS DSP 422 - Practical Machine Learning
- **Dataset:** Kaggle Spotify Tracks Dataset
- **Models:** Sentence-Transformers library (Hugging Face)
- **Deployment:** Streamlit Cloud

---

##  Contact

For questions or collaboration opportunities:
- **Email:** [your-email@example.com]
- **Project Link:** [https://github.com/YOUR_USERNAME/MoodSense-DSP422](https://github.com/YOUR_USERNAME/MoodSense-DSP422)

---

**Built with ❤️ for MS DSP 422 | Northwestern University**