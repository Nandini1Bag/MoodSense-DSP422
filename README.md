# MoodSense: Intelligent Music Mood Classification  
**Course:** MS DSP 422 – Practical Machine Learning  
**Team:** Group 3  
**Members:** Ankit Mittal, Albin Anto Jose, Nandini Bag, Kasheena Mulla  

---

## Project Overview
Music streaming platforms contain vast libraries of songs, making emotion-based organization and discovery increasingly important. Traditional metadata such as genre and artist do not fully capture the emotional characteristics of music.  

**MoodSense** is a supervised machine learning project that classifies songs into distinct mood categories—**Happy, Sad, Energetic, and Calm**—using Spotify audio features. The project focuses on classical machine learning models, systematic feature encoding, and explainable AI techniques to ensure both predictive performance and interpretability.

This project is designed as a practical ML pipeline suitable for educational and research-oriented applications, and serves as a foundation for future mood-aware recommendation systems.

---

## Problem Statement
Can we accurately classify the mood of a song using audio-based features and traditional machine learning models, while maintaining interpretability and scalability?

---

## Dataset
### Primary Dataset
- **Spotify Tracks Dataset**
- Source: Kaggle  
- Size: 114,000+ tracks  
- Features used:
  - Valence
  - Energy
  - Danceability
  - Acousticness
  - Instrumentalness
  - Speechiness
  - Tempo
  - Loudness

Spotify does not provide explicit mood labels. Therefore, mood categories are generated using **rule-based and weak supervision techniques** derived from audio feature patterns.

---

## Mood Categories
- Happy  
- Sad  
- Energetic  
- Calm  

---

## Project Structure
MoodSense-DSP422/
│
├── data/
│ ├── raw/ # Original datasets (not pushed if large)
│ └── processed/ # Cleaned and labeled data
│
├── notebooks/
│ ├── 01_data_exploration.ipynb
│ ├── 02_label_generation.ipynb
│ ├── 03_feature_engineering.ipynb
│ ├── 04_model_training.ipynb
│ └── 05_explainability.ipynb
│
├── models/ # Saved trained models
├── reports/ # Project report and figures
└── README.md

yaml
Copy code

---

## Methodology
1. **Data Exploration & Cleaning**
   - Analyze feature distributions and correlations
   - Handle missing values and duplicates

2. **Synthetic Mood Label Generation**
   - Rule-based mapping using valence, energy, tempo, and acousticness
   - Validation through manual inspection

3. **Feature Engineering**
   - Feature scaling using StandardScaler
   - Dimensionality reduction using PCA/SVD

4. **Model Training**
   - Logistic Regression
   - Random Forest
   - Gradient Boosting
   - Neural Network (MLP)

5. **Evaluation**
   - Accuracy
   - Precision, Recall, F1-score
   - Confusion matrices

6. **Explainability**
   - SHAP analysis for feature contribution and interpretability

---

## Tools & Technologies
- Python  
- Google Colab  
- GitHub  
- pandas, numpy  
- scikit-learn  
- SHAP  
- matplotlib / seaborn  

---

## Collaboration Workflow
- GitHub is used for version control and collaboration
- Google Colab is used for notebook execution
- Each notebook has a designated owner to avoid conflicts

---

## Results & Deliverables
- Labeled mood classification dataset
- Multiple trained ML models with performance comparison
- SHAP-based explainability analysis
- Well-documented Jupyter notebooks
- Final project report and presentation

---

## Limitations
- Mood labels are synthetically generated and subject to interpretation
- Lyrics and user-context modeling are limited in this phase
- Real-time Spotify integration is not included

---

## Future Work
- NLP-based user prompt understanding
- Hybrid recommendation systems
- Multi-label mood classification
- Real-time personalization using user feedback

---

## How to Run
1. Open notebooks using **Google Colab**
2. Mount Google Drive for dataset access (if needed)
3. Run notebooks sequentially from `01` to `05`

---

## Acknowledgements
This project is developed as part of **MS DSP 422 – Practical Machine Learning** and is intended for academic and educational purposes.
