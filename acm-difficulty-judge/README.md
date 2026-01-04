# AuToJudge Classifier - ACM Problem Difficulty Predictor

A machine learning-powered web application that predicts the difficulty level and score of competitive programming problems using advanced NLP and ensemble learning techniques.

## 📋 Overview

AuToJudge Classifier analyzes problem statements from ACM/competitive programming contests and predicts:
- **Difficulty Classification**: Easy, Medium, or Hard
- **Difficulty Score**: A numerical score (0-10) representing problem complexity

The system combines text embeddings with engineered features to capture both semantic meaning and algorithmic complexity indicators.

## 🌐 Live Demo

**Try it now**: https://huggingface.co/spaces/bshah0596/acm-difficulty-judge

No installation required! Test the classifier directly in your browser.

## ✨ Features

- **Intelligent Text Analysis**: Processes problem descriptions, input constraints, and output specifications
- **LaTeX Preprocessing**: Handles mathematical notation and converts it to semantic representations
- **Smart Feature Engineering**:
  - Text embeddings using SentenceTransformer
  - Mathematical symbol counting
  - Constraint analysis (detects powers like 10^5, 10^8)
  - Keyword frequency for algorithmic concepts (DP, graphs, trees, etc.)
  - Text structure metrics (length, word count, punctuation)
  
- **Ensemble ML Models**:
  - **LightGBM** for classification with leaf-wise tree growth
  - **Gradient Boosting Regressor** for precise difficulty scoring
  - StandardScaler for feature normalization
  
- **Beautiful Web UI**:
  - Glassmorphic design with neon gradients
  - Real-time predictions
  - Responsive layout for desktop and mobile
  - Color-coded difficulty output

## 🏗️ Architecture

### Backend
- **Framework**: Flask
- **Model**: Pickle-serialized ML pipeline
  - Classification model (LightGBM)
  - Regression model (Gradient Boosting)
  - StandardScaler for feature scaling

### Frontend
- **HTML/CSS/JavaScript**: Interactive web interface
- **Styling**: Modern glassmorphic design with CSS gradients
- **API Communication**: Async fetch requests to Flask backend

### Data Flow
```
Problem Text Input
    ↓
Text Preprocessing (LaTeX normalization, cleaning)
    ↓
Feature Extraction (17 features)
    ↓
Embedding (SentenceTransformer - all-MiniLM-L6-v2)
    ↓
Feature Scaling (StandardScaler)
    ↓
Classification (LightGBM) + Regression (Gradient Boosting)
    ↓
Difficulty Class + Score Output
```

## 📊 Model Details

### Training Data
- **Source**: TaskComplexityEval-24 dataset
- **Size**: ~4112 competitive programming problems
- **Classes**: Easy (0), Medium (1), Hard (2)
- **Score Range**: 1.0 - 10.0

### Feature Set (17 dimensions)
1. Text length
2. Word count
3. Average word length
4. LaTeX/Math count
5. Dollar sign count
6. Number count
7. Maximum number
8. Large number flag (>1M)
9. Algorithm keyword frequency
10. Period count
11. Question mark count
12. "Input" keyword presence
13. "Constraint" keyword presence
14. "Example" keyword presence
15. "Output" keyword presence
16. Mathematical weight (constraint complexity)
17. Time limit (default 1.0)

### Embedding
- **Model**: `all-MiniLM-L6-v2` (384 dimensions)
- **Type**: Sentence transformers with normalized embeddings
- **Total Features**: 384 + 17 = 401 features

### Model Performance
- **Classifier (LightGBM)**:
  - Test Accuracy: ~55%
  - Balanced for Easy/Medium/Hard classes
  
- **Regressor (Gradient Boosting)**:
  - Mean Absolute Error (MAE): ~0.5-0.8
  - Captures non-linear difficulty scoring patterns

## 🚀 Installation

### Quick Start (Online)

👉 **Try the live demo first**: https://huggingface.co/spaces/bshah0596/acm-difficulty-judge

No setup needed—just visit the link and start analyzing problems!

### Local Setup

#### Prerequisites
- Python 3.8+
- pip or conda

#### Setup

1. **Clone the repository**
   ```bash
   git clone <https://github.com/bshah123/AutoJudge.git>
   cd acm-difficulty-judge
   ```
2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify model file**
   - Ensure `improved_model.pkl` is in the project root
   - If missing, download from the training notebook output

4. **Run the Flask application**
   ```bash
   python app_production.py
   ```

5. **Access the web interface**
   - Open browser to `http://localhost:7860`

### Docker Deployment
   ```bash
   docker build -t autojudge-classifier .
   docker run -p 7860:7860 autojudge-classifier
   ```

## 💻 Usage

### Web Interface
1. Enter problem description in the first text field
2. (Optional) Add input constraints/description
3. (Optional) Add output format description
4. Click **"ANALYZE PROBLEM"**
5. View results:
   - Classification (Easy/Medium/Hard)
   - Numerical difficulty score

### API Endpoint

**POST** `/predict`

**Request:**
```json
{
  "description": "Problem statement...",
  "input_description": "N constraints...",
  "output_description": "Output format..."
}
```

**Response:**
```json
{
  "difficulty_class": "Hard",
  "difficulty_score": 7.5
}
```

## 📁 Project Structure

```
acm-difficulty-judge/
├── app_production.py              # Flask application
├── Final_Submission_ACM.ipynb    # Training notebook with full ML pipeline
├── improved_model.pkl            # Serialized ML model (not in repo)
├── requirements.txt              # Python dependencies
├── Dockerfile                    # Docker configuration
├── templates/
│   └── index.html               # Web UI
└── README.md                    # This file
```

## 📦 Dependencies

- **Flask**: Web framework
- **scikit-learn**: ML utilities (StandardScaler, etc.)
- **LightGBM**: Gradient boosting classifier
- **CatBoost**: Alternative classifier (for training)
- **sentence-transformers**: Text embeddings
- **numpy, pandas**: Data processing
- **matplotlib**: Visualization (training)
- **joblib, pickle**: Model serialization

See `requirements.txt` for complete list with versions.

## 🔬 Training Process

The Jupyter notebook (`Final_Submission_ACM.ipynb`) contains:

1. **Data Loading**: Fetches TaskComplexityEval-24 dataset
2. **Preprocessing**: 
   - LaTeX normalization
   - Text cleaning
   - Missing value handling
3. **Feature Engineering**:
   - Text embeddings
   - Constraint analysis
   - Keyword detection
4. **Model Selection**:
   - Compared: Logistic Regression, SVM (RBF/Poly), LightGBM, CatBoost
   - Selected: LightGBM (best test accuracy)
5. **Regression Models**:
   - Compared: Ridge, Gradient Boosting, LightGBM
   - Selected: Gradient Boosting (best MAE)
6. **Model Export**: Saves ensemble to `improved_model.pkl`

## 🎨 UI Design

The web interface features:
- **Color Scheme**: Dark theme with neon blue (#00d2ff) and purple (#9d50bb)
- **Glassmorphism**: Frosted glass effect with backdrop blur
- **Responsive**: Adapts to mobile and desktop
- **Animations**: Smooth transitions and fade-in effects
- **Accessibility**: Clear labels and intuitive layout

### Difficulty Colors
- **Easy**: Green (#00ff88)
- **Medium**: Yellow (#ffcc00)
- **Hard**: Orange-Red (#ff4b2b)

## 🔍 Key Insights

- **Text Length Matters**: Longer problem statements tend to be harder
- **Mathematical Content**: Presence of complex constraints (10^8, etc.) strongly indicates difficulty
- **Keywords Are Predictive**: Words like "dynamic", "graph", "dp" correlate with harder problems
- **LaTeX Usage**: More mathematical notation suggests higher difficulty
- **Ensemble Power**: Combining embedding-based and symbolic features outperforms either alone

## 🛠️ Troubleshooting

**Model not loading?**
- Check if `improved_model.pkl` exists in project root
- Verify Python version compatibility (3.8+)
- Check file permissions

**Predictions seem off?**
- Ensure problem text is reasonably detailed
- Include constraint information for better accuracy
- Note: Model trained on ACM-style problems; other formats may have lower accuracy

**Slow predictions?**
- First request may be slower (model loading)
- Subsequent requests should be near-instant
- GPU acceleration available with CUDA support

## 📈 Future Improvements

- [ ] Fine-tuning on larger datasets
- [ ] Multi-language support
- [ ] Batch prediction API
- [ ] Model interpretability (feature importance visualization)
- [ ] Real-time performance metrics dashboard
- [ ] Support for different problem types (HackerRank, Codeforces, etc.)

## 👨‍💻 Author
~ Bhavya Shah 
Created for ACM Open Project (AutoJudge) 
## 🙏 Acknowledgments

- **Dataset**: TaskComplexityEval-24 by AREEG94FAHAD
- **Libraries**: scikit-learn, LightGBM, sentence-transformers communities,Numpy,Panda,etc

---

**Happy Problem Solving! 🚀**

