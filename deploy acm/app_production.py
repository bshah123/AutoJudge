"""
ACM Problem Difficulty Classifier - Web UI (Optimized Production Version)
"""

from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd
import re
import os
import sys
import lightgbm as lgb

app = Flask(__name__)

# Global variables
MODEL = None
EXTRACTOR = None

# ============================================================================
# FEATURE EXTRACTOR (Must match ACM_rating_judge.ipynb logic)
# ============================================================================

class UltraFastExtractor:
    def __init__(self):
        self.patterns = {
            'latex': re.compile(r'\$[^\$]+\$|\\\w+'),
            'numbers': re.compile(r'\d+'),
            'constraints': re.compile(r'(?:10\^|10\*\*|1e)(\d+)|(\d{4,12})')
        }
        self.keywords = ['dynamic', 'graph', 'tree', 'greedy', 'sort', 'prime', 'segment', 'flow', 'dp', 'bitmask']
    
    def extract(self, text):
        if not text: return np.zeros(17)
        
        text_lower = text.lower()
        words = text.split()
        nums = [int(n) for n in self.patterns['numbers'].findall(text) if len(n)<10]
        
        # Mathematical Complexity Weight
        math_weight = 0
        found_constraints = self.patterns['constraints'].findall(text)
        for pwr, val in found_constraints:
            if pwr:
                p = int(pwr)
                if p >= 8: math_weight = max(math_weight, 3.0)
                elif p >= 5: math_weight = max(math_weight, 2.0)
                else: math_weight = max(math_weight, 1.0)
            elif val:
                v = int(val)
                if v > 1e7: math_weight = max(math_weight, 3.0)
                elif v > 1e4: math_weight = max(math_weight, 2.0)
        
        # Time Limit Parser
        time_limit = 1.0
        time_match = re.search(r'(\d+(?:\.\d+)?)\s*sec', text_lower)
        if time_match:
            try: time_limit = float(time_match.group(1))
            except: pass

        return np.array([
            len(text),
            len(words),
            len(text) / max(len(words), 1),
            len(self.patterns['latex'].findall(text)),
            text.count('$'),
            len(nums),
            max(nums, default=0),
            1 if nums and max(nums) > 1e6 else 0,
            sum(text_lower.count(k) for k in self.keywords),
            text.count('.'),
            text.count('?'),
            1 if 'input' in text_lower else 0,
            1 if 'constraint' in text_lower else 0,
            1 if 'example' in text_lower else 0,
            1 if 'output' in text_lower else 0,
            math_weight,
            time_limit
        ])

# ============================================================================
# MODEL LOADING
# ============================================================================

def load_model():
    global MODEL, EXTRACTOR
    try:
        print("Loading model components from .pkl files...")
        
        # Load individual components saved by joblib in Colab
        clf = joblib.load('clf_model.pkl')
        reg = joblib.load('reg_model.pkl')
        scaler = joblib.load('scaler.pkl')
        
        # If you saved TF-IDF, load it here. Otherwise set to None.
        tfidf = None
        if os.path.exists('tfidf.pkl'):
            tfidf = joblib.load('tfidf.pkl')

        MODEL = {
            'clf': clf,
            'reg': reg,
            'scaler': scaler,
            'tfidf': tfidf
        }
        
        EXTRACTOR = UltraFastExtractor()
        print("✅ Models loaded successfully!")
        return True
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False

# ============================================================================
# PREDICTION LOGIC
# ============================================================================

def get_prediction(desc, input_desc, output_desc):
    full_text = f"{desc} {input_desc} {output_desc}"
    
    # 1. Feature Extraction
    features = EXTRACTOR.extract(full_text).reshape(1, -1)
    
    # 2. Scaling
    if MODEL['scaler']:
        features = MODEL['scaler'].transform(features)
    
    # 3. Model Inference
    class_idx = int(MODEL['clf'].predict(features)[0])
    raw_score = float(MODEL['reg'].predict(features)[0])
    
    # 4. Post-Processing (Clamping Range)
    classes = ['Easy', 'Medium', 'Hard']
    pred_class = classes[class_idx]
    
    if pred_class == 'Easy':
        final_score = min(raw_score, 3.8)
    elif pred_class == 'Hard':
        final_score = max(raw_score, 7.0)
    else: # Medium
        final_score = max(3.9, min(6.9, raw_score))
        
    return pred_class, round(final_score, 1)

# ============================================================================
# FLASK ROUTES
# ============================================================================

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    desc = data.get('description', '')
    input_d = data.get('input_description', '')
    output_d = data.get('output_description', '')
    
    p_class, p_score = get_prediction(desc, input_d, output_d)
    
    return jsonify({
        'difficulty_class': p_class,
        'difficulty_score': p_score
    })

if __name__ == '__main__':
    # Ensure templates folder exists
    if not os.path.exists('templates'):
        os.makedirs('templates')
    
    if load_model():
        # Port for Render deployment
        port = int(os.environ.get("PORT", 5000))
        app.run(host='0.0.0.0', port=port)
