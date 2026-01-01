"""
ACM Problem Difficulty Classifier - Web UI
Simplified version that works with your trained model
"""

from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd
import re
import os
import sys

app = Flask(__name__)

# Global variables
MODEL = None
EXTRACTOR = None

print("="*70)
print("  ACM PROBLEM DIFFICULTY CLASSIFIER - WEB UI")
print("="*70)
print()

# ============================================================================
# FEATURE EXTRACTOR (must match training)
# ============================================================================
class UltraFastExtractor:
    def __init__(self):
        self.patterns = {
            'latex': re.compile(r'\$[^\$]+\$|\\\w+'),
            'numbers': re.compile(r'\d+'),
            # Specifically find powers of 10 or large constraints
            'constraints': re.compile(r'(?:10\^|10\*\*|1e)(\d+)|(\d{4,12})')
        }
        self.keywords = ['dynamic', 'graph', 'tree', 'greedy', 'sort', 'prime', 'segment', 'flow', 'dp']
    
    def extract(self, text):
        if not text: return np.zeros(17) # Increased size to 17
        
        text_lower = text.lower()
        words = text.split()
        nums = [int(n) for n in self.patterns['numbers'].findall(text) if len(n)<10]
        
        # New Feature: Expected Complexity Weight
        # We look for 10^5, 10^9 etc. and map to a difficulty "bonus"
        constraint_score = 0
        found_constraints = self.patterns['constraints'].findall(text)
        for pwr, val in found_constraints:
            if pwr: # if 10^5 format
                p = int(pwr)
                if p >= 8: constraint_score = max(constraint_score, 3.0) # Hard (O(N) or O(1))
                elif p >= 5: constraint_score = max(constraint_score, 2.0) # Med-Hard (O(N log N))
                else: constraint_score = max(constraint_score, 1.0)
            elif val: # if 100000 format
                v = int(val)
                if v > 1e7: constraint_score = max(constraint_score, 3.0)
                elif v > 1e4: constraint_score = max(constraint_score, 2.0)
        
        # New Feature: Time Limit sensitivity
        time_limit = 1.0
        if 'sec' in text_lower:
            try:
                # Find the number immediately before "sec"
                time_match = re.search(r'(\d+(?:\.\d+)?)\s*sec', text_lower)
                if time_match: time_limit = float(time_match.group(1))
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
            constraint_score, # Feature 16: Mathematical complexity
            time_limit        # Feature 17: Time budget
        ])
# ============================================================================
# MODEL LOADING
# ============================================================================

def load_model():
    """Load the trained model from pickle file"""
    global MODEL, EXTRACTOR
    
    model_path = 'optimized_model.pkl'
    
    if not os.path.exists(model_path):
        print(f"❌ ERROR: {model_path} not found!")
        return False
    
    try:
        with open(model_path, 'rb') as f:
            MODEL = pickle.load(f)
        
        EXTRACTOR = UltraFastExtractor()
        print(f"✅ Model loaded successfully from {model_path}")
        return True
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        return False

# ============================================================================
# PREDICTION
# ============================================================================
def predict(title, description, input_desc, output_desc):
    if MODEL is None:
        return None, None, "Model not loaded"
    
    try:
        text = f"{title} {description} {input_desc} {output_desc}"
        text = re.sub(r'\s+', ' ', text).strip()
        
        # 1. Extract features (Ensure this matches your 16-feature version)
        features = EXTRACTOR.extract(text).reshape(1, -1)
        
        tfidf = MODEL.get('tfidf')
        if tfidf:
            tfidf_feat = tfidf.transform([text]).toarray()
            features = np.hstack([features, tfidf_feat])
        
        scaler = MODEL.get('scaler')
        if scaler:
            features = scaler.transform(features)
        
        # 2. Get Raw Predictions
        clf = MODEL.get('clf')
        reg = MODEL.get('reg')
        
        class_idx = int(clf.predict(features)[0])
        raw_score = float(reg.predict(features)[0])
        
        # 3. Apply Post-Processing Rules (The "Clamping" Logic)
        # Class 0: Easy, Class 1: Medium, Class 2: Hard
        if class_idx == 0:  # EASY
            # Ensure Easy doesn't exceed 3.8
            final_score = min(raw_score, 3.8)
            # If the regressor was way off, give it a baseline Easy score
            if final_score > 3.8: final_score = 3.0 
            
        elif class_idx == 2:  # HARD
            # Ensure Hard is at least 7.0
            final_score = max(raw_score, 7.0)
            # If the regressor was way off, give it a baseline Hard score
            if final_score < 7.0: final_score = 7.5
            
        else:  # MEDIUM
            # Keep Medium in the 3.9 to 6.9 range
            final_score = max(3.9, min(6.9, raw_score))
        
        classes = ['Easy', 'Medium', 'Hard']
        return classes[class_idx], round(final_score, 1), None
        
    except Exception as e:
        return None, None, str(e)
# ============================================================================
# ROUTES
# ============================================================================

@app.route('/')
def home():
    """Main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_route():
    """Prediction endpoint"""
    try:
        data = request.json
        
        title = data.get('title', '').strip()
        description = data.get('description', '').strip()
        input_desc = data.get('input_description', '').strip()
        output_desc = data.get('output_description', '').strip()
        
        if not description:
            return jsonify({
                'success': False,
                'error': 'Description is required'
            })
        
        # Make prediction
        difficulty_class, difficulty_score, error = predict(
            title, description, input_desc, output_desc
        )
        
        if error:
            return jsonify({
                'success': False,
                'error': error
            })
        
        combined_text = f"{title} {description} {input_desc} {output_desc}"
        
        return jsonify({
            'success': True,
            'difficulty_class': difficulty_class,
            'difficulty_score': difficulty_score,
            'text_length': len(combined_text.strip())
        })
        
    except Exception as e:
        import traceback
        return jsonify({
            'success': False,
            'error': f"Server error: {str(e)}\n{traceback.format_exc()}"
        })

@app.route('/health')
def health():
    """Health check"""
    return jsonify({
        'status': 'ok',
        'model_loaded': MODEL is not None
    })

# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print("\n📁 Current directory:", os.getcwd())
    print("📂 Files here:", os.listdir('.'))
    
    # Check for templates folder
    if not os.path.exists('templates'):
        print("\n⚠️  WARNING: 'templates' folder not found!")
        print("Creating 'templates' folder...")
        os.makedirs('templates')
        print("✓ Created 'templates' folder")
        print("\n❗ Please move index.html into the templates folder:")
        print("   mv index.html templates/")
        sys.exit(1)
    
    # Check for index.html
    if not os.path.exists('templates/index.html'):
        print("\n❌ ERROR: templates/index.html not found!")
        print("\nPlease move your index.html file into the templates folder:")
        print("  mv index.html templates/")
        sys.exit(1)
    
    # Load model
    if not load_model():
        print("\n❌ Cannot start server without model!")
        print("\nSteps to fix:")
        print("1. Download 'optimized_model.pkl' from Colab")
        print("2. Place it in the same folder as this script")
        print("3. Run again: python app_production.py")
        sys.exit(1)
    
    # Start server
    print("\n" + "="*70)
    print("✅ SERVER READY!")
    print("="*70)
    print("\n🌐 Open your browser and go to:")
    print("   http://localhost:5000")
    print("   or")
    print("   http://127.0.0.1:5000")
    print("\n⌨️  Press Ctrl+C to stop the server")
    print("="*70)
    print()
    
    try:
        app.run(debug=False, host='0.0.0.0', port=5000)
    except OSError as e:
        if "Address already in use" in str(e):
            print("\n❌ Port 5000 is already in use!")
            print("\nTrying port 5001...")
            app.run(debug=False, host='0.0.0.0', port=5001)
        else:
            raise