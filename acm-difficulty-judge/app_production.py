import os
import re
import pickle
import numpy as np
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# ============================================================================
# FEATURE EXTRACTOR
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
        
        math_weight = 0
        found_constraints = self.patterns['constraints'].findall(text)
        for pwr, val in found_constraints:
            if pwr:
                p = int(pwr)
                if p >= 8: math_weight = max(math_weight, 3.0)
                elif p >= 5: math_weight = max(math_weight, 2.0)
            elif val:
                v = int(val)
                if v > 1e7: math_weight = max(math_weight, 3.0)
        
        time_limit = 1.0
        time_match = re.search(r'(\d+(?:\.\d+)?)\s*sec', text_lower)
        if time_match:
            try: time_limit = float(time_match.group(1))
            except: pass

        return np.array([
            len(text), len(words), len(text) / max(len(words), 1),
            len(self.patterns['latex'].findall(text)), text.count('$'),
            len(nums), max(nums, default=0), 1 if nums and max(nums) > 1e6 else 0,
            sum(text_lower.count(k) for k in self.keywords),
            text.count('.'), text.count('?'),
            1 if 'input' in text_lower else 0, 1 if 'constraint' in text_lower else 0,
            1 if 'example' in text_lower else 0, 1 if 'output' in text_lower else 0,
            math_weight, time_limit
        ])

# ============================================================================
# GLOBAL LOADING (This must happen OUTSIDE any functions)
# ============================================================================

MODEL = None
EXTRACTOR = UltraFastExtractor() # Initialize immediately so it's never None

def initialize_app():
    global MODEL
    model_path = 'improved_model.pkl'
    if os.path.exists(model_path):
        try:
            with open(model_path, 'rb') as f:
                MODEL = pickle.load(f)
            print("✅ Model loaded successfully!")
        except Exception as e:
            print(f"❌ Model load error: {e}")
    else:
        print(f"❌ {model_path} NOT FOUND!")

# Call this NOW so it runs when Hugging Face starts the app
initialize_app()

# ============================================================================
# ROUTES
# ============================================================================

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if MODEL is None:
        return jsonify({"error": "Model not loaded on server"}), 500
        
    try:
        data = request.get_json()
        desc = data.get('description', '')
        input_d = data.get('input_description', '')
        output_d = data.get('output_description', '')
        
        full_text = f"{desc} {input_d} {output_d}"
        
        # EXTRACTOR is already initialized at the top level
        features = EXTRACTOR.extract(full_text).reshape(1, -1)
        
        if MODEL['scaler']:
            features = MODEL['scaler'].transform(features)
        
        class_idx = int(MODEL['clf'].predict(features)[0])
        raw_score = float(MODEL['reg'].predict(features)[0])
        
        classes = ['Easy', 'Medium', 'Hard']
        pred_class = classes[class_idx]
        
        # Clamping logic
        if pred_class == 'Easy': final_score = min(raw_score, 3.8)
        elif pred_class == 'Hard': final_score = max(raw_score, 7.0)
        else: final_score = max(3.9, min(6.9, raw_score))
            
        return jsonify({
            'difficulty_class': pred_class,
            'difficulty_score': round(final_score, 1)
        })
    except Exception as e:
        print(f"PREDICT ERROR: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860)