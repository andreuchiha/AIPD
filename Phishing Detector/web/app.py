from flask import Flask, render_template, request, jsonify
import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.inference import load_artifacts, predict_email

app = Flask(__name__)

# Load all models at startup
rf_vec, rf_clf, _, _ = load_artifacts('rf')
nb_vec, nb_clf, nb_selector, nb_features = load_artifacts('nb')
lr_vec, lr_clf, lr_selector, lr_features = load_artifacts('lr')
gb_vec, gb_clf, gb_selector, gb_features = load_artifacts('gb')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    text = data.get('text', '')
    model_type = data.get('model_type', 'nb')
    show_lime = data.get('show_lime', True)  # Default to True for backward compatibility
    
    # Select the appropriate model and its components
    if model_type == 'nb':
        vec, clf = nb_vec, nb_clf
        selector = nb_selector
        selected_features = nb_features
    elif model_type == 'lr':
        vec, clf = lr_vec, lr_clf
        selector = lr_selector
        selected_features = lr_features
    elif model_type == 'gb':
        vec, clf = gb_vec, gb_clf
        selector = gb_selector
        selected_features = gb_features
    else:  # rf
        vec, clf = rf_vec, rf_clf
        selector = None
        selected_features = None
    
    # Get prediction and explanation
    label, conf, lime_explanation, urgency_explanation = predict_email(
        text,
        vec,
        clf,
        selector=selector,
        selected_features=selected_features,
        model_type=model_type,
        explain=show_lime,  # Only generate LIME explanation if toggle is on
        num_samples=200
    )
    
    # Format the LIME explanation for the frontend (only if enabled)
    formatted_lime = []
    if show_lime and lime_explanation:
        for word, weight in lime_explanation:
            formatted_lime.append({
                'word': word,
                'weight': float(weight),  # Convert numpy float to Python float
                'is_positive': weight > 0
            })
    
    # Format the urgency explanation for the frontend
    formatted_urgency = []
    if urgency_explanation:
        for indicator, weight in urgency_explanation:
            formatted_urgency.append({
                'word': indicator,
                'weight': float(weight),  # Convert numpy float to Python float
                'is_positive': weight > 0
            })
    
    return jsonify({
        'is_phishing': bool(label),
        'confidence': float(conf) if conf is not None else None,
        'lime_explanation': formatted_lime,
        'urgency_explanation': formatted_urgency,
        'model_type': model_type
    })

if __name__ == '__main__':
    app.run(debug=True) 