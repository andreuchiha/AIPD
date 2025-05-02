"""
Run Inference Module

This module provides a command-line interface for running phishing detection
on email files. It loads the trained model and displays predictions with
explanations using LIME.
"""

import sys

# Handle both package and standalone imports
try:
    # Try relative imports (for package usage)
    from .inference import load_artifacts, predict_email
except ImportError:
    # Fall back to absolute imports (for standalone usage)
    from inference import load_artifacts, predict_email

# Configuration
EMAIL_FILE = "./emails/hacked.txt"  # Path to email file for analysis
MODE = "chunk"  # Use chunk-based analysis for better accuracy
NUM_SAMPLES = 500  # Number of samples for LIME explanation

def main():
    """
    Main function to run phishing detection on an email file.
    Loads model artifacts, makes predictions, and displays results with explanations.
    """
    try:
        with open(EMAIL_FILE, "r", encoding="utf-8", errors="ignore") as f:
            raw = f.read()
    except FileNotFoundError:
        print(f"Error: file not found: {EMAIL_FILE}")
        sys.exit(1)

    vec, clf = load_artifacts()

    label, conf, explanation = predict_email(
        raw,
        vec,
        clf,
        mode=MODE,     
        chunk_size=500,
        explain=True,
        num_samples=NUM_SAMPLES
    )
    verdict = "Phishing" if label == 1 else "Legitimate"

    print(f"\nResult:     {verdict}")
    if conf is not None:
        print(f"Confidence: {conf:.2%}")
    
    if explanation:
        print("\nKey phrases that contributed to the prediction:")
        for word, weight in explanation:
            weight_pct = f"{weight:.1%}"
            if weight > 0:
                print(f"  +{weight_pct}: {word}")
            else:
                print(f"  {weight_pct}: {word}")

if __name__ == "__main__":
    main()
