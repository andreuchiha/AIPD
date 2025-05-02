import unittest
import numpy as np
import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.inference import predict_email, load_artifacts
from src.data_preprocessing import clean_text
from src.feature_engineering import extract_urgency_features_batch
import pandas as pd

class TestPhishingDetector(unittest.TestCase):
    def setUp(self):
        # Load test models
        self.nb_vec, self.nb_clf, self.nb_selector, self.nb_features = load_artifacts('nb')
        self.rf_vec, self.rf_clf, _, _ = load_artifacts('rf')
        self.lr_vec, self.lr_clf, self.lr_selector, self.lr_features = load_artifacts('lr')
        self.gb_vec, self.gb_clf, self.gb_selector, self.gb_features = load_artifacts('gb')
        
        # Test emails
        self.phishing_email = """
        URGENT: Your account has been compromised!
        Click here immediately to verify your identity: http://suspicious-link.com
        Your account will be suspended if you don't act now!
        """
        
        self.legitimate_email = """
        Dear valued customer,
        Thank you for your recent purchase. Your order #12345 has been confirmed.
        Best regards,
        Customer Service Team
        """

    def test_model_loading(self):
        """Test that all models load correctly"""
        self.assertIsNotNone(self.nb_clf)
        self.assertIsNotNone(self.rf_clf)
        self.assertIsNotNone(self.lr_clf)
        self.assertIsNotNone(self.gb_clf)

    def test_text_cleaning(self):
        """Test text cleaning functionality"""
        cleaned = clean_text(self.phishing_email)
        self.assertNotIn('http://suspicious-link.com', cleaned)
        self.assertTrue(cleaned.islower())
        self.assertNotIn('\n', cleaned)

    def test_feature_extraction(self):
        """Test feature extraction pipeline"""
        urgency_features = extract_urgency_features_batch([self.phishing_email])
        self.assertIsNotNone(urgency_features)
        self.assertTrue(isinstance(urgency_features, pd.DataFrame))

    def test_prediction_consistency(self):
        """Test that predictions are consistent across models"""
        # Test phishing email
        nb_result = predict_email(self.phishing_email, self.nb_vec, self.nb_clf, 
                                self.nb_selector, self.nb_features, 'nb')
        rf_result = predict_email(self.phishing_email, self.rf_vec, self.rf_clf, 
                                None, None, 'rf')
        
        # Both models should predict phishing
        self.assertEqual(nb_result[0], rf_result[0])
        
        # Test legitimate email
        nb_result = predict_email(self.legitimate_email, self.nb_vec, self.nb_clf, 
                                self.nb_selector, self.nb_features, 'nb')
        rf_result = predict_email(self.legitimate_email, self.rf_vec, self.rf_clf, 
                                None, None, 'rf')
        
        # Both models should predict legitimate
        self.assertEqual(nb_result[0], rf_result[0])

    def test_confidence_scores(self):
        """Test confidence score generation"""
        result = predict_email(self.phishing_email, self.nb_vec, self.nb_clf, 
                             self.nb_selector, self.nb_features, 'nb')
        confidence = result[1]
        
        self.assertIsNotNone(confidence)
        self.assertTrue(0 <= confidence <= 1)

    def test_lime_explanation(self):
        """Test LIME explanation generation"""
        result = predict_email(self.phishing_email, self.nb_vec, self.nb_clf, 
                             self.nb_selector, self.nb_features, 'nb', explain=True)
        lime_explanation = result[2]
        
        self.assertIsNotNone(lime_explanation)
        self.assertTrue(len(lime_explanation) > 0)
        self.assertTrue(all(isinstance(x, tuple) for x in lime_explanation))

if __name__ == '__main__':
    unittest.main() 