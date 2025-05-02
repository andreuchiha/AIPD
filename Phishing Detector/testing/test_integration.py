import unittest
import json
import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from web.app import app
import pandas as pd
from src.inference import load_artifacts

class TestPhishingDetectorIntegration(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True
        
        # Load test data
        self.test_emails = {
            'phishing': """
            URGENT: Your account has been compromised!
            Click here immediately to verify your identity: http://suspicious-link.com
            Your account will be suspended if you don't act now!
            """,
            'legitimate': """
            Dear valued customer,
            Thank you for your recent purchase. Your order #12345 has been confirmed.
            Best regards,
            Customer Service Team
            """
        }

    def test_api_endpoint(self):
        """Test the main API endpoint"""
        response = self.app.post('/analyze',
                               data=json.dumps({
                                   'text': self.test_emails['phishing'],
                                   'model_type': 'nb',
                                   'show_lime': True
                               }),
                               content_type='application/json')
        
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('is_phishing', data)
        self.assertIn('confidence', data)
        self.assertIn('lime_explanation', data)

    def test_model_selection(self):
        """Test different model selections"""
        models = ['nb', 'rf', 'lr', 'gb']
        for model in models:
            response = self.app.post('/analyze',
                                   data=json.dumps({
                                       'text': self.test_emails['phishing'],
                                       'model_type': model,
                                       'show_lime': True
                                   }),
                                   content_type='application/json')
            
            self.assertEqual(response.status_code, 200)
            data = json.loads(response.data)
            self.assertEqual(data['model_type'], model)

    def test_lime_toggle(self):
        """Test LIME explanation toggle"""
        # Test with LIME enabled
        response = self.app.post('/analyze',
                               data=json.dumps({
                                   'text': self.test_emails['phishing'],
                                   'model_type': 'nb',
                                   'show_lime': True
                               }),
                               content_type='application/json')
        
        data = json.loads(response.data)
        self.assertIn('lime_explanation', data)
        self.assertTrue(len(data['lime_explanation']) > 0)
        
        # Test with LIME disabled
        response = self.app.post('/analyze',
                               data=json.dumps({
                                   'text': self.test_emails['phishing'],
                                   'model_type': 'nb',
                                   'show_lime': False
                               }),
                               content_type='application/json')
        
        data = json.loads(response.data)
        self.assertIn('lime_explanation', data)
        self.assertEqual(len(data['lime_explanation']), 0)

    def test_error_handling(self):
        """Test error handling for invalid inputs"""
        # Test empty text
        response = self.app.post('/analyze',
                               data=json.dumps({
                                   'text': '',
                                   'model_type': 'nb',
                                   'show_lime': True
                               }),
                               content_type='application/json')
        
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('is_phishing', data)
        
        # Test invalid model type
        response = self.app.post('/analyze',
                               data=json.dumps({
                                   'text': self.test_emails['phishing'],
                                   'model_type': 'invalid',
                                   'show_lime': True
                               }),
                               content_type='application/json')
        
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('is_phishing', data)

    def test_prediction_consistency(self):
        """Test prediction consistency across different models"""
        results = {}
        for model in ['nb', 'rf', 'lr', 'gb']:
            response = self.app.post('/analyze',
                                   data=json.dumps({
                                       'text': self.test_emails['phishing'],
                                       'model_type': model,
                                       'show_lime': True
                                   }),
                                   content_type='application/json')
            
            data = json.loads(response.data)
            results[model] = data['is_phishing']
        
        # Check if all models agree on phishing detection
        self.assertTrue(all(results.values()))

if __name__ == '__main__':
    unittest.main() 