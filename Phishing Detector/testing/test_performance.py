import unittest
import time
import pandas as pd
import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.inference import load_artifacts, predict_email
from web.app import app
import json

class TestPhishingDetectorPerformance(unittest.TestCase):
    def setUp(self):
        # Load models
        self.nb_vec, self.nb_clf, self.nb_selector, self.nb_features = load_artifacts('nb')
        self.rf_vec, self.rf_clf, _, _ = load_artifacts('rf')
        self.lr_vec, self.lr_clf, self.lr_selector, self.lr_features = load_artifacts('lr')
        self.gb_vec, self.gb_clf, self.gb_selector, self.gb_features = load_artifacts('gb')
        
        # Test client
        self.app = app.test_client()
        self.app.testing = True
        
        # Load test data
        self.test_emails = pd.read_csv("./data/Kaggle.csv")
        self.test_emails = self.test_emails.dropna(subset=['body', 'label'])
        self.test_emails = self.test_emails.sample(n=100, random_state=42)  # Use 100 samples for testing

    def test_model_load_time(self):
        """Test model loading performance"""
        start_time = time.time()
        load_artifacts('nb')
        nb_load_time = time.time() - start_time
        
        start_time = time.time()
        load_artifacts('rf')
        rf_load_time = time.time() - start_time
        
        start_time = time.time()
        load_artifacts('lr')
        lr_load_time = time.time() - start_time
        
        start_time = time.time()
        load_artifacts('gb')
        gb_load_time = time.time() - start_time
        
        # Assert that model loading takes less than 5 seconds
        self.assertLess(nb_load_time, 5)
        self.assertLess(rf_load_time, 5)
        self.assertLess(lr_load_time, 5)
        self.assertLess(gb_load_time, 5)

    def test_prediction_time(self):
        """Test prediction performance"""
        models = {
            'nb': (self.nb_vec, self.nb_clf, self.nb_selector, self.nb_features),
            'rf': (self.rf_vec, self.rf_clf, None, None),
            'lr': (self.lr_vec, self.lr_clf, self.lr_selector, self.lr_features),
            'gb': (self.gb_vec, self.gb_clf, self.gb_selector, self.gb_features)
        }
        
        results = {}
        for model_name, (vec, clf, selector, features) in models.items():
            total_time = 0
            for _, row in self.test_emails.iterrows():
                start_time = time.time()
                predict_email(
                    row['body'],
                    vec,
                    clf,
                    selector=selector,
                    selected_features=features,
                    model_type=model_name
                )
                total_time += time.time() - start_time
            
            avg_time = total_time / len(self.test_emails)
            results[model_name] = avg_time
            
            # Assert that average prediction time is less than 1 second
            self.assertLess(avg_time, 1.0)
        
        print("\nAverage prediction times:")
        for model, time_taken in results.items():
            print(f"{model}: {time_taken:.3f} seconds")

    def test_api_response_time(self):
        """Test API response time"""
        # Use only 10 samples for API testing
        test_samples = self.test_emails.sample(n=10, random_state=42)
        total_time = 0
        for _, row in test_samples.iterrows():
            start_time = time.time()
            self.app.post('/analyze',
                         data=json.dumps({
                             'text': row['body'],
                             'model_type': 'nb',
                             'show_lime': False  # Disable LIME for performance testing
                         }),
                         content_type='application/json')
            total_time += time.time() - start_time
        
        avg_time = total_time / len(test_samples)
        
        # Assert that average API response time is less than 2 seconds
        self.assertLess(avg_time, 2.0)
        print(f"\nAverage API response time: {avg_time:.3f} seconds")

    def test_memory_usage(self):
        """Test memory usage during predictions"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Run predictions
        for _, row in self.test_emails.iterrows():
            predict_email(
                row['body'],
                self.nb_vec,
                self.nb_clf,
                selector=self.nb_selector,
                selected_features=self.nb_features,
                model_type='nb'
            )
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Assert that memory increase is less than 500MB
        self.assertLess(memory_increase, 500)
        print(f"\nMemory usage increase: {memory_increase:.2f} MB")

if __name__ == '__main__':
    unittest.main() 