"""
Urgency Scoring Module

This module implements comprehensive urgency detection for phishing emails.
It analyzes text for various urgency indicators including time pressure,
threats, scarcity, and other manipulative tactics commonly used in phishing.
"""

import re
from typing import Dict, List, Tuple
from nltk.util import ngrams

def simple_tokenize(text: str) -> List[str]:
    """
    Tokenize text by splitting on whitespace and removing punctuation.
    Returns a list of cleaned tokens.
    """
    words = text.lower().split()
    words = [re.sub(r'[^\w\s]', '', word) for word in words]
    return [word for word in words if word]

def get_ngrams(text: str, n: int) -> List[str]:
    """
    Generate n-grams from tokenized text.
    Returns a list of n-gram strings.
    """
    tokens = simple_tokenize(text)
    return [' '.join(gram) for gram in ngrams(tokens, n)]

# Comprehensive list of urgency indicators by category
URGENCY_INDICATORS = {
    'time': [
        'urgent', 'immediate', 'today', 'now', 'asap', 'right away',
        'within 24 hours', 'within 48 hours', 'expires', 'deadline',
        'time-sensitive', 'time limit', 'time frame', 'time period',
        '24 hours', '48 hours', '72 hours', '1 day', '2 days', '3 days',
        'one day', 'two days', 'three days', 'one week', 'two weeks',
        '1 week', '2 weeks', '1 month', '2 months', '1 year', '2 years'
    ],
    'pressure': [
        'act now', 'limited time', 'last chance', 'don\'t miss out',
        'exclusive offer', 'special deal', 'limited offer', 'one-time offer',
        'first come first serve', 'while supplies last', 'limited availability',
        'limited quantity', 'limited stock', 'limited seats', 'limited spots'
    ],
    'threat': [
        'account will be closed', 'action required', 'account suspended',
        'account blocked', 'account locked', 'account terminated',
        'account deactivated', 'account disabled', 'account restricted',
        'account frozen', 'account cancelled', 'account deleted',
        'security breach', 'security threat', 'security risk',
        'security issue', 'security problem', 'security concern'
    ],
    'scarcity': [
        'limited offer', 'exclusive deal', 'special offer',
        'special deal', 'special price', 'special rate',
        'special discount', 'special savings', 'special promotion',
        'special opportunity', 'special chance', 'special occasion',
        'special event', 'special sale', 'special price'
    ],
    'urgency': [
        'urgent', 'immediate', 'asap', 'right away', 'right now',
        'without delay', 'without hesitation', 'without waiting',
        'without further delay', 'without further ado', 'without further notice',
        'without further warning', 'without further action', 'without further steps'
    ]
}

def calculate_urgency_score(text: str) -> float:
    """
    Calculate a normalized urgency score for the given text.
    Analyzes presence of urgency indicators across multiple categories.
    Returns a score between 0 and 1.
    """
    text = text.lower()
    total_indicators = sum(len(indicators) for indicators in URGENCY_INDICATORS.values())
    found_indicators = 0
    
    all_ngrams = []
    for n in range(1, 4):
        all_ngrams.extend(get_ngrams(text, n))
    
    for category, indicators in URGENCY_INDICATORS.items():
        for indicator in indicators:
            indicator_words = indicator.split()
            if len(indicator_words) == 1:
                if any(indicator in ngram for ngram in all_ngrams):
                    found_indicators += 1
            else:
                if indicator in all_ngrams:
                    found_indicators += 1
    
    return found_indicators / total_indicators if total_indicators > 0 else 0

def extract_urgency_features(text: str) -> Dict[str, float]:
    """
    Extract detailed urgency features from the text.
    Calculates various metrics including word and sentence-level urgency ratios.
    Returns a dictionary of urgency-related metrics.
    """
    text = text.lower()
    words = simple_tokenize(text)
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    all_ngrams = []
    for n in range(1, 4):
        all_ngrams.extend(get_ngrams(text, n))
    
    urgency_words = []
    for category, indicators in URGENCY_INDICATORS.items():
        for indicator in indicators:
            indicator_words = indicator.split()
            if len(indicator_words) == 1:
                if any(indicator in ngram for ngram in all_ngrams):
                    urgency_words.append(indicator)
            else:
                if indicator in all_ngrams:
                    urgency_words.append(indicator)
    
    total_words = len(words)
    total_sentences = len(sentences)
    urgency_word_count = len(urgency_words)
    
    urgency_sentences = 0
    for sentence in sentences:
        sentence_ngrams = []
        for n in range(1, 4):
            sentence_ngrams.extend(get_ngrams(sentence, n))
        
        if any(indicator in sentence_ngrams for indicators in URGENCY_INDICATORS.values() for indicator in indicators):
            urgency_sentences += 1
    
    return {
        'urgency_score': calculate_urgency_score(text),
        'urgency_word_ratio': urgency_word_count / total_words if total_words > 0 else 0,
        'urgency_sentence_ratio': urgency_sentences / total_sentences if total_sentences > 0 else 0,
        'urgency_word_count': urgency_word_count,
        'urgency_sentence_count': urgency_sentences
    }

def get_urgency_explanation(text: str) -> List[Tuple[str, float]]:
    """
    Get a list of urgency indicators found in the text with their weights.
    Weights are assigned based on the category of the urgency indicator.
    Returns a list of (indicator, weight) tuples.
    """
    text = text.lower()
    explanation = []
    
    all_ngrams = []
    for n in range(1, 4):
        all_ngrams.extend(get_ngrams(text, n))
    
    category_weights = {
        'time': 1.5,     
        'pressure': 2.5,  
        'threat': 2.5,    
        'scarcity': 1.5, 
        'urgency': 2.0    
    }
    
    for category, indicators in URGENCY_INDICATORS.items():
        for indicator in indicators:
            indicator_words = indicator.split()
            if len(indicator_words) == 1:
                if any(indicator in ngram for ngram in all_ngrams):
                    weight = category_weights[category]
                    explanation.append((indicator, weight))
            else:
                if indicator in all_ngrams:
                    weight = category_weights[category]
                    explanation.append((indicator, weight))
    
    return explanation

if __name__ == "__main__":
    test_text = """
    URGENT: Your account will be suspended within 24 hours if you don't act now!
    This is a limited time offer, don't miss out on this exclusive deal.
    """
    
    print("Urgency Score:", calculate_urgency_score(test_text))
    print("\nUrgency Features:")
    for key, value in extract_urgency_features(test_text).items():
        print(f"{key}: {value}")
    print("\nUrgency Explanation:")
    for indicator, weight in get_urgency_explanation(test_text):
        print(f"{indicator}: {weight}") 