"""
URL Analysis Module

This module implements comprehensive URL analysis for phishing detection.
It includes checks for suspicious patterns, SSL certificates, domain age,
and various other security indicators in URLs.
"""

import re
import tld
from urllib.parse import urlparse
import whois
from datetime import datetime
import requests
from typing import Dict, List, Tuple
import socket
import ssl
import OpenSSL
import concurrent.futures

# List of known URL shortening services
URL_SHORTENERS = {
    'bit.ly', 'tinyurl.com', 'goo.gl', 't.co', 'is.gd', 'cli.gs', 'ow.ly', 'yfrog.com',
    'migre.me', 'ff.im', 'tiny.cc', 'url4.eu', 'tr.im', 'twit.ac', 'su.pr', 'twurl.nl',
    'snipurl.com', 'short.to', 'BudURL.com', 'ping.fm', 'post.ly', 'Just.as', 'bkite.com',
    'snipr.com', 'fic.kr', 'loopt.us', 'doiop.com', 'twitthis.com', 'htxt.it', 'alturl.com',
    'redir.ec', 'tiny.pl', 'url.za', 'zz.gd', 'twit.tv', 'ht.ly', 'htl.li', 'ht.ly', 'htl.li',
    'ht.ly', 'htl.li', 'ht.ly', 'htl.li', 'ht.ly', 'htl.li', 'ht.ly', 'htl.li', 'ht.ly', 'htl.li'
}

# List of suspicious top-level domains
SUSPICIOUS_TLDS = {
    'xyz', 'top', 'loan', 'work', 'click', 'bid', 'win', 'review', 'download', 'stream',
    'gq', 'tk', 'ml', 'ga', 'cf', 'pw', 'cc', 'info', 'biz', 'online', 'site', 'website',
    'space', 'tech', 'science', 'download', 'click', 'link', 'web', 'online', 'site'
}

# Regular expressions for detecting suspicious URL patterns
SUSPICIOUS_PATTERNS = [
    r'login[^a-zA-Z]',
    r'secure[^a-zA-Z]',
    r'account[^a-zA-Z]',
    r'verify[^a-zA-Z]',
    r'confirm[^a-zA-Z]',
    r'bank[^a-zA-Z]',
    r'paypal[^a-zA-Z]',
    r'ebay[^a-zA-Z]',
    r'amazon[^a-zA-Z]',
    r'apple[^a-zA-Z]',
    r'google[^a-zA-Z]',
    r'facebook[^a-zA-Z]',
    r'twitter[^a-zA-Z]',
    r'linkedin[^a-zA-Z]',
    r'instagram[^a-zA-Z]',
    r'netflix[^a-zA-Z]',
    r'spotify[^a-zA-Z]',
    r'yahoo[^a-zA-Z]',
    r'hotmail[^a-zA-Z]',
    r'outlook[^a-zA-Z]',
    r'gmail[^a-zA-Z]',
    r'icloud[^a-zA-Z]',
    r'dropbox[^a-zA-Z]',
    r'onedrive[^a-zA-Z]',
    r'box[^a-zA-Z]',
    r'drive[^a-zA-Z]',
    r'docs[^a-zA-Z]',
    r'sheets[^a-zA-Z]',
    r'slides[^a-zA-Z]',
    r'forms[^a-zA-Z]',
    r'sites[^a-zA-Z]',
    r'blog[^a-zA-Z]',
    r'wordpress[^a-zA-Z]',
    r'blogspot[^a-zA-Z]',
    r'medium[^a-zA-Z]',
    r'github[^a-zA-Z]',
    r'gitlab[^a-zA-Z]',
    r'bitbucket[^a-zA-Z]',
    r'stackoverflow[^a-zA-Z]',
    r'reddit[^a-zA-Z]',
    r'quora[^a-zA-Z]',
    r'medium[^a-zA-Z]',
    r'dev[^a-zA-Z]',
    r'io[^a-zA-Z]',
    r'co[^a-zA-Z]',
    r'me[^a-zA-Z]',
    r'app[^a-zA-Z]',
    r'dev[^a-zA-Z]',
    r'io[^a-zA-Z]',
    r'co[^a-zA-Z]',
    r'me[^a-zA-Z]',
    r'app[^a-zA-Z]',
]

def extract_urls(text: str) -> List[str]:
    """
    Extract all URLs from a text string using regex pattern matching.
    """
    url_pattern = r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+|www\.(?:[-\w.]|(?:%[\da-fA-F]{2}))+'
    return re.findall(url_pattern, text)

def analyze_url_structure(url: str) -> Dict[str, float]:
    """
    Analyze URL structure for suspicious patterns and characteristics.
    Returns a dictionary of risk scores for various URL attributes.
    """
    try:
        parsed = urlparse(url)
        domain = parsed.netloc
        path = parsed.path
        query = parsed.query
        
        scores = {
            'suspicious_patterns': 0.0,
            'suspicious_tld': 0.0,
            'is_shortener': 0.0,
            'has_ip': 0.0,
            'long_path': 0.0,
            'many_subdomains': 0.0,
            'suspicious_query': 0.0
        }
        
        for pattern in SUSPICIOUS_PATTERNS:
            if re.search(pattern, url.lower()):
                scores['suspicious_patterns'] += 1.0
        
        try:
            tld_info = tld.get_tld(url, as_object=True)
            if tld_info.tld in SUSPICIOUS_TLDS:
                scores['suspicious_tld'] = 1.0
        except:
            pass
        
        if any(shortener in domain for shortener in URL_SHORTENERS):
            scores['is_shortener'] = 1.0
        
        try:
            socket.inet_aton(domain)
            scores['has_ip'] = 1.0
        except:
            pass
        
        if len(path) > 50:
            scores['long_path'] = 1.0
        
        if domain.count('.') > 2:
            scores['many_subdomains'] = 1.0
        
        if len(query) > 50:
            scores['suspicious_query'] = 1.0
        
        return scores
    except:
        return {
            'suspicious_patterns': 0.0,
            'suspicious_tld': 0.0,
            'is_shortener': 0.0,
            'has_ip': 0.0,
            'long_path': 0.0,
            'many_subdomains': 0.0,
            'suspicious_query': 0.0
        }

def check_ssl(url: str) -> Dict[str, float]:
    """
    Check SSL certificate information and validity.
    Returns a dictionary of SSL-related risk scores.
    """
    scores = {
        'has_ssl': 0.0,
        'valid_ssl': 0.0,
        'recent_ssl': 0.0
    }
    
    try:
        parsed = urlparse(url)
        if parsed.scheme != 'https':
            return scores
        
        hostname = parsed.netloc
        context = ssl.create_default_context()
        with socket.create_connection((hostname, 443)) as sock:
            with context.wrap_socket(sock, server_hostname=hostname) as ssock:
                cert = ssock.getpeercert()
                scores['has_ssl'] = 1.0
                
                if cert:
                    scores['valid_ssl'] = 1.0
                    
                    not_before = datetime.strptime(cert['notBefore'], '%b %d %H:%M:%S %Y %Z')
                    if (datetime.now() - not_before).days < 365:
                        scores['recent_ssl'] = 1.0
    except:
        pass
    
    return scores

def check_domain_age(url: str) -> Dict[str, float]:
    """
    Check domain age and registration information.
    Returns a dictionary of domain age-related risk scores.
    """
    scores = {
        'has_whois': 0.0,
        'old_domain': 0.0,
        'recent_registration': 0.0
    }
    
    try:
        domain = tld.get_tld(url, as_object=True).fld
        w = whois.whois(domain)
        
        if w.creation_date:
            scores['has_whois'] = 1.0
            
            if isinstance(w.creation_date, list):
                creation_date = w.creation_date[0]
            else:
                creation_date = w.creation_date
                
            age_days = (datetime.now() - creation_date).days
            if age_days > 365:
                scores['old_domain'] = 1.0
            if age_days < 30:
                scores['recent_registration'] = 1.0
    except:
        pass
    
    return scores

def analyze_urls_batch(texts: List[str]) -> List[Dict[str, float]]:
    """
    Analyze URLs in a batch of texts.
    Returns a list of dictionaries containing URL analysis results.
    """
    results = []
    for text in texts:
        urls = extract_urls(text)
        if not urls:
            results.append({
                'url_count': 0.0,
                'url_ratio': 0.0,
                'suspicious_patterns': 0.0,
                'suspicious_tld': 0.0,
                'is_shortener': 0.0,
                'has_ip': 0.0,
                'long_path': 0.0,
                'many_subdomains': 0.0,
                'suspicious_query': 0.0,
                'has_ssl': 0.0,
                'valid_ssl': 0.0,
                'recent_ssl': 0.0,
                'has_whois': 0.0,
                'old_domain': 0.0,
                'recent_registration': 0.0
            })
            continue
        
        # Analyze each URL and combine results
        url_scores = {
            'url_count': float(len(urls)),
            'url_ratio': float(len(urls)) / len(text.split()),
            'suspicious_patterns': 0.0,
            'suspicious_tld': 0.0,
            'is_shortener': 0.0,
            'has_ip': 0.0,
            'long_path': 0.0,
            'many_subdomains': 0.0,
            'suspicious_query': 0.0,
            'has_ssl': 0.0,
            'valid_ssl': 0.0,
            'recent_ssl': 0.0,
            'has_whois': 0.0,
            'old_domain': 0.0,
            'recent_registration': 0.0
        }
        
        for url in urls:
            structure = analyze_url_structure(url)
            ssl_info = check_ssl(url)
            domain_info = check_domain_age(url)
            
            for key in structure:
                url_scores[key] = max(url_scores[key], structure[key])
            for key in ssl_info:
                url_scores[key] = max(url_scores[key], ssl_info[key])
            for key in domain_info:
                url_scores[key] = max(url_scores[key], domain_info[key])
        
        results.append(url_scores)
    
    return results

if __name__ == "__main__":
    # Test the URL analysis
    test_texts = [
        "Check out this link: https://www.google.com",
        "URGENT: Your account will be suspended. Click here: http://suspicious-site.xyz/login",
        "Visit our secure site: https://legitimate-site.com"
    ]
    
    results = analyze_urls_batch(test_texts)
    for text, result in zip(test_texts, results):
        print(f"\nText: {text}")
        print("URL Analysis Results:")
        for key, value in result.items():
            print(f"{key}: {value}") 