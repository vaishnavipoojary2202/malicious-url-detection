import re
from urllib.parse import urlparse


def extract_url_features(url):
    features = {}
    url = str(url)

    # Basic URL characteristics
    features['url_length'] = len(url)
    features['num_digits'] = sum(c.isdigit() for c in url)
    features['num_letters'] = sum(c.isalpha() for c in url)

    # Special characters
    features['num_dots'] = url.count('.')
    features['num_hyphens'] = url.count('-')
    features['num_at'] = url.count('@')
    features['num_question_marks'] = url.count('?')
    features['num_equals'] = url.count('=')
    features['num_ampersands'] = url.count('&')
    features['num_slashes'] = url.count('/')
    features['num_percent'] = url.count('%')
    features['num_underscores'] = url.count('_')
    features['num_colons'] = url.count(':')

    # HTTPS
    features['has_https'] = int(url.lower().startswith('https://'))

    # IP address detection
    features['has_ip'] = int(bool(
        re.search(
            r'(?:(?:25[0-5]|2[0-4][0-9]|1?[0-9]{1,2})\.){3}'
            r'(?:25[0-5]|2[0-4][0-9]|1?[0-9]{1,2})',
            url
        )
    ))

    # Hostname and path features
    try:
        parsed = urlparse(url)
        hostname = parsed.hostname or ''
        path = parsed.path or ''

        features['hostname_length'] = len(hostname)
        features['path_length'] = len(path)

        if hostname:
            parts = hostname.split('.')
            features['num_subdomains'] = max(len(parts) - 2, 0)
        else:
            features['num_subdomains'] = 0

    except Exception:
        features['hostname_length'] = 0
        features['path_length'] = 0
        features['num_subdomains'] = 0

    # Suspicious keywords
    suspicious_words = [
        'login',
        'verify',
        'verification',
        'secure',
        'account',
        'update',
        'confirm',
        'password',
        'signin',
        'bank',
        'payment'
    ]

    url_lower = url.lower()

    features['suspicious_keyword_count'] = sum(
        word in url_lower for word in suspicious_words
    )

    # URL shortening services
    shortening_domains = [
        'bit.ly',
        'tinyurl.com',
        'goo.gl',
        't.co',
        'ow.ly',
        'is.gd',
        'buff.ly'
    ]

    features['has_shortening_service'] = int(
        any(domain in url_lower for domain in shortening_domains)
    )

    return features
