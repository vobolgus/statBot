"""
Text processing utilities for NLP analysis.
"""

import re
import logging
from typing import Set

logger = logging.getLogger(__name__)


def load_russian_stopwords() -> Set[str]:
    """Load Russian stop words from file and combine with technical terms."""
    stop_words = set()
    
    # Try to load from stopwords-ru.txt
    try:
        with open('stopwords-ru.txt', 'r', encoding='utf-8') as f:
            for line in f:
                word = line.strip().lower()
                if word:
                    stop_words.add(word)
    except FileNotFoundError:
        logger.warning("stopwords-ru.txt not found, using default stop words")
    
    # Add common English stop words
    english_stopwords = {
        'if', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did',
        'will', 'would', 'could', 'should', 'may', 'might', 'can', 'must', 'a', 'an', 'this', 'that', 'these', 'those'
    }
    stop_words.update(english_stopwords)
    
    # Add technical terms that shouldn't appear in word clouds
    technical_terms = {
        'text', 'type', 'bold', 'text_link', 'href', 'url', 'code', 'pre', 'message', 'entities', 'entity_type',
        'italic', 'underline', 'strikethrough', 'spoiler', 'blockquote', 'mention', 'hashtag', 'bot_command',
        'email', 'phone_number', 'cashtag', 'media', 'photo', 'video', 'document', 'audio', 'voice', 'sticker',
        'animation', 'contact', 'location', 'venue', 'poll', 'dice', 'game', 'invoice', 'successful_payment'
    }
    stop_words.update(technical_terms)
    
    return stop_words


def clean_text_for_analysis(text: str) -> str:
    """Clean text for analysis by removing URLs, mentions, hashtags, and special characters."""
    # Clean text: remove URLs, mentions, hashtags, and special characters
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'[^\w\s\u0400-\u04FF]', ' ', text)  # Keep Cyrillic and Latin
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def extract_clean_texts(df, user=None):
    """Extract and clean texts from DataFrame for analysis."""
    # Filter by user if specified
    if user:
        df_filtered = df[df['username'] == user].copy()
        if df_filtered.empty:
            return []
    else:
        df_filtered = df.copy()
    
    # Extract and clean text - only process actual message text
    texts = []
    for text in df_filtered['text'].fillna(''):
        # Skip empty or non-string values
        if not text or not isinstance(text, str):
            continue
        # Skip JSON-like content (starts with { or [)
        text_stripped = text.strip()
        if text_stripped.startswith('{') or text_stripped.startswith('['):
            continue
        # Skip very short messages (likely metadata)
        if len(text_stripped) < 3:
            continue
        texts.append(text_stripped)
    
    return texts