"""
Word cloud generation and word frequency analysis.
"""

import logging
from collections import Counter
from src.utils.text_processing import load_russian_stopwords, clean_text_for_analysis, extract_clean_texts

logger = logging.getLogger(__name__)


def generate_wordcloud(df, user=None):
    """
    Generate a word cloud from message texts.
    
    Args:
        df: DataFrame with messages containing 'text' and 'username' columns
        user: Optional username to filter messages (default: None for all users)
        
    Returns:
        str: Path to saved word cloud image or None if error
    """
    try:
        from wordcloud import WordCloud
        
        # Extract clean texts
        texts = extract_clean_texts(df, user)
        
        if not texts:
            return None
        
        # Combine all texts
        all_text = ' '.join(texts)
        
        # Clean text
        all_text = clean_text_for_analysis(all_text)
        
        if len(all_text.strip()) < 10:
            return None
        
        # Load comprehensive Russian stop words + technical terms
        stop_words = load_russian_stopwords()
        
        # Create WordCloud
        wordcloud = WordCloud(
            width=1200,
            height=600,
            background_color='white',
            max_words=100,
            stopwords=stop_words,
            font_path=None,  # Use default font
            relative_scaling=0.5,
            min_font_size=10,
            max_font_size=80,
            prefer_horizontal=0.7,
            colormap='viridis'
        ).generate(all_text)
        
        # Save to file
        if user:
            filename = f"wordcloud_{user.replace(' ', '_')}.png"
        else:
            filename = "wordcloud_all.png"
        
        wordcloud.to_file(filename)
        return filename
        
    except Exception as e:
        logger.error(f"Error generating word cloud: {e}")
        return None


def get_word_frequency_stats(df, user=None, top_n=20):
    """
    Get word frequency statistics from messages.
    
    Args:
        df: DataFrame with messages containing 'text' and 'username' columns
        user: Optional username to filter messages (default: None for all users)
        top_n: Number of top words to return (default: 20)
        
    Returns:
        str: Formatted text with word frequency statistics
    """
    try:
        # Extract clean texts
        texts = extract_clean_texts(df, user)
        
        if not texts:
            if user:
                return f"Пользователь '{user}' не найден или у него нет сообщений."
            else:
                return "Недостаточно текста для анализа частоты слов."
        
        all_text = ' '.join(texts)
        
        # Clean and tokenize
        all_text = clean_text_for_analysis(all_text)
        
        # Split into words and filter
        words = [word.lower() for word in all_text.split() if len(word) > 2]
        
        # Load comprehensive Russian stop words + technical terms
        stop_words = load_russian_stopwords()
        
        filtered_words = [word for word in words if word not in stop_words]
        
        if not filtered_words:
            return "Недостаточно текста для анализа частоты слов."
        
        # Count words
        word_counts = Counter(filtered_words)
        top_words = word_counts.most_common(top_n)
        
        # Format results
        if user:
            result = f"📊 Топ-{top_n} слов пользователя {user}:\n\n"
        else:
            result = f"📊 Топ-{top_n} слов в чате:\n\n"
        
        total_words = len(filtered_words)
        for i, (word, count) in enumerate(top_words, 1):
            percentage = (count / total_words) * 100
            result += f"{i:2d}. {word} — {count} раз ({percentage:.1f}%)\n"
        
        result += f"\nВсего уникальных слов: {len(word_counts)}\n"
        result += f"Всего слов (без стоп-слов): {total_words}"
        
        return result
        
    except Exception as e:
        logger.error(f"Error generating word frequency stats: {e}")
        return f"Ошибка при анализе частоты слов: {str(e)}"