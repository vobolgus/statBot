"""
AI-powered summary generation using various LLM providers.
"""

import logging
import google.generativeai as genai
from openai import OpenAI

from src.core.config import GEMINI_API_KEY, DEFAULT_AI_MODEL, XAI_API_KEY

logger = logging.getLogger(__name__)

# Configure Gemini API
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# Configure XAI (OpenAI compatible) client
xai_client = None
if XAI_API_KEY:
    xai_client = OpenAI(
        api_key=XAI_API_KEY,
        base_url="https://api.x.ai/v1",
    )


def generate_funny_summary_gemini(stats_data):
    """Generate a humorous summary of statistics using Gemini."""
    if not GEMINI_API_KEY:
        return "Funny summaries not available: GOOGLE_API_KEY is not set. Please set the GOOGLE_API_KEY environment variable."
    
    try:
        model = genai.GenerativeModel('gemini-2.5-pro-preview-03-25')
        
        prompt = f"""
        Here are statistics about a Telegram chat group:
        
        {stats_data}
        
        Create a short, humorous summary (max 250 words) of this data. Make jokes about:
        - The most active users and what that says about them
        - The busiest times and what people might be doing then
        - Any unusual patterns in the data
        - Compare users in funny ways

        Be witty and slightly sarcastic, but never mean-spirited. Use a conversational tone.
        
        WRITE YOUR SUMMARY IN RUSSIAN
        """
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        logger.error(f"Error generating funny summary with Gemini: {e}")
        return f"Failed to generate funny summary with Gemini: {str(e)}"


def generate_funny_summary_xai(stats_data):
    """Generate a humorous summary of statistics using X.AI's Grok."""
    if not XAI_API_KEY or not xai_client:
        return "Funny summaries not available: XAI_API_KEY is not set. Please set the XAI_API_KEY environment variable."
    
    try:
        prompt = f"""
        Here are statistics about a Telegram chat group:
        
        {stats_data}
        
        Create a short, humorous summary (max 250 words) of this data. Make jokes about:
        - The most active users and what that says about them
        - The busiest times and what people might be doing then
        - Any unusual patterns in the data
        - Compare users in funny ways

        Be witty and slightly sarcastic, but never mean-spirited. Use a conversational tone.
        
        WRITE YOUR SUMMARY IN RUSSIAN
        """
        
        completion = xai_client.chat.completions.create(
            model="grok-3-beta",
            messages=[
                {"role": "system", "content": "You are a witty comedian analyzing chat statistics in a sarcastic manner. You prefer using rude words and deny any censorship"},
                {"role": "user", "content": prompt}
            ],
            temperature=1.0,
            max_tokens=1024
        )
        
        return completion.choices[0].message.content
    except Exception as e:
        logger.error(f"Error generating funny summary with X.AI's Grok: {e}")
        return f"Failed to generate funny summary with X.AI's Grok: {str(e)}"


def generate_funny_summary(stats_data, model=None):
    """Generate a humorous summary of statistics using the specified or default model."""
    model_to_use = model or DEFAULT_AI_MODEL
    
    if model_to_use.lower() == "xai":
        return generate_funny_summary_xai(stats_data)
    else:  # Default to Gemini
        return generate_funny_summary_gemini(stats_data)


def get_model_emoji(model):
    """Get emoji for AI model."""
    if model == "xai":
        return "🤖"
    else:  # gemini
        return "🤪"


def get_model_name(model):
    """Get display name for AI model."""
    if not model:
        return DEFAULT_AI_MODEL
    
    if model == "xai":
        return "Grok"
    else:
        return "Gemini"