import json
import calendar
import logging

import pandas as pd
from pandas import Timestamp
import google.generativeai as genai
from openai import OpenAI

from config import USER_MAPPING, GEMINI_API_KEY, DEFAULT_AI_MODEL, XAI_API_KEY

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

def merge_users(df):
    """Merge user names according to mapping."""
    df['username'] = df['username'].replace(USER_MAPPING)
    return df

def parse_date_str(x):
    """
    Parse date string with timezone awareness.
    First tries explicit format, then infers UTC if needed.
    """
    dt = pd.to_datetime(x, format="%Y-%m-%dT%H:%M:%S%z", errors='coerce')
    if pd.isna(dt):
        dt = pd.to_datetime(x, infer_datetime_format=True, utc=True, errors='coerce')
    return dt

def load_and_prepare_data(json_file, start_date=None, end_date=None):
    """
    Load message database from JSON file and prepare DataFrame.
    Selects only messages, parses dates, applies user mapping,
    and filters by optional date range.
    """
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    messages = data.get("messages", [])
    df = pd.DataFrame(messages)
    df_messages = df[df.get('type') == 'message'].copy()
    if df_messages.empty:
        return df_messages
    df_messages['date'] = df_messages['date'].apply(parse_date_str)
    df_messages['username'] = df_messages['from'].fillna('Unknown')
    df_messages = df_messages[df_messages['username'] != 'Unknown']
    df_messages = merge_users(df_messages)
    if start_date is not None:
        df_messages = df_messages[df_messages['date'] >= pd.to_datetime(start_date, utc=True)]
    if end_date is not None:
        if not isinstance(end_date, Timestamp):
            end_dt = pd.to_datetime(end_date, utc=True)
        else:
            end_dt = end_date if end_date.tzinfo is not None else end_date.tz_localize('UTC')
        df_messages = df_messages[df_messages['date'] <= end_dt]
    return df_messages

def get_global_top_users(df, top_n=7):
    """Determine global top-N users by message count."""
    user_counts = df['username'].value_counts().reset_index()
    user_counts.columns = ['username', 'message_count']
    top_users = user_counts.head(top_n)['username'].tolist()
    return top_users, user_counts

def prepare_relative_data(df, top_users, freq='W'):
    """Prepare data for relative stacked-area chart."""
    df_copy = df.copy()
    df_copy['interval'] = df_copy['date'].dt.to_period(freq).apply(lambda r: r.start_time)
    show_other = len(top_users) < df_copy['username'].nunique()
    if show_other:
        df_copy['category'] = df_copy['username'].apply(lambda x: x if x in top_users else 'Другие')
    else:
        df_copy['category'] = df_copy['username']
    grouped = df_copy.groupby(['interval', 'category']).size().reset_index(name='count')
    totals = grouped.groupby('interval')['count'].sum().reset_index(name='total_count')
    merged = pd.merge(grouped, totals, on='interval')
    merged['relative_share'] = merged['count'] / merged['total_count']
    pivot = merged.pivot(index='interval', columns='category', values='relative_share').fillna(0)
    for u in top_users + (['Другие'] if show_other else []):
        if u not in pivot.columns:
            pivot[u] = 0
    pivot = pivot[top_users + (['Другие'] if show_other else [])]
    pivot['sum'] = pivot.sum(axis=1)
    if not (pivot['sum'].round(4) == 1).all():
        logger.warning("Sum of shares in some intervals is not 1.")
    return pivot.drop(columns=['sum'])

def prepare_cumulative_data(df, top_users, total_messages, freq='W'):
    """Prepare data for cumulative stacked-area chart."""
    df_copy = df.copy()
    df_copy['interval'] = df_copy['date'].dt.to_period(freq).apply(lambda r: r.start_time)
    show_other = len(top_users) < df_copy['username'].nunique()
    if show_other:
        df_copy['category'] = df_copy['username'].apply(lambda x: x if x in top_users else 'Другие')
    else:
        df_copy['category'] = df_copy['username']
    grouped = df_copy.groupby(['interval', 'category']).size().reset_index(name='count')
    grouped = grouped.sort_values('interval')
    pivot = grouped.pivot(index='interval', columns='category', values='count').fillna(0)
    pivot = pivot[top_users + (['Другие'] if show_other else [])]
    pivot = pivot.cumsum() / total_messages
    pivot['sum'] = pivot.sum(axis=1)
    if not (pivot['sum'] <= 1).all():
        logger.warning("Sum of shares in some intervals exceeds 1.")
    return pivot.drop(columns=['sum'])

def build_monthly_stats_text(df, month_str):
    """Generate text summary of statistics for specified month."""
    total_msgs = len(df)
    if total_msgs == 0:
        return f"No messages found for {month_str}."
    user_counts = df['username'].value_counts().reset_index()
    user_counts.columns = ['username', 'message_count']
    user_counts['percentage'] = user_counts['message_count'] / total_msgs * 100
    top_user = user_counts.iloc[0]
    df['date_only'] = df['date'].dt.date
    day_counts = df.groupby('date_only').size()
    active_days = day_counts.count()
    avg_msgs = total_msgs / active_days if active_days else 0
    median_msgs = day_counts.median()
    std_msgs = day_counts.std()
    busiest_day = day_counts.idxmax() if not day_counts.empty else "N/A"
    busiest_day_count = day_counts.max() if not day_counts.empty else 0
    least_day = day_counts.idxmin() if not day_counts.empty else "N/A"
    least_day_count = day_counts.min() if not day_counts.empty else 0
    year, month = map(int, month_str.split('-'))
    total_days = calendar.monthrange(year, month)[1]
    active_pct = active_days / total_days * 100 if total_days else 0
    df['hour'] = df['date'].dt.hour
    hour_counts = df['hour'].value_counts().sort_index()
    busiest_hour = hour_counts.idxmax() if not hour_counts.empty else "N/A"
    busiest_hour_count = hour_counts.max() if not hour_counts.empty else 0
    text = f"Statistics for {month_str}:\n" + f"Total messages: {total_msgs}\n\n" + "Messages by user:\n"
    for i, row in user_counts.iterrows():
        if i >= 5: break
        text += f"{i + 1}. {row['username']} — {row['message_count']} messages ({row['percentage']:.2f}%)\n"
    text += (
        f"\nMost active user: {top_user['username']} ({top_user['message_count']} messages)\n"
        f"\nDays in month: {total_days}, active days: {active_days} ({active_pct:.2f}%)\n"
        f"Average messages per active day: {avg_msgs:.2f}\n"
        f"Median messages per active day: {median_msgs:.2f}\n"
        f"Standard deviation: {std_msgs:.2f}\n"
        f"Busiest day: {busiest_day} ({busiest_day_count} messages)\n"
        f"Least active day: {least_day} ({least_day_count} messages)\n"
        f"\nBusiest hour: {busiest_hour:02d}:00 ({busiest_hour_count} messages)\n"
    )
    return text

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