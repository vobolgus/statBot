"""
Basic statistics command handlers.
"""

import os
import logging
import pandas as pd
import plotly.express as px

from src.core.config import DB_FILE
from src.analytics.stats import (
    load_and_prepare_data,
    get_global_top_users,
    prepare_relative_data,
    prepare_cumulative_data,
    build_monthly_stats_text,
)
from src.analytics.ai_summary import generate_funny_summary, get_model_emoji, get_model_name
from src.visualization.plotting import plot_stacked_area

logger = logging.getLogger(__name__)


async def handle_global_stats(event, tokens):
    """Handle global stats command: /stats global N"""
    try:
        top_n = int(tokens[2])
    except (IndexError, ValueError):
        top_n = 7

    df = load_and_prepare_data(DB_FILE)
    if df.empty:
        await event.respond("No messages in the database.")
        return

    top_users, user_counts_sorted = get_global_top_users(df, top_n)
    reply = f"Global top-{top_n} users:\n"
    for idx, user in enumerate(top_users, start=1):
        count = int(
            user_counts_sorted.loc[
                user_counts_sorted['username'] == user,
                'message_count'
            ].values[0]
        )
        reply += f"{idx}. {user} — {count} messages\n"
    await event.respond(reply)


async def handle_hourly_stats(event):
    """Handle hourly stats command: /stats hourly"""
    df = load_and_prepare_data(DB_FILE)
    if df.empty:
        await event.respond("No messages in the database.")
        return

    df['date'] = pd.to_datetime(df['date'])
    hourly_counts = df['date'].dt.hour.value_counts().sort_index()
    total_msgs = len(df)

    reply = "Hourly message statistics:\n"
    hours = list(range(24))
    counts = [hourly_counts.get(h, 0) for h in hours]
    for hour, count in zip(hours, counts):
        perc = (count / total_msgs * 100) if total_msgs else 0
        reply += f"{hour:02d}:00 — {count} messages, {perc:.2f}%\n"

    fig = px.bar(
        x=hours, y=counts,
        labels={'x': 'Hour', 'y': 'Message Count'},
        title="Message distribution by hour for all history"
    )
    fig.update_xaxes(tickmode="linear", tick0=0, dtick=1)
    temp_file = "hourly_stats.png"
    try:
        fig.write_image(temp_file, scale=2)
    except Exception as e:
        logger.error(f"Error saving chart: {e}")

    await event.respond(reply)
    if os.path.exists(temp_file):
        await event.respond(file=temp_file, caption="Hourly activity histogram")
        os.remove(temp_file)  # Cleanup


async def handle_plot_command(event, tokens, client):
    """Handle plot command: /stats plot N FREQ"""
    try:
        top_n = int(tokens[2])
    except (IndexError, ValueError):
        top_n = 7

    freq = tokens[3] if len(tokens) >= 4 else 'W'
    df = load_and_prepare_data(DB_FILE)
    if df.empty:
        await event.respond("No messages in the database.")
        return

    total_msgs = len(df)
    top_users, _ = get_global_top_users(df, top_n)
    pivot_df = prepare_cumulative_data(df, top_users, total_msgs, freq)

    temp_file = plot_stacked_area(
        pivot_df,
        plot_type='cumulative',
        save_as_file=True,
        file_name="cumulative_plot",
        file_format="png"
    )
    caption = (
        f"Cumulative share of messages from top-{top_n} users "
        f"(frequency: {freq}) for all chat history."
    )
    if temp_file and os.path.exists(temp_file):
        await client.send_file(event.chat_id, temp_file, caption=caption)
        os.remove(temp_file)  # Cleanup


async def handle_monthly_stats(event, tokens, client):
    """Handle monthly stats command: /stats YYYY-MM [plot N] or /stats YYYY-MM funny"""
    month_str = tokens[1]
    try:
        period = pd.Period(month_str, freq='M')
    except Exception:
        await event.respond("Invalid month format. Expected YYYY-MM.")
        return

    start_date = period.start_time.tz_localize('UTC')
    end_date = period.end_time.tz_localize('UTC')
    logger.info(f"start_date: {start_date}, end_date: {end_date}")
    df = load_and_prepare_data(DB_FILE, start_date=start_date, end_date=end_date)
    if df.empty:
        await event.respond(f"No messages found for {month_str}.")
        return

    # Generate standard statistics text
    stats_text = build_monthly_stats_text(df, month_str)
    
    if len(tokens) >= 3:
        if tokens[2].lower() == "plot":
            try:
                top_n = int(tokens[3]) if len(tokens) >= 4 else 7
            except ValueError:
                top_n = 7

            top_users, _ = get_global_top_users(df, top_n)
            pivot_df = prepare_relative_data(df, top_users, freq='D')
            temp_file = plot_stacked_area(
                pivot_df,
                plot_type='relative',
                save_as_file=True,
                file_name=f"monthly_plot_{month_str}",
                file_format="png"
            )
            await event.respond(stats_text)
            if temp_file and os.path.exists(temp_file):
                await client.send_file(
                    event.chat_id, temp_file,
                    caption=f"Relative chart for {month_str}"
                )
                os.remove(temp_file)  # Cleanup
        elif tokens[2].lower() == "funny":
            # Get model if specified as the fourth token
            model = tokens[3].lower() if len(tokens) >= 4 else None
            
            # Generate and send funny summary using specified or default model
            await event.respond(stats_text)
            funny_summary = generate_funny_summary(stats_text, model)
            model_emoji = get_model_emoji(model)
            model_name = get_model_name(model)
            await event.respond(f"{model_emoji} Funny Summary ({model_name}):\n\n{funny_summary}")
        else:
            await event.respond(stats_text)
    else:
        await event.respond(stats_text)


async def handle_funny_stats(event, tokens):
    """Handle funny stats command: /stats funny [model]"""
    # Check if a specific model was requested
    model = tokens[2].lower() if len(tokens) >= 3 else None
    
    df = load_and_prepare_data(DB_FILE)
    if df.empty:
        await event.respond("No messages in the database.")
        return
    
    # Generate global stats for all-time data
    total_msgs = len(df)
    user_counts = df['username'].value_counts().reset_index()
    user_counts.columns = ['username', 'message_count']
    user_counts['percentage'] = user_counts['message_count'] / total_msgs * 100
    
    # Get top users
    top_users, _ = get_global_top_users(df, 5)
    
    # Get active days count
    df['date_only'] = df['date'].dt.date
    day_counts = df.groupby('date_only').size()
    active_days = day_counts.count()
    
    # Get busiest hour
    df['hour'] = df['date'].dt.hour
    hour_counts = df['hour'].value_counts().sort_index()
    busiest_hour = hour_counts.idxmax() if not hour_counts.empty else "N/A"
    busiest_hour_count = hour_counts.max() if not hour_counts.empty else 0
    
    # Generate stats text
    stats_text = f"Global Statistics for All Time:\n"
    stats_text += f"Total messages: {total_msgs}\n\n"
    stats_text += "Top 5 Users:\n"
    
    for i, (_, row) in enumerate(user_counts.head(5).iterrows()):
        stats_text += f"{i+1}. {row['username']} — {row['message_count']} messages ({row['percentage']:.2f}%)\n"
    
    stats_text += f"\nActive days: {active_days}\n"
    stats_text += f"Busiest hour: {busiest_hour:02d}:00 ({busiest_hour_count} messages)\n"
    
    # Generate and send funny summary
    await event.respond(stats_text)
    funny_summary = generate_funny_summary(stats_text, model)
    model_emoji = get_model_emoji(model)
    model_name = get_model_name(model)
    await event.respond(f"{model_emoji} Funny Summary ({model_name}):\n\n{funny_summary}")