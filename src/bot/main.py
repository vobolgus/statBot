import asyncio
import os
import re
import logging

import tools.fix_pillow  # Fix for Pillow 10+ compatibility
import pandas as pd
import plotly.express as px
from telethon import TelegramClient, events

from src.core.config import API_ID, API_HASH, BOT_TOKEN, CHAT_ID, DB_FILE, DEFAULT_AI_MODEL
from src.core.database import load_db, save_db, message_exists, telethon_message_to_dict
from src.analytics.stats import (
    load_and_prepare_data,
    get_global_top_users,
    prepare_relative_data,
    prepare_cumulative_data,
    build_monthly_stats_text,
    calculate_dominance_index,
    analyze_user_correlation,
)
from src.analytics.ai_summary import generate_funny_summary, get_model_emoji, get_model_name
from src.analytics.clustering import (
    cluster_users_by_activity,
    format_cluster_analysis_text,
    create_cluster_visualization,
)
from src.visualization.plotting import plot_stacked_area
from src.visualization.wordcloud import generate_wordcloud, get_word_frequency_stats
from src.analytics.social_graph import (
    build_interaction_graph,
    calculate_graph_metrics,
    create_interactive_graph,
    generate_graph_statistics,
)

# Import handlers
from src.bot.handlers.stats import (
    handle_global_stats,
    handle_hourly_stats,
    handle_plot_command,
    handle_monthly_stats,
    handle_funny_stats,
)
from src.bot.handlers.graph import (
    handle_graph_stats,
)
from src.bot.handlers.analytics import (
    handle_clusters_stats,
    handle_wordcloud_stats,
    handle_topics_stats,
    handle_trends_stats,
    handle_bursts_stats,
    handle_dominance_stats,
    handle_correlation_stats,
    handle_flooders_stats,
)

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def main():
    """Main function to run the Telegram bot."""
    db = load_db()
    messages_db = db.get("messages", [])
    client = TelegramClient('bot_session', API_ID, API_HASH)
    await client.start(bot_token=BOT_TOKEN)
    logger.info("Bot successfully started.")

    @client.on(events.NewMessage(chats=CHAT_ID))
    async def unified_handler(event):
        message = event.message
        if not message_exists(messages_db, message.id):
            sender = await message.get_sender()
            new_msg = telethon_message_to_dict(message, sender)
            messages_db.append(new_msg)
            messages_db.sort(key=lambda m: m.get("id", 0))
            db["messages"] = messages_db
            save_db(db)
            logger.info(f"Saved new message, id: {message.id}")

        if message.message.startswith("/stats"):
            tokens = message.message.split()
            if len(tokens) < 2:
                await event.respond(
                    "Invalid command format. Examples:\n"
                    "/stats global 9\n"
                    "/stats 2025-02\n"
                    "/stats 2025-02 plot 9\n"
                    "/stats plot 8 Q\n"
                    "/stats hourly\n"
                    "/stats graph [min_interactions]\n"
                    "/stats clusters [n_clusters]\n"
                    "/stats wordcloud [user]\n"
                    "/stats topics [num_topics] [user]\n"
                    "/stats trends [day|week|month]\n"
                    "/stats bursts [threshold] [window_hours]\n"
                    "/stats dominance\n"
                    "/stats correlation [H|D|W|M] [min_messages]\n"
                    "/stats flooders [min_messages] [burst_threshold_minutes]\n"
                    "/stats funny [gemini|xai]\n"
                    "/stats 2025-02 funny [gemini|xai]"
                )
                return
            cmd = tokens[1].lower()
            if cmd == "global":
                await handle_global_stats(event, tokens)
            elif cmd == "hourly":
                await handle_hourly_stats(event)
            elif cmd == "plot":
                await handle_plot_command(event, tokens, client)
            elif cmd == "funny":
                await handle_funny_stats(event, tokens)
            elif cmd == "graph":
                await handle_graph_stats(event, tokens, client)
            elif cmd == "clusters":
                await handle_clusters_stats(event, tokens, client)
            elif cmd == "wordcloud":
                await handle_wordcloud_stats(event, tokens, client)
            elif cmd == "topics":
                await handle_topics_stats(event, tokens, client)
            elif cmd == "trends":
                await handle_trends_stats(event, tokens, client)
            elif cmd == "bursts":
                await handle_bursts_stats(event, tokens)
            elif cmd == "dominance":
                await handle_dominance_stats(event, tokens)
            elif cmd == "correlation":
                await handle_correlation_stats(event, tokens)
            elif cmd == "flooders":
                await handle_flooders_stats(event, tokens)
            elif re.match(r'^\d{4}-\d{2}$', cmd):
                await handle_monthly_stats(event, tokens, client)
            else:
                await event.respond(
                    "Invalid command format. Examples:\n"
                    "/stats global 9\n"
                    "/stats 2025-02\n"
                    "/stats 2025-02 plot 9\n"
                    "/stats plot 8 Q\n"
                    "/stats hourly\n"
                    "/stats graph [min_interactions]\n"
                    "/stats clusters [n_clusters]\n"
                    "/stats wordcloud [user]\n"
                    "/stats topics [num_topics] [user]\n"
                    "/stats trends [day|week|month]\n"
                    "/stats bursts [threshold] [window_hours]\n"
                    "/stats dominance\n"
                    "/stats correlation [H|D|W|M] [min_messages]\n"
                    "/stats flooders [min_messages] [burst_threshold_minutes]\n"
                    "/stats funny [gemini|xai]\n"
                    "/stats 2025-02 funny [gemini|xai]"
                )

    logger.info("Bot waiting for messages...")
    await client.run_until_disconnected()