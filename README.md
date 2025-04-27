# Telegram Statistics Bot

A Telegram bot that tracks message statistics in a chat and generates visualizations.

## Features

- Track and save all messages in a specified chat
- Generate statistics for different time periods
- Create charts showing user activity and message distribution
- Support for various commands to analyze chat data
- Generate humorous summaries of chat statistics using multiple AI models:
  - Google's Gemini
  - X.AI's Grok

## Commands

- `/stats global N` - Show global top N users
- `/stats YYYY-MM` - Show statistics for a specific month
- `/stats YYYY-MM plot N` - Show statistics for a specific month with a chart of top N users
- `/stats plot N FREQ` - Generate a cumulative chart for all history (FREQ can be D, W, M, Q)
- `/stats hourly` - Show message distribution by hour of day
- `/stats funny [gemini|xai]` - Show global statistics with a humorous AI-generated summary
- `/stats YYYY-MM funny [gemini|xai]` - Show monthly statistics with a humorous AI-generated summary

## Setup

1. Register an application on [my.telegram.org](https://my.telegram.org/)
2. Get your API_ID and API_HASH
3. Create a bot using [@BotFather](https://t.me/botfather) and get your BOT_TOKEN
4. Add the bot to your chat and get the CHAT_ID
5. (Optional) Get API keys for AI models:
   - Google API key for Gemini from [Google AI Studio](https://ai.google.dev/)
   - X.AI API key for Grok from [X.AI](https://x.ai/)
6. Update configuration variables in the code or environment:
   ```python
   API_ID = your_api_id
   API_HASH = 'your_api_hash'
   BOT_TOKEN = 'your_bot_token'
   CHAT_ID = your_chat_id
   GOOGLE_API_KEY = 'your_gemini_api_key'  # Optional, for Gemini summaries
   XAI_API_KEY = 'your_xai_api_key'  # Optional, for Grok summaries
   DEFAULT_AI_MODEL = 'gemini'  # or 'xai', defaults to 'gemini'
   ```

## Dependencies

- telethon
- pandas
- plotly
- python-dateutil
- kaleido (for exporting plotly charts to images)
- google-generativeai (for Gemini AI summaries)
- openai (for Grok AI summaries via OpenAI-compatible API)

Install dependencies:
```
pip install -r requirements.txt
```

## Running the Bot

```
python main.py
```

## Output Examples

The bot can generate:
- Statistical text summaries for a specific month or all-time
- Monthly activity distribution by user charts
- Cumulative message share over time charts
- Hourly message distribution histograms
- Humorous AI-generated interpretations of the statistics (in Russian) using different language models:
  - 🤪 Gemini (default)
  - 🤖 Grok

Charts are saved as PNG files and sent back to the chat.

## User Mapping

To handle users with multiple names, update the USER_MAPPING dictionary:
```python
USER_MAPPING = {
    'Nickname1': 'Real Name',
    'Nickname2': 'Real Name',
    # Add your mappings here
}
```

## Data Storage

Messages are stored in a JSON file (default: amogus.json) with the following structure:
```json
{
  "name": "Chat Name",
  "type": "chat_type",
  "id": chat_id,
  "messages": [
    {
      "id": message_id,
      "type": "message",
      "date": "2023-01-01T12:00:00+00:00",
      "from": "User Name",
      "from_id": "user123456789",
      "text": "Message text"
    },
    ...
  ]
}
```