import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Enforce required Telegram API credentials and bot configuration
required_vars = ["API_ID", "API_HASH", "BOT_TOKEN", "CHAT_ID"]
missing = [var for var in required_vars if not os.getenv(var)]
if missing:
    raise EnvironmentError(
        f"Missing required environment variables: {', '.join(missing)}"
    )
API_ID = int(os.getenv("API_ID"))
API_HASH = os.getenv("API_HASH")
BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = int(os.getenv("CHAT_ID"))

# Support multiple chats: parse CHAT_IDS env or fallback to CHAT_ID
chat_ids_env = os.getenv("CHAT_IDS", "").strip()
if chat_ids_env:
    TELEGRAM_CHAT_IDS = [int(cid) for cid in chat_ids_env.split(',') if cid.strip()]
else:
    TELEGRAM_CHAT_IDS = [CHAT_ID]

# AI API keys configuration
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY", "")
XAI_API_KEY = os.getenv("XAI_API_KEY", "")

# Model configuration
# Available options: 'gemini' or 'xai'
DEFAULT_AI_MODEL = os.getenv("DEFAULT_AI_MODEL", "gemini")

# Path to JSON database file (optional override)
DB_FILE = os.getenv("DB_FILE", "amogus.json")

# User mapping for consistent names across the chat
USER_MAPPING = {
    'Канал Кости': 'Константин Телелюхин',
    'Амогус': 'Константин Телелюхин',
    'говно 27': 'Константин Телелюхин',
    'Святослав': 'Святослав Суглобов',
    'Ymnumi': 'Карина Кучма',
    'Karina Kuchma': 'Карина Кучма',
    'Ваня': 'Иван Яхин',
    'Николай': 'Николай Гончар',
    'Konst': 'Константин Телелюхин',
    'Egor': 'Егор Житко',
    'иви ♥ любовь моя': 'Эвелина Абрамова',
    'Максим 𒇻 𒊒𒁍𒌑 (Кузин)': 'Максим Кузин',
    'Daniil Karchenko': 'Даниил Карченко',
}