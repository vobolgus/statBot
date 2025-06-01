import os
import json
import logging

from telethon.tl.types import User, Channel
from config import DB_FILE

logger = logging.getLogger(__name__)

def load_db():
    """
    Load message database from JSON file.
    If file doesn't exist or error occurs, return minimal structure.
    Expected structure:
    {
      "name": ..., "type": ..., "id": ..., "messages": [...] }
    """
    if os.path.exists(DB_FILE):
        try:
            with open(DB_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if "messages" not in data:
                data["messages"] = []
            return data
        except Exception as e:
            logger.error(f"Error loading database: {e}")
            return {"name": "", "type": "", "id": 0, "messages": []}
    else:
        return {"name": "", "type": "", "id": 0, "messages": []}

def save_db(db):
    """Save message database to JSON file."""
    try:
        with open(DB_FILE, 'w', encoding='utf-8') as f:
            json.dump(db, f, ensure_ascii=False, indent=4)
    except Exception as e:
        logger.error(f"Error saving database: {e}")

def message_exists(messages, msg_id):
    """Check if message with given ID exists in database."""
    return any(m.get("id") == msg_id for m in messages)

def telethon_message_to_dict(message, sender):
    """
    Convert Telethon message object to dictionary for storage.
    Handles regular and service messages.
    """
    result = {
        "id": message.id,
        "date": message.date.isoformat() if message.date else None,
        "date_unixtime": str(int(message.date.timestamp())) if message.date else None,
    }

    if message.action is not None:
        result["type"] = "service"
        if sender is not None:
            if isinstance(sender, User):
                result["actor"] = sender.first_name or str(sender.id)
                result["actor_id"] = f"user{sender.id}"
            elif isinstance(sender, Channel):
                result["actor"] = sender.title or str(sender.id)
                result["actor_id"] = f"channel{sender.id}"
            else:
                result["actor"] = str(sender)
                result["actor_id"] = str(getattr(sender, 'id', "unknown"))
        else:
            result["actor"] = "unknown"
            result["actor_id"] = "unknown"
        result["action"] = message.action.__class__.__name__
    else:
        result["type"] = "message"
        if sender is not None:
            if isinstance(sender, User):
                result["from"] = sender.first_name or str(sender.id)
                result["from_id"] = f"user{sender.id}"
            elif isinstance(sender, Channel):
                result["from"] = sender.title or str(sender.id)
                result["from_id"] = f"channel{sender.id}"
            else:
                result["from"] = str(sender)
                result["from_id"] = str(getattr(sender, 'id', "unknown"))
        else:
            result["from"] = "unknown"
            result["from_id"] = "unknown"

    result["text"] = message.message or ""
    if message.reply_to_msg_id:
        result["reply_to_message_id"] = message.reply_to_msg_id
    if message.edit_date:
        result["edited"] = message.edit_date.isoformat()
        result["edited_unixtime"] = str(int(message.edit_date.timestamp()))
    result["reactions"] = []
    return result