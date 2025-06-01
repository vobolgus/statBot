import os
import re
import asyncio
import logging
import html

import openai
import asyncpg
import pgvector.asyncpg
from telethon import TelegramClient, events

from config import API_ID, API_HASH, BOT_TOKEN, DATABASE_URL, TELEGRAM_CHAT_IDS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s %(message)s'
)
logger = logging.getLogger(__name__)

# Batch settings
BATCH_INTERVAL = 5  # seconds between batch uploads
# Maximum total tokens per batch (approximate)
MAX_BATCH_TOKENS = 2048

# In-memory queue for pending messages
_queue = []
_queue_lock = asyncio.Lock()

def clean_text(text: str) -> str:
    """Strip HTML tags and unescape entities."""
    text = html.unescape(text)
    # Remove any HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    return text.strip()

async def embed_and_insert(conn):
    """Embed texts in the queue and insert into Postgres."""
    async with _queue_lock:
        batch = _queue.copy()
        _queue.clear()
    if not batch:
        return
    texts = [item['text'] for item in batch]
    # Call OpenAI embedding in thread pool to avoid blocking
    try:
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: openai.Embedding.create(
                input=texts,
                model='text-embedding-ada-002'
            )
        )
        embeddings = [d['embedding'] for d in response['data']]
    except Exception as e:
        logger.exception(f"Embedding API error: {e}")
        # On failure, re-queue for next attempt
        async with _queue_lock:
            _queue[:0] = batch
        return
    # Prepare records for insertion
    records = []
    for item, emb in zip(batch, embeddings):
        records.append((
            item['id'],
            item['chat_id'],
            item['date'],
            item['text'],
            emb
        ))
    # Register pgvector type codec
    await pgvector.asyncpg.register_vector(conn)
    # Bulk insert, ignore conflicts
    stmt = (
        'INSERT INTO messages (id, chat_id, date, text, embedding) '
        'VALUES ($1, $2, $3, $4, $5) '
        'ON CONFLICT (id) DO NOTHING'
    )
    try:
        await conn.executemany(stmt, records)
        logger.info(f"Inserted {len(records)} messages into database.")
    except Exception as e:
        logger.exception(f"Database insert error: {e}")

async def _batch_loop(conn):
    """Periodically flush queue to embedding+DB."""
    while True:
        try:
            await embed_and_insert(conn)
        except Exception:
            logger.exception("Unexpected error in batch loop")
        await asyncio.sleep(BATCH_INTERVAL)

async def main():
    # Initialize OpenAI key
    openai.api_key = os.getenv('OPENAI_API_KEY')
    if not openai.api_key:
        logger.error('OPENAI_API_KEY not set')
        return
    # Connect to Postgres
    conn = await asyncpg.connect(DATABASE_URL)
    # Start batch loop
    loop_task = asyncio.create_task(_batch_loop(conn))

    # Start Telethon client
    client = TelegramClient('ingest_session', API_ID, API_HASH)
    await client.start(bot_token=BOT_TOKEN)
    logger.info('Telethon ingest client started.')

    @client.on(events.NewMessage(chats=TELEGRAM_CHAT_IDS))
    async def new_message_handler(event):
        msg = event.message
        raw = msg.raw_text or msg.message or ''
        text = clean_text(raw)
        record = {
            'id': msg.id,
            'chat_id': event.chat_id,
            'date': msg.date,
            'text': text
        }
        async with _queue_lock:
            _queue.append(record)
        logger.debug(f"Queued message {msg.id} for embedding.")

    # Run until disconnected
    await client.run_until_disconnected()
    await loop_task
    await conn.close()

if __name__ == '__main__':
    asyncio.run(main())