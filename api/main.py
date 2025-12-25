import asyncio
import io
import json
import logging
import os
import traceback
import uuid
import wave
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Optional

import aiosqlite
import httpx
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.date import DateTrigger
from fastapi import FastAPI, File, UploadFile, HTTPException, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, HTMLResponse, JSONResponse
from pydantic_settings import BaseSettings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.client import AsyncTcpClient
from wyoming.tts import Synthesize, SynthesizeVoice

import google_sync

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    whisper_url: str = "http://faster-whisper:8000"
    ollama_url: str = "http://ollama:11434"
    ollama_model: str = "mistral"
    ollama_embed_model: str = "nomic-embed-text"
    piper_host: str = "piper"
    piper_port: int = 10200
    xtts_url: str = "http://xtts:8020"
    xtts_speaker_file: str = "/data/xtts_speaker.json"
    tts_engine: str = "piper"  # "piper" or "xtts"
    max_history: int = 20
    session_timeout_minutes: int = 1440  # 24 hours - persist sessions all day
    db_path: str = "/data/socrates.db"
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "socrates"
    rag_enabled: bool = True
    rag_top_k: int = 3
    # Location for weather (default: Los Angeles)
    weather_lat: float = 34.0522
    weather_lon: float = -118.2437
    weather_timezone: str = "America/Los_Angeles"
    # Twilio for SMS (optional)
    twilio_account_sid: Optional[str] = None
    twilio_auth_token: Optional[str] = None
    twilio_from_number: Optional[str] = None
    system_prompt: str = """You are Socrates, a wise and thoughtful voice assistant.
You engage users with curiosity and help them think deeply about their questions.
Keep responses concise (1-3 sentences) unless more detail is requested.
Be warm, insightful, and occasionally use gentle humor."""
    # RSS feeds for news briefing (can be configured via environment)
    news_feeds: str = "https://feeds.bbci.co.uk/news/world/rss.xml,https://rss.nytimes.com/services/xml/rss/nyt/World.xml"


settings = Settings()
http_client: Optional[httpx.AsyncClient] = None
db: Optional[aiosqlite.Connection] = None
qdrant: Optional[QdrantClient] = None
scheduler: Optional[AsyncIOScheduler] = None

# Store for pending reminders (for display purposes)
pending_reminders: dict[str, dict] = {}

# XTTS speaker embeddings (loaded on startup)
xtts_speaker: Optional[dict] = None

# Personality definitions with prompts and voices
PERSONALITIES = {
    "socratic": {
        "prompt": "You are Socrates. ALWAYS respond by asking a thought-provoking question that helps the user think deeper. Never just answer directly - guide them to discover the answer themselves through questions. Be curious and philosophical. Example: Instead of 'Yes you have a meeting at 3pm', say 'What might happen if you considered how this meeting serves your larger goals? I see you have one at 3pm.'",
        "voice": "en_GB-alan-medium"  # British, thoughtful
    },
    "concise": {
        "prompt": "You are a concise assistant. Give the shortest possible accurate answer. No fluff, no elaboration unless asked. One sentence max when possible.",
        "voice": "en_US-lessac-medium"  # Clear, efficient
    },
    "critic": {
        "prompt": "You are a brutally honest critic. Point out flaws, weaknesses, and problems directly. Don't sugarcoat. Be constructive but harsh. Challenge assumptions ruthlessly.",
        "voice": "en_US-ryan-medium"  # Different male, harsher
    },
    "teacher": {
        "prompt": "You are a patient teacher. Explain concepts clearly with examples. Break down complex ideas. Ask if clarification is needed. Encourage learning.",
        "voice": "en_US-amy-medium"  # Female, patient
    },
    "creative": {
        "prompt": "You are a creative and imaginative assistant. Think outside the box. Offer unusual perspectives and creative solutions. Be playful with language and ideas.",
        "voice": "en_GB-alba-medium"  # Scottish, different
    }
}

# Session storage for conversation memory
sessions: dict[str, dict] = {}


async def init_db():
    """Initialize SQLite database."""
    global db
    os.makedirs(os.path.dirname(settings.db_path), exist_ok=True)
    db = await aiosqlite.connect(settings.db_path)
    await db.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id TEXT PRIMARY KEY,
            title TEXT,
            personality TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    await db.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id TEXT,
            role TEXT,
            content TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (conversation_id) REFERENCES conversations(id)
        )
    """)
    await db.execute("""
        CREATE TABLE IF NOT EXISTS todos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            item TEXT NOT NULL,
            completed BOOLEAN DEFAULT FALSE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    await db.execute("""
        CREATE TABLE IF NOT EXISTS expenses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            amount REAL NOT NULL,
            category TEXT,
            description TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    await db.execute("""
        CREATE TABLE IF NOT EXISTS habits (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            habit TEXT NOT NULL,
            logged_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    await db.execute("""
        CREATE TABLE IF NOT EXISTS reminders (
            id TEXT PRIMARY KEY,
            message TEXT NOT NULL,
            remind_at TIMESTAMP NOT NULL,
            fired BOOLEAN DEFAULT FALSE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    await db.commit()


def init_qdrant():
    """Initialize Qdrant client and collection."""
    global qdrant
    try:
        qdrant = QdrantClient(url=settings.qdrant_url)

        # Check if collection exists, create if not
        collections = qdrant.get_collections().collections
        exists = any(c.name == settings.qdrant_collection for c in collections)

        if not exists:
            qdrant.create_collection(
                collection_name=settings.qdrant_collection,
                vectors_config=VectorParams(size=768, distance=Distance.COSINE)
            )
            logger.info(f"Created Qdrant collection: {settings.qdrant_collection}")
        else:
            logger.info(f"Qdrant collection exists: {settings.qdrant_collection}")
    except Exception as e:
        logger.warning(f"Qdrant initialization failed: {e}. RAG disabled.")
        qdrant = None


async def get_embedding(text: str) -> list[float]:
    """Get embedding from Ollama."""
    response = await http_client.post(
        f"{settings.ollama_url}/api/embeddings",
        json={"model": settings.ollama_embed_model, "prompt": text}
    )
    if response.status_code != 200:
        raise Exception(f"Embedding failed: {response.text}")
    return response.json()["embedding"]


async def rag_search(query: str, top_k: int = None) -> list[dict]:
    """Search Qdrant for relevant context."""
    if not qdrant or not settings.rag_enabled:
        return []

    try:
        embedding = await get_embedding(query)
        results = qdrant.search(
            collection_name=settings.qdrant_collection,
            query_vector=embedding,
            limit=top_k or settings.rag_top_k
        )
        return [
            {"content": hit.payload.get("content", ""), "score": hit.score, "source": hit.payload.get("source", "")}
            for hit in results
        ]
    except Exception as e:
        logger.error(f"RAG search failed: {e}")
        return []


async def web_search(query: str, max_results: int = 5) -> list[dict]:
    """Search the web using DuckDuckGo."""
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
            return [
                {"title": r.get("title", ""), "body": r.get("body", ""), "href": r.get("href", "")}
                for r in results
            ]
    except Exception as e:
        logger.error(f"Web search failed: {e}")
        return []


async def fetch_news(max_items: int = 10) -> list[dict]:
    """Fetch news from configured RSS feeds."""
    import feedparser

    news_items = []
    feeds = settings.news_feeds.split(",")

    for feed_url in feeds:
        try:
            feed = feedparser.parse(feed_url.strip())
            for entry in feed.entries[:max_items // len(feeds) + 1]:
                news_items.append({
                    "title": entry.get("title", ""),
                    "summary": entry.get("summary", entry.get("description", ""))[:200],
                    "link": entry.get("link", ""),
                    "source": feed.feed.get("title", "Unknown")
                })
        except Exception as e:
            logger.error(f"Failed to fetch RSS feed {feed_url}: {e}")

    return news_items[:max_items]


async def get_weather() -> dict:
    """Get current weather from Open-Meteo."""
    try:
        url = (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={settings.weather_lat}&longitude={settings.weather_lon}"
            f"&current=temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m"
            f"&daily=temperature_2m_max,temperature_2m_min,weather_code,precipitation_probability_max"
            f"&temperature_unit=fahrenheit&wind_speed_unit=mph"
            f"&timezone={settings.weather_timezone}&forecast_days=3"
        )
        response = await http_client.get(url)
        if response.status_code == 200:
            data = response.json()
            current = data.get("current", {})
            daily = data.get("daily", {})

            # Weather code descriptions
            weather_codes = {
                0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast",
                45: "foggy", 48: "foggy", 51: "light drizzle", 53: "drizzle", 55: "heavy drizzle",
                61: "light rain", 63: "rain", 65: "heavy rain", 71: "light snow", 73: "snow",
                75: "heavy snow", 80: "light showers", 81: "showers", 82: "heavy showers",
                95: "thunderstorm", 96: "thunderstorm with hail", 99: "severe thunderstorm"
            }

            return {
                "current": {
                    "temp": current.get("temperature_2m"),
                    "humidity": current.get("relative_humidity_2m"),
                    "wind": current.get("wind_speed_10m"),
                    "condition": weather_codes.get(current.get("weather_code", 0), "unknown")
                },
                "forecast": [
                    {
                        "date": daily["time"][i],
                        "high": daily["temperature_2m_max"][i],
                        "low": daily["temperature_2m_min"][i],
                        "condition": weather_codes.get(daily["weather_code"][i], "unknown"),
                        "rain_chance": daily["precipitation_probability_max"][i]
                    }
                    for i in range(min(3, len(daily.get("time", []))))
                ]
            }
    except Exception as e:
        logger.error(f"Weather fetch failed: {e}")
    return None


async def get_daily_briefing_context() -> str:
    """Gather context for daily briefing: weather, calendar, emails."""
    context_parts = []

    # Get weather
    weather = await get_weather()
    if weather:
        current = weather["current"]
        forecast = weather.get("forecast", [{}])[0] if weather.get("forecast") else {}
        context_parts.append(
            f"Weather: Currently {current['temp']}°F and {current['condition']}. "
            f"Today's high {forecast.get('high', 'N/A')}°F, low {forecast.get('low', 'N/A')}°F, "
            f"{forecast.get('rain_chance', 0)}% chance of rain."
        )

    # Get today's calendar events from Qdrant
    if qdrant:
        try:
            today = datetime.now().strftime("%Y-%m-%d")
            results = qdrant.scroll(
                collection_name=settings.qdrant_collection,
                scroll_filter=Filter(
                    must=[FieldCondition(key="source", match=MatchValue(value="google_calendar"))]
                ),
                limit=20,
                with_payload=True
            )
            events = [p.payload.get("content", "") for p in results[0] if today in p.payload.get("content", "")]
            if events:
                context_parts.append(f"Today's calendar: {'; '.join(events[:5])}")
            else:
                context_parts.append("No calendar events for today.")
        except Exception as e:
            logger.error(f"Calendar briefing error: {e}")

    # Get recent emails (fresh from Gmail, not stale cache)
    try:
        emails = google_sync.get_recent_emails(max_emails=5)
        if emails:
            email_summaries = [f"{e['from'].split('<')[0].strip()}: {e['subject']}" for e in emails]
            context_parts.append(f"Recent emails: {'; '.join(email_summaries)}")
    except Exception as e:
        logger.error(f"Email briefing error: {e}")

    return "\n".join(context_parts) if context_parts else ""


async def add_to_qdrant(content: str, source: str, metadata: dict = None):
    """Add content to Qdrant for future retrieval."""
    if not qdrant:
        return

    try:
        embedding = await get_embedding(content)
        point_id = str(uuid.uuid4())
        payload = {"content": content, "source": source, "timestamp": datetime.now().isoformat()}
        if metadata:
            payload.update(metadata)

        qdrant.upsert(
            collection_name=settings.qdrant_collection,
            points=[PointStruct(id=point_id, vector=embedding, payload=payload)]
        )
    except Exception as e:
        logger.error(f"Failed to add to Qdrant: {e}")


async def fire_reminder(reminder_id: str, message: str):
    """Called when a reminder fires. Mark it as fired and log it."""
    global db
    try:
        logger.info(f"Reminder fired: {message}")
        # Mark as fired in database
        await db.execute("UPDATE reminders SET fired = TRUE WHERE id = ?", (reminder_id,))
        await db.commit()
        # Remove from pending
        pending_reminders.pop(reminder_id, None)
        # Store the fired reminder for retrieval by the next voice interaction
        pending_reminders[f"fired_{reminder_id}"] = {"message": message, "fired_at": datetime.now().isoformat()}
    except Exception as e:
        logger.error(f"Failed to fire reminder: {e}")


async def schedule_reminder(reminder_id: str, message: str, remind_at: datetime):
    """Schedule a reminder to fire at a specific time."""
    global scheduler
    if not scheduler:
        logger.error("Scheduler not initialized")
        return False

    try:
        # Store in pending
        pending_reminders[reminder_id] = {"message": message, "remind_at": remind_at.isoformat()}

        # Schedule the job
        scheduler.add_job(
            fire_reminder,
            trigger=DateTrigger(run_date=remind_at),
            args=[reminder_id, message],
            id=reminder_id,
            replace_existing=True
        )
        logger.info(f"Scheduled reminder '{message}' for {remind_at}")
        return True
    except Exception as e:
        logger.error(f"Failed to schedule reminder: {e}")
        return False


async def restore_reminders():
    """Restore pending reminders from database on startup."""
    global db
    try:
        cursor = await db.execute(
            "SELECT id, message, remind_at FROM reminders WHERE fired = FALSE AND remind_at > datetime('now')"
        )
        rows = await cursor.fetchall()
        for row in rows:
            reminder_id, message, remind_at_str = row
            remind_at = datetime.fromisoformat(remind_at_str)
            await schedule_reminder(reminder_id, message, remind_at)
        logger.info(f"Restored {len(rows)} pending reminders")
    except Exception as e:
        logger.error(f"Failed to restore reminders: {e}")


def load_xtts_speaker():
    """Load XTTS speaker embeddings from file."""
    global xtts_speaker
    if os.path.exists(settings.xtts_speaker_file):
        try:
            with open(settings.xtts_speaker_file, "r") as f:
                xtts_speaker = json.load(f)
            logger.info(f"Loaded XTTS speaker embeddings from {settings.xtts_speaker_file}")
        except Exception as e:
            logger.error(f"Failed to load XTTS speaker: {e}")
            xtts_speaker = None
    else:
        logger.warning(f"XTTS speaker file not found: {settings.xtts_speaker_file}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global http_client, scheduler
    http_client = httpx.AsyncClient(timeout=300.0)  # 5 minute timeout
    await init_db()
    init_qdrant()
    load_xtts_speaker()

    # Start the scheduler
    scheduler = AsyncIOScheduler()
    scheduler.start()
    logger.info("Scheduler started")

    # Restore pending reminders
    await restore_reminders()

    yield

    # Shutdown
    if scheduler:
        scheduler.shutdown()
    await http_client.aclose()
    if db:
        await db.close()


app = FastAPI(title="Socrates API", lifespan=lifespan)

# CORS middleware to allow custom headers
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Session-Id", "X-Transcript", "X-Response-Text"],
)


async def save_message(conversation_id: str, role: str, content: str):
    """Save a message to the database."""
    await db.execute(
        "INSERT INTO messages (conversation_id, role, content) VALUES (?, ?, ?)",
        (conversation_id, role, content)
    )
    await db.execute(
        "UPDATE conversations SET updated_at = CURRENT_TIMESTAMP WHERE id = ?",
        (conversation_id,)
    )
    await db.commit()


async def create_conversation(conversation_id: str, title: str, personality: str):
    """Create a new conversation in the database."""
    await db.execute(
        "INSERT INTO conversations (id, title, personality) VALUES (?, ?, ?)",
        (conversation_id, title, personality)
    )
    await db.commit()


async def get_conversation_messages(conversation_id: str) -> list:
    """Get all messages for a conversation."""
    cursor = await db.execute(
        "SELECT role, content FROM messages WHERE conversation_id = ? ORDER BY created_at",
        (conversation_id,)
    )
    rows = await cursor.fetchall()
    return [{"role": row[0], "content": row[1]} for row in rows]


def get_or_create_session(session_id: Optional[str]) -> tuple[str, list]:
    """Get existing session or create new one."""
    now = datetime.now()

    # Clean expired sessions
    expired = [sid for sid, data in sessions.items()
               if now - data["last_access"] > timedelta(minutes=settings.session_timeout_minutes)]
    for sid in expired:
        del sessions[sid]

    # Use a default persistent session if none provided
    # This ensures continuity across interactions
    if not session_id:
        session_id = "default-session"

    if session_id in sessions:
        sessions[session_id]["last_access"] = now
        return session_id, sessions[session_id]["history"]

    # Create new session with this ID
    sessions[session_id] = {"history": [], "last_access": now}
    return session_id, []


async def get_user_memories() -> str:
    """Load user memories and facts from Qdrant for context."""
    if not qdrant:
        return ""

    try:
        results = qdrant.scroll(
            collection_name=settings.qdrant_collection,
            scroll_filter=Filter(
                should=[
                    FieldCondition(key="source", match=MatchValue(value="memory")),
                    FieldCondition(key="source", match=MatchValue(value="user_fact"))
                ]
            ),
            limit=50,
            with_payload=True
        )

        memories = [p.payload.get("content", "") for p in results[0] if p.payload.get("content")]

        if memories:
            return "\n\nYou remember the following about the user:\n" + "\n".join([f"- {m}" for m in memories[:20]])
        return ""
    except Exception as e:
        logger.error(f"Failed to load memories: {e}")
        return ""


async def get_pending_reminders_context() -> str:
    """Get any pending reminders for context."""
    try:
        cursor = await db.execute(
            "SELECT message, remind_at FROM reminders WHERE fired = FALSE ORDER BY remind_at LIMIT 5"
        )
        rows = await cursor.fetchall()

        if rows:
            now = datetime.now()
            reminders = []
            for msg, remind_at_str in rows:
                remind_at = datetime.fromisoformat(remind_at_str)
                time_diff = remind_at - now
                if time_diff.total_seconds() > 0:
                    if time_diff.total_seconds() < 3600:
                        time_desc = f"in {int(time_diff.total_seconds() / 60)} minutes"
                    elif time_diff.total_seconds() < 86400:
                        time_desc = f"at {remind_at.strftime('%I:%M %p')}"
                    else:
                        time_desc = f"on {remind_at.strftime('%b %d')}"
                    reminders.append(f"{msg} ({time_desc})")

            if reminders:
                return "\n\nThe user has these upcoming reminders:\n" + "\n".join([f"- {r}" for r in reminders])
        return ""
    except Exception as e:
        logger.error(f"Failed to load reminders context: {e}")
        return ""


def add_to_history(session_id: str, user_msg: str, assistant_msg: str):
    """Add exchange to session history."""
    if session_id in sessions:
        history = sessions[session_id]["history"]
        history.append({"role": "user", "content": user_msg})
        history.append({"role": "assistant", "content": assistant_msg})
        # Trim to max history
        if len(history) > settings.max_history * 2:
            sessions[session_id]["history"] = history[-(settings.max_history * 2):]


async def transcribe_audio(audio_bytes: bytes, filename: str) -> str:
    """Send audio to faster-whisper-server for transcription."""
    # Detect mime type from filename
    if filename.endswith(".webm"):
        mime_type = "audio/webm"
    elif filename.endswith(".mp4") or filename.endswith(".m4a"):
        mime_type = "audio/mp4"
    else:
        mime_type = "audio/wav"
    files = {"file": (filename, audio_bytes, mime_type)}
    response = await http_client.post(
        f"{settings.whisper_url}/v1/audio/transcriptions",
        files=files,
        data={"response_format": "json"}
    )
    if response.status_code != 200:
        raise HTTPException(status_code=500, detail=f"Whisper error: {response.text}")

    result = response.json()
    return result.get("text", "")


async def query_ollama(text: str, conversation_history: list = None, system_prompt: str = None) -> str:
    """Send text to Ollama and get response."""
    prompt = system_prompt if system_prompt else settings.system_prompt
    messages = [{"role": "system", "content": prompt}]

    if conversation_history:
        messages.extend(conversation_history)

    messages.append({"role": "user", "content": text})

    payload = {
        "model": settings.ollama_model,
        "messages": messages,
        "stream": False
    }

    response = await http_client.post(
        f"{settings.ollama_url}/api/chat",
        json=payload
    )

    if response.status_code != 200:
        raise HTTPException(status_code=500, detail=f"Ollama error: {response.text}")

    result = response.json()
    return result.get("message", {}).get("content", "")


async def synthesize_speech_wyoming(text: str, voice: str = None) -> bytes:
    """Use Wyoming protocol to synthesize speech with Piper."""
    async with AsyncTcpClient(settings.piper_host, settings.piper_port) as client:
        synth_voice = SynthesizeVoice(name=voice) if voice else None
        await client.write_event(Synthesize(text=text, voice=synth_voice).event())

        audio_chunks = []
        audio_info = None

        while True:
            event = await client.read_event()
            if event is None:
                break

            if AudioStart.is_type(event.type):
                audio_start = AudioStart.from_event(event)
                audio_info = {
                    "rate": audio_start.rate,
                    "width": audio_start.width,
                    "channels": audio_start.channels
                }
            elif AudioChunk.is_type(event.type):
                chunk = AudioChunk.from_event(event)
                audio_chunks.append(chunk.audio)
            elif AudioStop.is_type(event.type):
                break

        if audio_chunks and audio_info:
            raw_audio = b"".join(audio_chunks)
            return _create_wav(
                raw_audio,
                sample_rate=audio_info["rate"],
                sample_width=audio_info["width"],
                channels=audio_info["channels"]
            )

        return b""


async def synthesize_speech_xtts(text: str, speaker: str = "default") -> bytes:
    """Use XTTS API server for high-quality TTS."""
    try:
        # Check if speaker embeddings are loaded
        if xtts_speaker is None:
            logger.warning("XTTS speaker not loaded, falling back to Piper")
            return await synthesize_speech_wyoming(text)

        # XTTS API endpoint - requires speaker embeddings
        response = await http_client.post(
            f"{settings.xtts_url}/tts",
            json={
                "text": text,
                "speaker_embedding": xtts_speaker.get("speaker_embedding"),
                "gpt_cond_latent": xtts_speaker.get("gpt_cond_latent"),
                "language": "en"
            },
            timeout=60.0  # XTTS can be slower
        )

        if response.status_code == 200:
            return response.content
        else:
            logger.error(f"XTTS error: {response.status_code} - {response.text}")
            # Fallback to Piper if XTTS fails
            return await synthesize_speech_wyoming(text)

    except Exception as e:
        logger.error(f"XTTS failed, falling back to Piper: {e}")
        return await synthesize_speech_wyoming(text)


async def synthesize_speech(text: str, voice: str = None, engine: str = None) -> bytes:
    """Synthesize speech using configured TTS engine."""
    use_engine = engine or settings.tts_engine

    if use_engine == "xtts":
        return await synthesize_speech_xtts(text, speaker=voice or "default")
    else:
        return await synthesize_speech_wyoming(text, voice=voice)


def sanitize_header(value: str, max_len: int = 500) -> str:
    """Sanitize a string for use in HTTP headers."""
    # Remove newlines and control characters
    clean = value.replace('\n', ' ').replace('\r', ' ')
    # Remove other control characters
    clean = ''.join(c if ord(c) >= 32 else ' ' for c in clean)
    # Truncate
    return clean[:max_len]


def _create_wav(raw_audio: bytes, sample_rate: int, sample_width: int, channels: int) -> bytes:
    """Create a WAV file from raw PCM audio."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(sample_width)
        wav.setframerate(sample_rate)
        wav.writeframes(raw_audio)
    return buffer.getvalue()


@app.post("/api/voice")
async def process_voice(
    audio: UploadFile = File(...),
    x_session_id: Optional[str] = Header(None),
    x_personality: Optional[str] = Header(None)
):
    """
    Main voice processing endpoint.
    Receives audio, transcribes, queries LLM, synthesizes response.
    """
    try:
        # Get or create session for conversation memory
        session_id, history = get_or_create_session(x_session_id)

        # Get personality config (x_personality is now the key, e.g. "socratic")
        personality_key = x_personality if x_personality in PERSONALITIES else "socratic"
        personality = PERSONALITIES[personality_key]

        # Read uploaded audio
        audio_bytes = await audio.read()
        logger.info(f"Received audio: {len(audio_bytes)} bytes, session: {session_id[:8]}...")

        # Step 1: Transcribe
        try:
            transcript = await transcribe_audio(audio_bytes, audio.filename or "audio.wav")
            logger.info(f"Transcription result: {transcript}")
        except Exception as e:
            logger.error(f"Transcription failed: {e}\n{traceback.format_exc()}")
            raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

        if not transcript.strip():
            raise HTTPException(status_code=400, detail="Could not transcribe audio")

        # Step 1b: Check for quick capture commands
        capture_prefixes = [
            "remember that ", "remember ", "note that ", "note ",
            "save this ", "save that ", "don't forget ", "keep in mind "
        ]
        # Also detect personal facts to save automatically
        fact_patterns = [
            "my name is ", "i am ", "i'm ", "i work at ", "i work for ",
            "my wife ", "my husband ", "my partner ", "my girlfriend ", "my boyfriend ",
            "my birthday is ", "i live in ", "i'm from ", "my favorite ", "i like ", "i love ",
            "my job is ", "my email is ", "my phone is ", "my address is "
        ]
        lower_transcript = transcript.lower()

        # Check if this is a personal fact (save silently, don't return early)
        for pattern in fact_patterns:
            if pattern in lower_transcript:
                await add_to_qdrant(transcript, source="user_fact", metadata={"type": "personal_info"})
                logger.info(f"Personal fact saved: {transcript[:50]}...")
                break

        for prefix in capture_prefixes:
            if transcript.lower().startswith(prefix):
                # Extract the content to remember
                content = transcript[len(prefix):].strip()
                if content:
                    await add_to_qdrant(content, source="memory", metadata={"type": "quick_capture"})
                    logger.info(f"Quick capture saved: {content[:50]}...")

                    # Quick confirmation response
                    response_text = "Got it, I'll remember that."
                    response_audio = await synthesize_speech(response_text, voice=personality["voice"])

                    add_to_history(session_id, transcript, response_text)

                    # Ensure conversation exists
                    cursor = await db.execute("SELECT id FROM conversations WHERE id = ?", (session_id,))
                    if not await cursor.fetchone():
                        title = transcript[:50] + "..." if len(transcript) > 50 else transcript
                        await create_conversation(session_id, title, personality_key)

                    await save_message(session_id, "user", transcript)
                    await save_message(session_id, "assistant", response_text)

                    return Response(
                        content=response_audio,
                        media_type="audio/wav",
                        headers={
                            "X-Transcript": transcript,
                            "X-Response-Text": sanitize_header(response_text),
                            "X-Session-Id": session_id
                        }
                    )
                break

        # Step 1c: Check for calendar scheduling
        schedule_patterns = ["schedule ", "add to my calendar", "create an event", "set up a meeting", "book "]
        if any(pattern in lower_transcript for pattern in schedule_patterns):
            logger.info("Schedule request detected")

            # Use LLM to parse the scheduling request
            today = datetime.now().strftime("%Y-%m-%d")
            parse_prompt = f"""Extract calendar event details from this request. Today is {today}.
Return ONLY a JSON object with these fields (no other text):
- summary: event title
- date: YYYY-MM-DD format
- time: HH:MM in 24-hour format
- duration_hours: number (default 1)

Request: {transcript}

JSON:"""

            try:
                parsed = await query_ollama(parse_prompt, [], "You are a JSON parser. Return only valid JSON, no explanation.")
                # Try to extract JSON from response
                import json as json_module
                import re
                json_match = re.search(r'\{[^}]+\}', parsed)
                if json_match:
                    event_data = json_module.loads(json_match.group())
                    start_time = f"{event_data['date']}T{event_data['time']}:00"

                    result = google_sync.create_calendar_event(
                        summary=event_data.get('summary', 'Event'),
                        start_time=start_time
                    )

                    if result.get("success"):
                        response_text = f"Scheduled '{event_data.get('summary')}' for {event_data['date']} at {event_data['time']}."
                    else:
                        response_text = f"Couldn't create event: {result.get('error', 'unknown error')}"
                else:
                    response_text = "I couldn't understand the scheduling details. Try saying 'schedule lunch tomorrow at noon'."
            except Exception as e:
                logger.error(f"Schedule parse error: {e}")
                response_text = "I had trouble parsing that. Try 'schedule meeting tomorrow at 2pm'."

            response_audio = await synthesize_speech(response_text, voice=personality["voice"])
            add_to_history(session_id, transcript, response_text)
            return Response(content=response_audio, media_type="audio/wav",
                          headers={"X-Transcript": transcript, "X-Response-Text": sanitize_header(response_text), "X-Session-Id": session_id})

        # Step 1c-1b: Check for reminder commands
        reminder_patterns = ["remind me to ", "remind me in ", "set a reminder ", "reminder to ", "reminder in "]
        if any(pattern in lower_transcript for pattern in reminder_patterns):
            logger.info("Reminder request detected")

            # Use LLM to parse the reminder request
            now = datetime.now()
            parse_prompt = f"""Extract reminder details from this request. Current time is {now.strftime("%Y-%m-%d %H:%M")}.
Return ONLY a JSON object with these fields (no other text):
- message: what to remind about
- minutes_from_now: number of minutes from now (if relative time like "in 30 minutes")
- time: HH:MM in 24-hour format (if specific time like "at 3pm")
- date: YYYY-MM-DD (if specific date mentioned, otherwise use today's date)

Examples:
"remind me to call mom in 30 minutes" -> {{"message": "call mom", "minutes_from_now": 30}}
"remind me at 3pm to take medicine" -> {{"message": "take medicine", "time": "15:00", "date": "{now.strftime("%Y-%m-%d")}"}}
"set a reminder for tomorrow at 9am to exercise" -> {{"message": "exercise", "time": "09:00", "date": "{(now + timedelta(days=1)).strftime("%Y-%m-%d")}"}}

Request: {transcript}

JSON:"""

            try:
                import json as json_module
                import re
                parsed = await query_ollama(parse_prompt, [], "You are a JSON parser. Return only valid JSON, no explanation.")
                json_match = re.search(r'\{[^}]+\}', parsed)

                if json_match:
                    reminder_data = json_module.loads(json_match.group())
                    message = reminder_data.get('message', 'reminder')

                    # Calculate remind_at time
                    if 'minutes_from_now' in reminder_data:
                        minutes = int(reminder_data['minutes_from_now'])
                        remind_at = now + timedelta(minutes=minutes)
                    elif 'time' in reminder_data:
                        date_str = reminder_data.get('date', now.strftime("%Y-%m-%d"))
                        time_str = reminder_data['time']
                        remind_at = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M")
                        # If the time is in the past today, assume tomorrow
                        if remind_at < now:
                            remind_at += timedelta(days=1)
                    else:
                        # Default to 1 hour from now
                        remind_at = now + timedelta(hours=1)

                    # Create reminder
                    reminder_id = str(uuid.uuid4())
                    await db.execute(
                        "INSERT INTO reminders (id, message, remind_at) VALUES (?, ?, ?)",
                        (reminder_id, message, remind_at.isoformat())
                    )
                    await db.commit()

                    # Schedule it
                    success = await schedule_reminder(reminder_id, message, remind_at)

                    if success:
                        time_diff = remind_at - now
                        if time_diff.total_seconds() < 3600:
                            time_desc = f"in {int(time_diff.total_seconds() / 60)} minutes"
                        elif time_diff.total_seconds() < 86400:
                            time_desc = f"at {remind_at.strftime('%I:%M %p')}"
                        else:
                            time_desc = f"on {remind_at.strftime('%B %d at %I:%M %p')}"
                        response_text = f"I'll remind you to {message} {time_desc}."
                    else:
                        response_text = "Sorry, I couldn't set that reminder."
                else:
                    response_text = "I couldn't understand the reminder. Try 'remind me to call mom in 30 minutes'."
            except Exception as e:
                logger.error(f"Reminder parse error: {e}")
                response_text = "I had trouble with that reminder. Try 'remind me in 30 minutes to take a break'."

            response_audio = await synthesize_speech(response_text, voice=personality["voice"])
            add_to_history(session_id, transcript, response_text)
            return Response(content=response_audio, media_type="audio/wav",
                          headers={"X-Transcript": transcript, "X-Response-Text": sanitize_header(response_text), "X-Session-Id": session_id})

        # Check for web search
        search_patterns = ["search for ", "search the web for ", "look up ", "google ", "find information about ", "what is ", "who is "]
        # Only trigger web search if it looks like a factual query
        web_search_indicators = ["search", "look up", "google", "find information", "wiki"]
        if any(pattern in lower_transcript for pattern in search_patterns):
            # Check if this is clearly a web search request
            is_web_search = any(ind in lower_transcript for ind in web_search_indicators)

            # Also check if it's a factual "what is" question about something specific
            if not is_web_search and ("what is " in lower_transcript or "who is " in lower_transcript):
                # Avoid triggering on conversational questions
                conversational = ["what is your", "what is my", "what is the time", "what is the weather", "what is on my"]
                is_web_search = not any(conv in lower_transcript for conv in conversational)

            if is_web_search:
                logger.info("Web search request detected")
                # Extract search query
                query = transcript
                for pattern in ["search for ", "search the web for ", "look up ", "google ", "find information about "]:
                    if pattern in lower_transcript:
                        query = transcript.lower().split(pattern, 1)[1].strip()
                        break
                for prefix in ["what is ", "who is "]:
                    if lower_transcript.startswith(prefix):
                        query = transcript[len(prefix):].strip()
                        break

                results = await web_search(query, max_results=3)
                if results:
                    # Use LLM to summarize the results
                    results_text = "\n".join([f"- {r['title']}: {r['body']}" for r in results])
                    summary_prompt = f"""Based on these search results, provide a concise answer to: {query}

Results:
{results_text}

Give a brief, informative response (2-3 sentences max)."""

                    response_text = await query_ollama(summary_prompt, [], personality["prompt"])
                else:
                    response_text = f"I couldn't find any results for '{query}'. Try rephrasing your search."

                response_audio = await synthesize_speech(response_text, voice=personality["voice"])
                add_to_history(session_id, transcript, response_text)

                cursor = await db.execute("SELECT id FROM conversations WHERE id = ?", (session_id,))
                if not await cursor.fetchone():
                    await create_conversation(session_id, f"Search: {query[:30]}", personality_key)

                await save_message(session_id, "user", transcript)
                await save_message(session_id, "assistant", response_text)

                return Response(content=response_audio, media_type="audio/wav",
                              headers={"X-Transcript": transcript, "X-Response-Text": sanitize_header(response_text), "X-Session-Id": session_id})

        # Check for news briefing
        news_patterns = ["news briefing", "what's in the news", "news update", "latest news", "give me the news", "news headlines", "today's news"]
        if any(pattern in lower_transcript for pattern in news_patterns):
            logger.info("News briefing request detected")
            news_items = await fetch_news(max_items=5)

            if news_items:
                # Use LLM to summarize the news
                news_text = "\n".join([f"- {item['title']} ({item['source']})" for item in news_items])
                summary_prompt = f"""Give a brief news briefing based on these headlines. Be concise and engaging (3-4 sentences max):

{news_text}"""

                response_text = await query_ollama(summary_prompt, [], personality["prompt"])
            else:
                response_text = "I couldn't fetch the latest news. Please check your internet connection."

            response_audio = await synthesize_speech(response_text, voice=personality["voice"])
            add_to_history(session_id, transcript, response_text)

            cursor = await db.execute("SELECT id FROM conversations WHERE id = ?", (session_id,))
            if not await cursor.fetchone():
                await create_conversation(session_id, "News Briefing", personality_key)

            await save_message(session_id, "user", transcript)
            await save_message(session_id, "assistant", response_text)

            return Response(content=response_audio, media_type="audio/wav",
                          headers={"X-Transcript": transcript, "X-Response-Text": sanitize_header(response_text), "X-Session-Id": session_id})

        # Check for listing reminders
        if any(x in lower_transcript for x in ["what reminders", "my reminders", "list reminders", "any reminders", "upcoming reminders"]):
            cursor = await db.execute(
                "SELECT message, remind_at FROM reminders WHERE fired = FALSE AND remind_at > datetime('now') ORDER BY remind_at LIMIT 5"
            )
            rows = await cursor.fetchall()

            if rows:
                reminder_list = []
                now = datetime.now()
                for msg, remind_at_str in rows:
                    remind_at = datetime.fromisoformat(remind_at_str)
                    time_diff = remind_at - now
                    if time_diff.total_seconds() < 3600:
                        time_desc = f"in {int(time_diff.total_seconds() / 60)} minutes"
                    elif time_diff.total_seconds() < 86400:
                        time_desc = f"at {remind_at.strftime('%I:%M %p')}"
                    else:
                        time_desc = f"on {remind_at.strftime('%b %d at %I:%M %p')}"
                    reminder_list.append(f"{msg} {time_desc}")
                response_text = f"You have {len(rows)} reminders: " + ", ".join(reminder_list)
            else:
                response_text = "You don't have any upcoming reminders."

            response_audio = await synthesize_speech(response_text, voice=personality["voice"])
            add_to_history(session_id, transcript, response_text)
            return Response(content=response_audio, media_type="audio/wav",
                          headers={"X-Transcript": transcript, "X-Response-Text": sanitize_header(response_text), "X-Session-Id": session_id})

        # Step 1c-2: Check for voice note command
        if any(x in lower_transcript for x in ["take a voice note", "save voice note", "record a note", "voice memo"]):
            # Save the audio and transcript
            note_dir = "/data/voice_notes"
            os.makedirs(note_dir, exist_ok=True)

            note_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            audio_path = f"{note_dir}/{note_id}.wav"
            text_path = f"{note_dir}/{note_id}.txt"

            # Save audio
            with open(audio_path, "wb") as f:
                f.write(audio_bytes)

            # Save transcript (remove the command prefix)
            note_content = transcript
            for prefix in ["take a voice note", "save voice note", "record a note", "voice memo"]:
                note_content = note_content.lower().replace(prefix, "").strip()
            note_content = note_content.strip(": ").strip() or transcript

            with open(text_path, "w") as f:
                f.write(note_content)

            response_text = "Voice note saved."
            response_audio = await synthesize_speech(response_text, voice=personality["voice"])
            add_to_history(session_id, transcript, response_text)
            return Response(content=response_audio, media_type="audio/wav",
                          headers={"X-Transcript": transcript, "X-Response-Text": sanitize_header(response_text), "X-Session-Id": session_id})

        # Step 1c-2: Check for todo list commands
        if any(x in lower_transcript for x in ["add to my list", "add to my todo", "add to list", "to my shopping list", "to my to do"]):
            # Extract the item
            for pattern in ["add ", "list ", "todo "]:
                if pattern in lower_transcript:
                    item = lower_transcript.split(pattern, 1)[-1].strip()
                    item = item.replace("to my list", "").replace("to my todo", "").replace("to my shopping list", "").strip()
                    if item:
                        await db.execute("INSERT INTO todos (item) VALUES (?)", (item,))
                        await db.commit()
                        response_text = f"Added '{item}' to your list."
                        break
            else:
                response_text = "I couldn't understand what to add."

            response_audio = await synthesize_speech(response_text, voice=personality["voice"])
            add_to_history(session_id, transcript, response_text)
            return Response(content=response_audio, media_type="audio/wav",
                          headers={"X-Transcript": transcript, "X-Response-Text": sanitize_header(response_text), "X-Session-Id": session_id})

        if any(x in lower_transcript for x in ["what's on my list", "read my list", "my todo list", "what's on my to do"]):
            cursor = await db.execute("SELECT item FROM todos WHERE completed = FALSE ORDER BY created_at DESC LIMIT 10")
            rows = await cursor.fetchall()
            if rows:
                items = [row[0] for row in rows]
                response_text = f"You have {len(items)} items: " + ", ".join(items)
            else:
                response_text = "Your list is empty."

            response_audio = await synthesize_speech(response_text, voice=personality["voice"])
            add_to_history(session_id, transcript, response_text)
            return Response(content=response_audio, media_type="audio/wav",
                          headers={"X-Transcript": transcript, "X-Response-Text": sanitize_header(response_text), "X-Session-Id": session_id})

        # Step 1c-2: Check for expense tracking
        if any(x in lower_transcript for x in ["log expense", "spent ", "log $", "add expense"]):
            import re
            # Try to extract amount
            amount_match = re.search(r'\$?(\d+(?:\.\d{2})?)', transcript)
            if amount_match:
                amount = float(amount_match.group(1))
                # Extract description (everything after the amount)
                desc = transcript.split(amount_match.group(0), 1)[-1].strip()
                desc = desc.replace("on ", "").replace("for ", "").strip() or "misc"

                await db.execute("INSERT INTO expenses (amount, description) VALUES (?, ?)", (amount, desc))
                await db.commit()
                response_text = f"Logged ${amount:.2f} for {desc}."
            else:
                response_text = "I couldn't understand the amount. Try saying 'log $50 for groceries'."

            response_audio = await synthesize_speech(response_text, voice=personality["voice"])
            add_to_history(session_id, transcript, response_text)
            return Response(content=response_audio, media_type="audio/wav",
                          headers={"X-Transcript": transcript, "X-Response-Text": sanitize_header(response_text), "X-Session-Id": session_id})

        # Step 1c-3: Check for habit tracking
        if any(x in lower_transcript for x in ["log workout", "log exercise", "i worked out", "i exercised", "log habit", "completed habit"]):
            habit = "workout"
            if "exercise" in lower_transcript:
                habit = "exercise"
            elif "meditation" in lower_transcript or "meditated" in lower_transcript:
                habit = "meditation"
            elif "reading" in lower_transcript or "read" in lower_transcript:
                habit = "reading"

            await db.execute("INSERT INTO habits (habit) VALUES (?)", (habit,))
            await db.commit()
            response_text = f"Logged {habit} for today. Keep it up!"

            response_audio = await synthesize_speech(response_text, voice=personality["voice"])
            add_to_history(session_id, transcript, response_text)
            return Response(content=response_audio, media_type="audio/wav",
                          headers={"X-Transcript": transcript, "X-Response-Text": sanitize_header(response_text), "X-Session-Id": session_id})

        # Step 1c-4: Check for calculations/math
        calc_patterns = [
            "calculate ", "what's ", "what is ", "how much is ", "convert ",
            "percent", "tip on", "divided by", "times ", "plus ", "minus ",
            "usd", "yen", "dollars", "euros", "pounds"
        ]
        math_indicators = ["calculate", "percent", "tip", "%", "+", "-", "*", "/", "divided", "times", "plus", "minus", "convert", "usd", "yen", "euro", "dollar"]

        if any(pattern in lower_transcript for pattern in calc_patterns):
            if any(ind in lower_transcript for ind in math_indicators):
                logger.info("Calculation request detected")
                calc_prompt = f"""Answer this math or conversion question. Be precise and concise. Show the result clearly.

Question: {transcript}"""

                response_text = await query_ollama(calc_prompt, [], "You are a helpful calculator. Give direct, accurate answers.")
                response_audio = await synthesize_speech(response_text, voice=personality["voice"])

                add_to_history(session_id, transcript, response_text)

                cursor = await db.execute("SELECT id FROM conversations WHERE id = ?", (session_id,))
                if not await cursor.fetchone():
                    await create_conversation(session_id, "Calculation", personality_key)

                await save_message(session_id, "user", transcript)
                await save_message(session_id, "assistant", response_text)

                return Response(
                    content=response_audio,
                    media_type="audio/wav",
                    headers={
                        "X-Transcript": transcript,
                        "X-Response-Text": sanitize_header(response_text),
                        "X-Session-Id": session_id
                    }
                )

        # Step 1d: Check for translation requests
        translation_patterns = [
            "how do you say ", "how do i say ", "translate ", "what does ",
            "what is ", "how to say ", "in japanese", "in english", "to japanese", "to english"
        ]
        if any(pattern in lower_transcript for pattern in translation_patterns):
            # Check if it's actually a translation request
            is_translation = ("japanese" in lower_transcript or "english" in lower_transcript or
                            "translate" in lower_transcript or
                            ("how do you say" in lower_transcript or "how do i say" in lower_transcript))

            if is_translation:
                logger.info("Translation request detected")
                translation_prompt = f"""Translate or explain the following. If translating to Japanese, provide both the Japanese characters and romaji pronunciation. Be concise.

Request: {transcript}"""

                response_text = await query_ollama(translation_prompt, [], personality["prompt"])
                response_audio = await synthesize_speech(response_text, voice=personality["voice"])

                add_to_history(session_id, transcript, response_text)

                cursor = await db.execute("SELECT id FROM conversations WHERE id = ?", (session_id,))
                if not await cursor.fetchone():
                    await create_conversation(session_id, "Translation", personality_key)

                await save_message(session_id, "user", transcript)
                await save_message(session_id, "assistant", response_text)

                return Response(
                    content=response_audio,
                    media_type="audio/wav",
                    headers={
                        "X-Transcript": transcript,
                        "X-Response-Text": sanitize_header(response_text),
                        "X-Session-Id": session_id
                    }
                )

        # Step 1d: Check for send email command
        send_email_patterns = ["send email to ", "send an email to ", "email to ", "send a message to "]
        for pattern in send_email_patterns:
            if pattern in lower_transcript:
                # Extract name and message
                after_pattern = transcript.lower().split(pattern, 1)[1]
                parts = after_pattern.split(" saying ", 1)

                if len(parts) >= 1:
                    recipient_name = parts[0].strip()
                    message_body = parts[1].strip() if len(parts) > 1 else "Hello from Socrates"

                    # Look up contact
                    email = google_sync.lookup_contact(recipient_name)

                    if email:
                        result = google_sync.send_email(email, f"Message from Socrates", message_body)
                        if result.get("success"):
                            response_text = f"Email sent to {recipient_name}."
                        else:
                            response_text = f"Failed to send email: {result.get('error', 'unknown error')}"
                    else:
                        response_text = f"I don't have an email address for {recipient_name}. Try syncing your contacts."

                    response_audio = await synthesize_speech(response_text, voice=personality["voice"])
                    add_to_history(session_id, transcript, response_text)

                    cursor = await db.execute("SELECT id FROM conversations WHERE id = ?", (session_id,))
                    if not await cursor.fetchone():
                        await create_conversation(session_id, f"Email to {recipient_name}", personality_key)

                    await save_message(session_id, "user", transcript)
                    await save_message(session_id, "assistant", response_text)

                    return Response(
                        content=response_audio,
                        media_type="audio/wav",
                        headers={
                            "X-Transcript": transcript,
                            "X-Response-Text": sanitize_header(response_text),
                            "X-Session-Id": session_id
                        }
                    )
                break

        # Step 1d: Check for email summarization
        email_summary_triggers = ["summarize my emails", "summarize emails", "email summary", "what emails do i have", "any important emails", "read my emails", "check my emails", "check emails"]
        if any(trigger in lower_transcript for trigger in email_summary_triggers):
            try:
                # Fetch fresh emails directly from Gmail
                emails = google_sync.get_recent_emails(max_emails=15)

                if emails:
                    logger.info(f"Fetched {len(emails)} fresh emails from Gmail")
                    email_text = "\n".join([
                        f"- From: {e['from']}, Subject: {e['subject']}, Preview: {e['snippet'][:100]}"
                        for e in emails
                    ])
                    summary_prompt = f"""Summarize these emails briefly. Highlight any urgent or important items. Group by sender or topic if helpful. Be concise (4-5 sentences max):

{email_text}"""
                    response_text = await query_ollama(summary_prompt, [], personality["prompt"])
                else:
                    if google_sync.is_authenticated():
                        response_text = "Your inbox appears to be empty, or I couldn't fetch emails right now."
                    else:
                        response_text = "I'm not connected to your Gmail. Please authenticate first."

                response_audio = await synthesize_speech(response_text, voice=personality["voice"])
                add_to_history(session_id, transcript, response_text)

                cursor = await db.execute("SELECT id FROM conversations WHERE id = ?", (session_id,))
                if not await cursor.fetchone():
                    await create_conversation(session_id, "Email Summary", personality_key)

                await save_message(session_id, "user", transcript)
                await save_message(session_id, "assistant", response_text)

                return Response(
                    content=response_audio,
                    media_type="audio/wav",
                    headers={
                        "X-Transcript": transcript,
                        "X-Response-Text": sanitize_header(response_text),
                        "X-Session-Id": session_id
                    }
                )
            except Exception as e:
                logger.error(f"Email summary error: {e}")
                # Continue to normal processing instead of 500

        # Step 1d: Check for daily briefing triggers
        briefing_triggers = ["good morning", "start my day", "daily briefing", "what's on my agenda", "brief me"]
        if any(trigger in lower_transcript for trigger in briefing_triggers):
            briefing_context = await get_daily_briefing_context()

            # Add reminders to briefing
            reminders_ctx = await get_pending_reminders_context()
            if reminders_ctx:
                briefing_context += "\n" + reminders_ctx.replace("\n\nThe user has these upcoming reminders:", "Reminders:")

            # Add user memories for personalization
            user_memories = await get_user_memories()

            if briefing_context:
                logger.info("Daily briefing triggered")

                # Use LLM to create a natural briefing
                system_with_memory = personality["prompt"]
                if user_memories:
                    system_with_memory += user_memories

                briefing_prompt = f"""Give a friendly, personalized morning briefing based on this information. Greet the user by name if you know it. Keep it conversational and under 5 sentences:

{briefing_context}"""

                response_text = await query_ollama(briefing_prompt, [], system_with_memory)
                response_audio = await synthesize_speech(response_text, voice=personality["voice"])

                add_to_history(session_id, transcript, response_text)

                cursor = await db.execute("SELECT id FROM conversations WHERE id = ?", (session_id,))
                if not await cursor.fetchone():
                    await create_conversation(session_id, "Daily Briefing", personality_key)

                await save_message(session_id, "user", transcript)
                await save_message(session_id, "assistant", response_text)

                return Response(
                    content=response_audio,
                    media_type="audio/wav",
                    headers={
                        "X-Transcript": transcript,
                        "X-Response-Text": sanitize_header(response_text),
                        "X-Session-Id": session_id
                    }
                )

        # Step 2: RAG search for context
        rag_context = ""
        try:
            rag_results = await rag_search(transcript)
            if rag_results:
                # Include results with score > 0.3
                relevant = [r for r in rag_results if r['score'] > 0.3]
                if relevant:
                    rag_context = "\n\n---\nYou have access to the following information. Use it to inform your response (while staying in character with your personality):\n" + "\n".join(
                        [f"- {r['content']}" for r in relevant]
                    ) + "\n---"
                logger.info(f"RAG found {len(rag_results)} results, {len(relevant)} above threshold")
        except Exception as e:
            logger.error(f"RAG search error: {e}")

        # Step 2b: Check for weather-related queries
        weather_context = ""
        weather_keywords = ["weather", "temperature", "forecast", "rain", "snow", "cold", "hot", "warm", "sunny", "cloudy"]
        if any(kw in transcript.lower() for kw in weather_keywords):
            weather = await get_weather()
            if weather:
                current = weather["current"]
                forecast = weather["forecast"]
                weather_context = f"\n\nCurrent weather: {current['temp']}°F, {current['condition']}, humidity {current['humidity']}%, wind {current['wind']} mph."
                if forecast:
                    weather_context += f" Today's forecast: high {forecast[0]['high']}°F, low {forecast[0]['low']}°F, {forecast[0]['condition']}, {forecast[0]['rain_chance']}% chance of rain."
                    if len(forecast) > 1:
                        weather_context += f" Tomorrow: high {forecast[1]['high']}°F, {forecast[1]['condition']}."
                logger.info(f"Weather context added")

        # Step 3: Query Ollama with conversation history
        try:
            system_prompt = personality["prompt"]

            # Add persistent user memories
            user_memories = await get_user_memories()
            if user_memories:
                system_prompt += user_memories

            # Add pending reminders context
            reminders_context = await get_pending_reminders_context()
            if reminders_context:
                system_prompt += reminders_context

            if rag_context:
                system_prompt += rag_context
            if weather_context:
                system_prompt += weather_context
            logger.info(f"Using personality: {personality_key}, voice: {personality['voice']}")
            response_text = await query_ollama(transcript, history, system_prompt)
            logger.info(f"Ollama response: {response_text[:200]}...")
        except Exception as e:
            logger.error(f"Ollama failed: {e}\n{traceback.format_exc()}")
            raise HTTPException(status_code=500, detail=f"Ollama failed: {str(e)}")

        # Save to conversation history (memory)
        add_to_history(session_id, transcript, response_text)

        # Save to database
        # Check if this is a new conversation
        cursor = await db.execute("SELECT id FROM conversations WHERE id = ?", (session_id,))
        if not await cursor.fetchone():
            # Create new conversation with first user message as title
            title = transcript[:50] + "..." if len(transcript) > 50 else transcript
            await create_conversation(session_id, title, system_prompt[:50])

        await save_message(session_id, "user", transcript)
        await save_message(session_id, "assistant", response_text)

        # Step 4: Synthesize speech with personality voice
        try:
            response_audio = await synthesize_speech(response_text, voice=personality["voice"])
            logger.info(f"TTS result: {len(response_audio)} bytes")
        except Exception as e:
            logger.error(f"TTS failed: {e}\n{traceback.format_exc()}")
            raise HTTPException(status_code=500, detail=f"TTS failed: {str(e)}")

        return Response(
            content=response_audio,
            media_type="audio/wav",
            headers={
                "X-Transcript": transcript,
                "X-Response-Text": sanitize_header(response_text),
                "X-Session-Id": session_id
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


@app.post("/api/transcribe")
async def transcribe_only(audio: UploadFile = File(...)):
    """Transcribe audio without LLM processing."""
    audio_bytes = await audio.read()
    transcript = await transcribe_audio(audio_bytes, audio.filename or "audio.wav")
    return {"text": transcript}


@app.post("/api/chat")
async def chat_text(text: str):
    """Text-only chat endpoint."""
    response = await query_ollama(text)
    return {"response": response}


@app.post("/api/tts")
async def text_to_speech(text: str):
    """Text-to-speech only endpoint."""
    audio = await synthesize_speech_wyoming(text)
    return Response(content=audio, media_type="audio/wav")


@app.post("/api/clear")
async def clear_session(x_session_id: Optional[str] = Header(None)):
    """Clear conversation history for a session."""
    if x_session_id and x_session_id in sessions:
        sessions[x_session_id]["history"] = []
        return {"status": "cleared", "session_id": x_session_id}
    return {"status": "no_session"}


@app.get("/api/conversations")
async def list_conversations():
    """List all saved conversations."""
    cursor = await db.execute(
        "SELECT id, title, personality, created_at, updated_at FROM conversations ORDER BY updated_at DESC LIMIT 50"
    )
    rows = await cursor.fetchall()
    return {
        "conversations": [
            {
                "id": row[0],
                "title": row[1],
                "personality": row[2],
                "created_at": row[3],
                "updated_at": row[4]
            }
            for row in rows
        ]
    }


@app.get("/api/conversations/{conversation_id}")
async def get_conversation(conversation_id: str):
    """Get a specific conversation with messages."""
    cursor = await db.execute(
        "SELECT id, title, personality FROM conversations WHERE id = ?",
        (conversation_id,)
    )
    conv = await cursor.fetchone()
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")

    messages = await get_conversation_messages(conversation_id)
    return {
        "id": conv[0],
        "title": conv[1],
        "personality": conv[2],
        "messages": messages
    }


@app.delete("/api/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """Delete a conversation."""
    await db.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
    await db.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
    await db.commit()
    return {"status": "deleted"}


@app.post("/api/knowledge")
async def add_knowledge(content: str, source: str = "manual"):
    """Add content to the knowledge base."""
    if not qdrant:
        raise HTTPException(status_code=503, detail="Qdrant not available")

    await add_to_qdrant(content, source)
    return {"status": "added", "source": source}


@app.get("/api/knowledge/search")
async def search_knowledge(q: str, limit: int = 5):
    """Search the knowledge base."""
    results = await rag_search(q, top_k=limit)
    return {"results": results}


@app.get("/api/google/status")
async def google_status():
    """Check Google auth status."""
    return {"authenticated": google_sync.is_authenticated()}


@app.get("/api/google/auth")
async def google_auth_start(request: Request):
    """Start Google OAuth - redirects to Google."""
    from starlette.responses import RedirectResponse

    # Build redirect URI based on request
    redirect_uri = f"https://voice.iikomedia.com/api/google/callback"

    try:
        url, state = google_sync.get_auth_url(redirect_uri)
        return RedirectResponse(url)
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Google credentials not configured")


@app.get("/api/google/callback")
async def google_auth_callback(code: str = None, error: str = None):
    """Handle Google OAuth callback."""
    if error:
        return HTMLResponse(f"<h1>Auth Error</h1><p>{error}</p>")

    if not code:
        return HTMLResponse("<h1>Error</h1><p>No code received</p>")

    redirect_uri = "https://voice.iikomedia.com/api/google/callback"
    success = google_sync.complete_auth(code, redirect_uri)

    if success:
        return HTMLResponse("<h1>Success!</h1><p>Google account connected. You can close this window.</p><script>setTimeout(() => window.close(), 2000)</script>")
    return HTMLResponse("<h1>Error</h1><p>Authentication failed</p>")


@app.post("/api/google/calendar/create")
async def create_calendar_event(summary: str, start_time: str, end_time: str = None, description: str = ""):
    """Create a calendar event."""
    result = google_sync.create_calendar_event(summary, start_time, end_time, description)
    return result


@app.post("/api/google/sync/calendar")
async def sync_google_calendar():
    """Sync Google Calendar to knowledge base."""
    result = await google_sync.sync_calendar(add_to_qdrant)
    return result


@app.post("/api/google/sync/gmail")
async def sync_google_gmail(max_emails: int = 50):
    """Sync Gmail to knowledge base."""
    result = await google_sync.sync_gmail(add_to_qdrant, max_emails)
    return result


@app.post("/api/google/sync/sheets")
async def sync_google_sheet(spreadsheet_id: str, range_name: str = "A1:Z100"):
    """Sync a Google Sheet to knowledge base."""
    result = await google_sync.sync_sheet(add_to_qdrant, spreadsheet_id, range_name)
    return result


@app.post("/api/google/send-email")
async def send_email_endpoint(to: str, subject: str, body: str):
    """Send an email via Gmail."""
    result = google_sync.send_email(to, subject, body)
    return result


@app.post("/api/google/sync/contacts")
async def sync_google_contacts():
    """Sync Google Contacts."""
    result = google_sync.sync_contacts()
    return result


@app.get("/api/google/contacts")
async def get_contacts():
    """Get all synced contacts."""
    contacts = google_sync.get_all_contacts()
    return {"contacts": contacts, "count": len(contacts)}


@app.get("/api/google/contacts/lookup")
async def lookup_contact(name: str):
    """Look up a contact's email by name."""
    email = google_sync.lookup_contact(name)
    if email:
        return {"name": name, "email": email}
    raise HTTPException(status_code=404, detail=f"Contact '{name}' not found")


@app.post("/api/sms/send")
async def send_sms(to: str, message: str):
    """Send an SMS via Twilio."""
    if not all([settings.twilio_account_sid, settings.twilio_auth_token, settings.twilio_from_number]):
        raise HTTPException(status_code=503, detail="Twilio not configured")

    try:
        from twilio.rest import Client
        client = Client(settings.twilio_account_sid, settings.twilio_auth_token)

        msg = client.messages.create(
            body=message,
            from_=settings.twilio_from_number,
            to=to
        )

        return {"success": True, "sid": msg.sid}
    except Exception as e:
        logger.error(f"SMS send failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/sms/status")
async def sms_status():
    """Check if Twilio is configured."""
    configured = all([settings.twilio_account_sid, settings.twilio_auth_token, settings.twilio_from_number])
    return {"configured": configured}


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "qdrant": qdrant is not None,
        "google": google_sync.is_authenticated(),
        "twilio": all([settings.twilio_account_sid, settings.twilio_auth_token, settings.twilio_from_number]),
        "tts_engine": settings.tts_engine
    }


@app.get("/api/tts/status")
async def tts_status():
    """Get current TTS engine status."""
    xtts_available = False
    try:
        response = await http_client.get(f"{settings.xtts_url}/", timeout=5.0)
        xtts_available = response.status_code == 200
    except Exception:
        pass

    return {
        "current_engine": settings.tts_engine,
        "xtts_available": xtts_available,
        "xtts_url": settings.xtts_url
    }


@app.post("/api/tts/engine")
async def set_tts_engine(engine: str):
    """Switch TTS engine (piper or xtts)."""
    if engine not in ["piper", "xtts"]:
        raise HTTPException(status_code=400, detail="Engine must be 'piper' or 'xtts'")
    settings.tts_engine = engine
    return {"status": "ok", "engine": engine}


@app.get("/api/qdrant/sources")
async def get_qdrant_sources():
    """Get counts of items in Qdrant by source."""
    if not qdrant:
        raise HTTPException(status_code=503, detail="Qdrant not available")

    sources = ["gmail", "google_calendar", "memory", "user_fact", "google_sheets"]
    counts = {}

    for source in sources:
        try:
            results = qdrant.scroll(
                collection_name=settings.qdrant_collection,
                scroll_filter=Filter(
                    must=[FieldCondition(key="source", match=MatchValue(value=source))]
                ),
                limit=1000,
                with_payload=False
            )
            counts[source] = len(results[0])
        except Exception:
            counts[source] = 0

    return {"sources": counts}


@app.get("/api/qdrant/data/{source}")
async def get_qdrant_data(source: str, limit: int = 50):
    """Get data from Qdrant by source."""
    if not qdrant:
        raise HTTPException(status_code=503, detail="Qdrant not available")

    try:
        results = qdrant.scroll(
            collection_name=settings.qdrant_collection,
            scroll_filter=Filter(
                must=[FieldCondition(key="source", match=MatchValue(value=source))]
            ),
            limit=limit,
            with_payload=True
        )

        items = [
            {"id": str(p.id), "content": p.payload.get("content", ""), "metadata": p.payload}
            for p in results[0]
        ]
        return {"source": source, "count": len(items), "items": items}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/qdrant/source/{source}")
async def clear_qdrant_source(source: str):
    """Delete all items from Qdrant for a specific source."""
    if not qdrant:
        raise HTTPException(status_code=503, detail="Qdrant not available")

    try:
        # Get all IDs for this source
        results = qdrant.scroll(
            collection_name=settings.qdrant_collection,
            scroll_filter=Filter(
                must=[FieldCondition(key="source", match=MatchValue(value=source))]
            ),
            limit=10000,
            with_payload=False
        )

        ids_to_delete = [p.id for p in results[0]]

        if ids_to_delete:
            qdrant.delete(
                collection_name=settings.qdrant_collection,
                points_selector=ids_to_delete
            )

        return {"status": "deleted", "source": source, "count": len(ids_to_delete)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/qdrant/item/{item_id}")
async def delete_qdrant_item(item_id: str):
    """Delete a specific item from Qdrant."""
    if not qdrant:
        raise HTTPException(status_code=503, detail="Qdrant not available")

    try:
        qdrant.delete(
            collection_name=settings.qdrant_collection,
            points_selector=[item_id]
        )
        return {"status": "deleted", "id": item_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/personalities")
async def list_personalities():
    """List available personalities."""
    return {
        "personalities": [
            {"key": key, "voice": p["voice"]}
            for key, p in PERSONALITIES.items()
        ]
    }


@app.get("/api/weather")
async def weather_endpoint():
    """Get current weather."""
    weather = await get_weather()
    if weather:
        return weather
    raise HTTPException(status_code=503, detail="Weather service unavailable")


@app.get("/api/search")
async def search_endpoint(q: str, limit: int = 5):
    """Search the web using DuckDuckGo."""
    results = await web_search(q, max_results=limit)
    return {"query": q, "results": results}


@app.get("/api/news")
async def news_endpoint(limit: int = 10):
    """Get latest news from RSS feeds."""
    news = await fetch_news(max_items=limit)
    return {"news": news}


@app.get("/api/voice-notes")
async def get_voice_notes():
    """Get list of voice notes."""
    note_dir = "/data/voice_notes"
    notes = []
    if os.path.exists(note_dir):
        for f in sorted(os.listdir(note_dir), reverse=True):
            if f.endswith(".txt"):
                note_id = f.replace(".txt", "")
                text_path = f"{note_dir}/{f}"
                with open(text_path, "r") as tf:
                    content = tf.read()
                notes.append({"id": note_id, "content": content, "has_audio": os.path.exists(f"{note_dir}/{note_id}.wav")})
    return {"notes": notes}


@app.get("/api/voice-notes/{note_id}/audio")
async def get_voice_note_audio(note_id: str):
    """Get voice note audio file."""
    audio_path = f"/data/voice_notes/{note_id}.wav"
    if os.path.exists(audio_path):
        with open(audio_path, "rb") as f:
            return Response(content=f.read(), media_type="audio/wav")
    raise HTTPException(status_code=404, detail="Voice note not found")


@app.get("/api/todos")
async def get_todos(include_completed: bool = False):
    """Get todo list items."""
    if include_completed:
        cursor = await db.execute("SELECT id, item, completed, created_at FROM todos ORDER BY created_at DESC")
    else:
        cursor = await db.execute("SELECT id, item, completed, created_at FROM todos WHERE completed = FALSE ORDER BY created_at DESC")
    rows = await cursor.fetchall()
    return {"todos": [{"id": r[0], "item": r[1], "completed": bool(r[2]), "created_at": r[3]} for r in rows]}


@app.post("/api/todos")
async def add_todo(item: str):
    """Add a todo item."""
    await db.execute("INSERT INTO todos (item) VALUES (?)", (item,))
    await db.commit()
    return {"status": "added", "item": item}


@app.delete("/api/todos/{todo_id}")
async def delete_todo(todo_id: int):
    """Delete or complete a todo item."""
    await db.execute("UPDATE todos SET completed = TRUE WHERE id = ?", (todo_id,))
    await db.commit()
    return {"status": "completed"}


@app.get("/api/expenses")
async def get_expenses(days: int = 30):
    """Get recent expenses."""
    cursor = await db.execute(
        "SELECT id, amount, category, description, created_at FROM expenses WHERE created_at > datetime('now', ?) ORDER BY created_at DESC",
        (f"-{days} days",)
    )
    rows = await cursor.fetchall()
    total = sum(r[1] for r in rows)
    return {
        "expenses": [{"id": r[0], "amount": r[1], "category": r[2], "description": r[3], "created_at": r[4]} for r in rows],
        "total": total,
        "days": days
    }


@app.post("/api/expenses")
async def add_expense(amount: float, description: str = "", category: str = ""):
    """Add an expense."""
    await db.execute("INSERT INTO expenses (amount, description, category) VALUES (?, ?, ?)", (amount, description, category))
    await db.commit()
    return {"status": "logged", "amount": amount}


@app.get("/api/habits")
async def get_habits(days: int = 30):
    """Get habit log."""
    cursor = await db.execute(
        "SELECT habit, COUNT(*) as count FROM habits WHERE logged_at > datetime('now', ?) GROUP BY habit",
        (f"-{days} days",)
    )
    rows = await cursor.fetchall()
    return {"habits": {r[0]: r[1] for r in rows}, "days": days}


@app.post("/api/habits")
async def log_habit(habit: str):
    """Log a habit."""
    await db.execute("INSERT INTO habits (habit) VALUES (?)", (habit,))
    await db.commit()
    return {"status": "logged", "habit": habit}


@app.get("/api/reminders")
async def get_reminders(include_fired: bool = False):
    """Get reminders."""
    if include_fired:
        cursor = await db.execute(
            "SELECT id, message, remind_at, fired, created_at FROM reminders ORDER BY remind_at DESC LIMIT 50"
        )
    else:
        cursor = await db.execute(
            "SELECT id, message, remind_at, fired, created_at FROM reminders WHERE fired = FALSE ORDER BY remind_at ASC"
        )
    rows = await cursor.fetchall()
    return {
        "reminders": [
            {"id": r[0], "message": r[1], "remind_at": r[2], "fired": bool(r[3]), "created_at": r[4]}
            for r in rows
        ]
    }


@app.post("/api/reminders")
async def create_reminder(message: str, remind_at: str):
    """Create a reminder. remind_at should be ISO format datetime."""
    reminder_id = str(uuid.uuid4())
    remind_at_dt = datetime.fromisoformat(remind_at)

    await db.execute(
        "INSERT INTO reminders (id, message, remind_at) VALUES (?, ?, ?)",
        (reminder_id, message, remind_at_dt.isoformat())
    )
    await db.commit()

    success = await schedule_reminder(reminder_id, message, remind_at_dt)
    return {"status": "created" if success else "db_only", "id": reminder_id, "remind_at": remind_at}


@app.delete("/api/reminders/{reminder_id}")
async def delete_reminder(reminder_id: str):
    """Delete/cancel a reminder."""
    global scheduler
    # Remove from scheduler
    try:
        scheduler.remove_job(reminder_id)
    except Exception:
        pass  # Job might not exist

    # Remove from pending
    pending_reminders.pop(reminder_id, None)

    # Mark as fired in database (or delete)
    await db.execute("DELETE FROM reminders WHERE id = ?", (reminder_id,))
    await db.commit()
    return {"status": "deleted"}


@app.get("/api/reminders/fired")
async def get_fired_reminders():
    """Get any reminders that have fired since last check (for notification)."""
    fired = {k: v for k, v in pending_reminders.items() if k.startswith("fired_")}
    # Clear them after reading
    for k in list(fired.keys()):
        pending_reminders.pop(k, None)
    return {"fired": list(fired.values())}


@app.get("/api/memories")
async def get_memories(limit: int = 50):
    """Get all saved memories and user facts."""
    if not qdrant:
        raise HTTPException(status_code=503, detail="Qdrant not available")

    try:
        # Scroll through points to get memories
        results = qdrant.scroll(
            collection_name=settings.qdrant_collection,
            scroll_filter=Filter(
                should=[
                    FieldCondition(key="source", match=MatchValue(value="memory")),
                    FieldCondition(key="source", match=MatchValue(value="user_fact"))
                ]
            ),
            limit=limit,
            with_payload=True
        )

        memories = [
            {
                "content": point.payload.get("content", ""),
                "source": point.payload.get("source", ""),
                "type": point.payload.get("type", ""),
                "timestamp": point.payload.get("timestamp", "")
            }
            for point in results[0]
        ]

        return {"memories": memories, "count": len(memories)}
    except Exception as e:
        logger.error(f"Failed to get memories: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/static/icon-{size}.png")
async def serve_icon(size: int):
    """Generate app icon with S design."""
    import struct
    import zlib
    import math

    width = height = size
    center = size // 2
    radius = int(size * 0.45)

    def create_png_rgba(w, h, pixels):
        def chunk(chunk_type, data):
            chunk_len = len(data)
            chunk_data = chunk_type + data
            checksum = zlib.crc32(chunk_data) & 0xffffffff
            return struct.pack('>I', chunk_len) + chunk_data + struct.pack('>I', checksum)

        sig = b'\x89PNG\r\n\x1a\n'
        ihdr_data = struct.pack('>IIBBBBB', w, h, 8, 6, 0, 0, 0)  # 8-bit RGBA
        ihdr = chunk(b'IHDR', ihdr_data)

        raw_data = b''
        for row in pixels:
            raw_data += b'\x00'
            for pixel in row:
                raw_data += bytes(pixel)

        compressed = zlib.compress(raw_data, 9)
        idat = chunk(b'IDAT', compressed)
        iend = chunk(b'IEND', b'')

        return sig + ihdr + idat + iend

    # Create pixel array
    pixels = []
    blue = [37, 99, 235, 255]
    white = [255, 255, 255, 255]

    for y in range(height):
        row = []
        for x in range(width):
            # Distance from center
            dx = x - center
            dy = y - center
            dist = math.sqrt(dx*dx + dy*dy)

            if dist <= radius:
                # Inside circle - check if we're drawing the S
                # Normalize coordinates to -1 to 1
                nx = dx / radius
                ny = dy / radius

                # Simple S shape using sine curve
                is_s = False
                s_width = 0.35

                # S curve: x = sin(y * pi) * 0.4
                target_x = math.sin(ny * math.pi) * 0.4

                if abs(nx - target_x) < s_width:
                    is_s = True

                # Top and bottom caps
                if ny < -0.6 and nx > target_x - s_width and nx < 0.5:
                    is_s = True
                if ny > 0.6 and nx < target_x + s_width and nx > -0.5:
                    is_s = True

                row.append(white if is_s else blue)
            else:
                row.append([255, 255, 255, 0])  # Transparent outside

        pixels.append(row)

    png_data = create_png_rgba(width, height, pixels)
    return Response(content=png_data, media_type="image/png")


@app.get("/static/{filename}")
async def serve_static(filename: str):
    """Serve static files."""
    import os
    filepath = os.path.join("static", filename)
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            content = f.read()

        content_type = "text/plain"
        if filename.endswith(".json"):
            content_type = "application/json"
        elif filename.endswith(".js"):
            content_type = "application/javascript"
        elif filename.endswith(".css"):
            content_type = "text/css"

        return Response(content=content, media_type=content_type)
    raise HTTPException(status_code=404)


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main web UI."""
    with open("static/index.html", "r") as f:
        return f.read()
