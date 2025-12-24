import asyncio
import io
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
from fastapi import FastAPI, File, UploadFile, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, HTMLResponse, JSONResponse
from pydantic_settings import BaseSettings
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.client import AsyncTcpClient
from wyoming.tts import Synthesize

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    whisper_url: str = "http://faster-whisper:8000"
    ollama_url: str = "http://ollama:11434"
    ollama_model: str = "mistral"
    piper_host: str = "piper"
    piper_port: int = 10200
    max_history: int = 20
    session_timeout_minutes: int = 30
    db_path: str = "/data/socrates.db"
    system_prompt: str = """You are Socrates, a wise and thoughtful voice assistant.
You engage users with curiosity and help them think deeply about their questions.
Keep responses concise (1-3 sentences) unless more detail is requested.
Be warm, insightful, and occasionally use gentle humor."""


settings = Settings()
http_client: Optional[httpx.AsyncClient] = None
db: Optional[aiosqlite.Connection] = None

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
    await db.commit()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global http_client
    http_client = httpx.AsyncClient(timeout=300.0)  # 5 minute timeout
    await init_db()
    yield
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

    if session_id and session_id in sessions:
        sessions[session_id]["last_access"] = now
        return session_id, sessions[session_id]["history"]

    # Create new session
    new_id = str(uuid.uuid4())
    sessions[new_id] = {"history": [], "last_access": now}
    return new_id, []


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


async def synthesize_speech_wyoming(text: str) -> bytes:
    """Use Wyoming protocol to synthesize speech with Piper."""
    async with AsyncTcpClient(settings.piper_host, settings.piper_port) as client:
        await client.write_event(Synthesize(text=text).event())

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

        # Step 2: Query Ollama with conversation history
        try:
            system_prompt = x_personality if x_personality else settings.system_prompt
            logger.info(f"Using personality: {system_prompt[:50]}...")
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

        # Step 3: Synthesize speech
        try:
            response_audio = await synthesize_speech_wyoming(response_text)
            logger.info(f"TTS result: {len(response_audio)} bytes")
        except Exception as e:
            logger.error(f"TTS failed: {e}\n{traceback.format_exc()}")
            raise HTTPException(status_code=500, detail=f"TTS failed: {str(e)}")

        return Response(
            content=response_audio,
            media_type="audio/wav",
            headers={
                "X-Transcript": transcript,
                "X-Response-Text": response_text[:500],
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


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}


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
