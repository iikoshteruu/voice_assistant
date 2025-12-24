import asyncio
import io
import logging
import traceback
import wave
from contextlib import asynccontextmanager
from typing import Optional

import httpx
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import Response, HTMLResponse
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
    system_prompt: str = """You are a helpful voice assistant. Keep responses concise and conversational -
aim for 1-3 sentences unless more detail is specifically requested. Be direct and helpful."""


settings = Settings()
http_client: Optional[httpx.AsyncClient] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global http_client
    http_client = httpx.AsyncClient(timeout=300.0)  # 5 minute timeout
    yield
    await http_client.aclose()


app = FastAPI(title="Socrates API", lifespan=lifespan)


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


async def query_ollama(text: str, conversation_history: list = None) -> str:
    """Send text to Ollama and get response."""
    messages = [{"role": "system", "content": settings.system_prompt}]

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
async def process_voice(audio: UploadFile = File(...)):
    """
    Main voice processing endpoint.
    Receives audio, transcribes, queries LLM, synthesizes response.
    """
    try:
        # Read uploaded audio
        audio_bytes = await audio.read()
        logger.info(f"Received audio: {len(audio_bytes)} bytes, filename: {audio.filename}")

        # Step 1: Transcribe
        try:
            transcript = await transcribe_audio(audio_bytes, audio.filename or "audio.wav")
            logger.info(f"Transcription result: {transcript}")
        except Exception as e:
            logger.error(f"Transcription failed: {e}\n{traceback.format_exc()}")
            raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

        if not transcript.strip():
            raise HTTPException(status_code=400, detail="Could not transcribe audio")

        # Step 2: Query Ollama
        try:
            response_text = await query_ollama(transcript)
            logger.info(f"Ollama response: {response_text[:200]}...")
        except Exception as e:
            logger.error(f"Ollama failed: {e}\n{traceback.format_exc()}")
            raise HTTPException(status_code=500, detail=f"Ollama failed: {str(e)}")

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
                "X-Response-Text": response_text[:500]
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


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main web UI."""
    with open("static/index.html", "r") as f:
        return f.read()
