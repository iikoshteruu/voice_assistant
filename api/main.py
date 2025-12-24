import asyncio
import io
import json
import logging
import struct
import traceback
import wave
from contextlib import asynccontextmanager
from typing import Optional

import httpx
from fastapi import FastAPI, File, UploadFile, HTTPException

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
from fastapi.responses import Response, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic_settings import BaseSettings


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
    reader, writer = await asyncio.open_connection(
        settings.piper_host, settings.piper_port
    )

    try:
        # Wyoming protocol uses newline-delimited JSON
        synthesize_event = {"type": "synthesize", "data": {"text": text}}
        writer.write(json.dumps(synthesize_event).encode() + b"\n")
        await writer.drain()

        audio_chunks = []
        audio_info = None

        while True:
            line = await reader.readline()
            if not line:
                break

            event = json.loads(line.decode())
            event_type = event.get("type", "")

            if event_type == "audio-start":
                audio_info = event.get("data", {})
            elif event_type == "audio-chunk":
                # Binary payload follows the JSON line
                payload_length = event.get("data", {}).get("audio", {}).get("length", 0)
                if payload_length > 0:
                    audio_data = await reader.readexactly(payload_length)
                    audio_chunks.append(audio_data)
            elif event_type == "audio-stop":
                break

        if audio_chunks and audio_info:
            raw_audio = b"".join(audio_chunks)
            return _create_wav(
                raw_audio,
                sample_rate=audio_info.get("rate", 22050),
                sample_width=audio_info.get("width", 2),
                channels=audio_info.get("channels", 1)
            )

        return b""
    finally:
        writer.close()
        await writer.wait_closed()


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
