#!/usr/bin/env python3
"""
FastAPI приложение для стриминга видео через WebRTC.
Использует aiortc для создания медиа потока из генерируемых кадров.
Архитектура: Services (ConnectionManager, StreamService, InferenceService).
"""
import os
import sys
import json
import asyncio
import threading
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
import subprocess
import tempfile
import yaml
import time
import hashlib
import requests
import urllib.parse
from typing import Optional, Dict
from pydub import AudioSegment

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from aiortc import RTCSessionDescription
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Настройка путей
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Импорт сервисов
from services.stream_service import StreamService
from services.connection_manager import ConnectionManager
from services.inference_service import InferenceService

load_dotenv()  # Загружает переменные из .env файла

# Настройка логирования
from logging.handlers import RotatingFileHandler
import traceback

def setup_debug_logging():
    log_formatter = logging.Formatter('%(asctime)s [%(levelname)s] [%(name)s] %(message)s')
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    root_logger.addHandler(console_handler)
    
    # File handler
    os.makedirs("logs", exist_ok=True)
    file_handler = RotatingFileHandler("logs/api_debug.log", maxBytes=10*1024*1024, backupCount=5)
    file_handler.setFormatter(log_formatter)
    root_logger.addHandler(file_handler)
    
    # Capture unhandled exceptions
    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        root_logger.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

    sys.excepthook = handle_exception

setup_debug_logging()
logger = logging.getLogger("StreamingAPI")

async def heartbeat_watchdog():
    """Фоновая задача для мониторинга здоровья Event Loop"""
    logger.info("Heartbeat watchdog started")
    while True:
        try:
            # Если этот лог не появляется каждые 5 сек, значит loop заблокирован
            active_tasks = len(asyncio.all_tasks())
            logger.info(f"Heartbeat: Alive. Active tasks: {active_tasks}")
            await asyncio.sleep(5)
        except asyncio.CancelledError:
            logger.info("Heartbeat watchdog stopped")
            break
        except Exception as e:
            logger.error(f"Heartbeat error: {e}")
            await asyncio.sleep(5)


# Конфигурация приложения
class AppConfig(BaseModel):
    # Основные настройки
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", 8009))
    musetalk_version: str = os.getenv("MUSETALK_VERSION", "v1.5")
    musetalk_mode: str = os.getenv("MUSETALK_MODE", "realtime")
    gpu_id: int = int(os.getenv("MUSETALK_GPU_ID", "0"))
    use_float16: bool = os.getenv("MUSETALK_USE_FLOAT16", "true").lower() == "true"
    
    # TURN/STUN настройки
    turn_server_url: Optional[str] = os.getenv("TURN_SERVER_URL")
    turn_server_username: Optional[str] = os.getenv("TURN_SERVER_USERNAME", "")
    turn_server_credential: Optional[str] = os.getenv("TURN_SERVER_CREDENTIAL", "")
    use_public_turn: bool = os.getenv("USE_PUBLIC_TURN", "false").lower() == "true"
    
    # Пути
    audio_dir: str = "data/audio"
    requests_dir: str = "./requests"
    
    # Custom
    greeting_delay: float = float(os.getenv("GREETING_DELAY", "0"))

config = AppConfig()

class WebRTCOfferRequest(BaseModel):
    sdp: str
    type: str
    video_path: str = "data/video/avatar_1.mp4"
    fps: int = 25
    batch_size: int = 4
    audio_path: Optional[str] = None
    task_id: Optional[str] = None
    version: Optional[str] = None
    video_hello_text: Optional[str] = None
    settings_id: Optional[str] = None
    audio_only: bool = False

class StopGenerationRequest(BaseModel):
    task_id: str

class RestartGenerationRequest(BaseModel):
    task_id: str

# Инициализация сервисов
stream_service = StreamService()
connection_manager = ConnectionManager(config)
inference_service = InferenceService(config)

try:
    from aiortc import RTCPeerConnection, RTCSessionDescription
    WEBRTC_AVAILABLE = True
except ImportError:
    WEBRTC_AVAILABLE = False
    logger.error("aiortc не установлен. Установите: pip install aiortc")

app = FastAPI(title="MuseTalk Streaming API")

# Создаем необходимые директории
os.makedirs(config.audio_dir, exist_ok=True)
os.makedirs(config.requests_dir, exist_ok=True)

# CORS config
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# --- Утилиты ---

def check_daemon_service(version="v1.5", mode="realtime"):
    """Проверяет, запущен ли демон-сервис через PID файл"""
    if version == "v15":
        version = "v1.5"
    
    # Исправлено: PID файл лежит в папке logs и называется model_service_...
    pid_file = os.path.join("logs", f"model_service_{version}_{mode}.pid")
    
    if not os.path.exists(pid_file):
        return False, f"PID файл {pid_file} не найден"
        
    try:
        with open(pid_file, "r") as f:
            pid = int(f.read().strip())
        
        # Проверка существования процесса
        if os.name == 'nt': # Windows
            # Простейший способ проверить существование процесса в Windows
            try:
                # OpenProcess возвращает дескриптор, если процесс существует и есть права
                import ctypes
                PROCESS_QUERY_INFORMATION = 0x0400
                PROCESS_VM_READ = 0x0010
                handle = ctypes.windll.kernel32.OpenProcess(PROCESS_QUERY_INFORMATION | PROCESS_VM_READ, False, pid)
                if handle:
                    ctypes.windll.kernel32.CloseHandle(handle)
                else:
                    return False, f"Процесс с PID {pid} не найден (OpenProcess failed)"
            except:
                 # Fallback method parsing tasklist output
                 cmd = f"tasklist /FI \"PID eq {pid}\""
                 output = os.popen(cmd).read()
                 if str(pid) not in output:
                     return False, f"Процесс с PID {pid} не найден в tasklist"
        else: # Linux/Mac
            if not os.path.exists(f"/proc/{pid}"):
                 return False, f"Директория /proc/{pid} не существует"
            else:
                result = subprocess.run(
                    ["ps", "-p", str(pid)],
                    capture_output=True
                )
                # В Linux ps возвращает 0 если процесс есть, но нужно проверить вывод
                if result.returncode != 0:
                     return False, f"Процесс с PID {pid} не найден"
        
        return True, pid
    except (ValueError, IOError) as e:
        return False, f"Ошибка при чтении PID файла: {e}"

# --- Welcome Video Logic ---

def get_welcome_video_hash(video_path: str, text: str, settings_id: str):
    """Generates a consistent hash for caching based on video path, text, and settings_id."""
    # Normalize paths and text to ensure consistency
    video_path_norm = os.path.normpath(video_path).lower()
    text_norm = text.strip()
    settings_id_norm = settings_id.strip()
    data = f"{video_path_norm}+{text_norm}+{settings_id_norm}"
    return hashlib.md5(data.encode('utf-8')).hexdigest()

async def ensure_welcome_audio(text: str, hash_str: str, audio_dir: str, settings_id: str):
        """Checks for existing audio or fetches from TTS."""
        welcome_dir = os.path.join(audio_dir, "welcome")
        os.makedirs(welcome_dir, exist_ok=True)

        audio_filename = f"welcome_{hash_str}.wav"
        audio_path = os.path.join(welcome_dir, audio_filename)
        
        if os.path.exists(audio_path):
            logger.info(f"Found cached welcome audio: {audio_path}")
            return audio_path
            
        logger.info(f"Generating welcome audio via TTS for: '{text}' (Settings ID: {settings_id})")
        
        tts_token = os.getenv("TTS_AUTH_TOKEN")
        tts_url = os.getenv("TTS_API_URL")
        
        url = f"{tts_url}?settings_id={settings_id}"
        
        headers = {
            'accept': 'audio/wav',
            'Content-Type': 'application/json'
        }
        
        if tts_token:
            headers['Authorization'] = f"Bearer {tts_token}"
            
        data = {
            "text": text
        }
        
        try:
            # Synchronous request in thread executor to avoid blocking loop
            loop = asyncio.get_event_loop()
            def fetch_tts():
                return requests.post(url, headers=headers, json=data, timeout=30)
            
            response = await loop.run_in_executor(None, fetch_tts)
            
            if response.status_code == 200:
                with open(audio_path, "wb") as f:
                    f.write(response.content)
                
                # Append 2 seconds of silence
                try:
                    audio_segment = AudioSegment.from_wav(audio_path)
                    silence = AudioSegment.silent(duration=0)
                    padded_audio = audio_segment + silence
                    padded_audio.export(audio_path, format="wav")
                    logger.info(f"Appended 2 seconds of silence to: {audio_path}")
                except Exception as pad_err:
                    logger.error(f"Failed to pad TTS audio: {pad_err}")
                    
                logger.info(f"Welcome audio saved to: {audio_path}")
                return audio_path
            else:
                logger.error(f"TTS API Error: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            logger.error(f"TTS Request failed: {e}")
            return None


# --- Endpoints ---

@app.post("/api/webrtc/offer")
async def webrtc_offer(
    request: WebRTCOfferRequest
):
    """
    Создает WebRTC соединение на основе SDP offer.
    """
    if not WEBRTC_AVAILABLE:
        raise HTTPException(status_code=500, detail="aiortc library not installed")

    try:
        sdp = request.sdp
        type = request.type
        # Use provided task_id or generate one
        task_id = request.task_id if request.task_id else str(int(time.time() * 1000))
        video_path = request.video_path
        
        if video_path.startswith("http://") or video_path.startswith("https://"):
            parsed_url = urllib.parse.urlparse(video_path)
            filename = os.path.basename(parsed_url.path)
            if not filename:
                filename = hashlib.md5(video_path.encode('utf-8')).hexdigest() + ".mp4"
                
            local_path = os.path.join("data", "video", filename)
            
            if not os.path.exists(local_path):
                logger.info(f"Downloading video from {video_path} to {local_path}...")
                loop = asyncio.get_event_loop()
                def fetch_video():
                    response = requests.get(video_path, stream=True, timeout=120)
                    response.raise_for_status()
                    os.makedirs(os.path.dirname(local_path), exist_ok=True)
                    with open(local_path, "wb") as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                try:
                    await loop.run_in_executor(None, fetch_video)
                    logger.info(f"Successfully downloaded video to {local_path}")
                except Exception as e:
                    logger.error(f"Failed to download video from {video_path}: {e}")
                    raise HTTPException(status_code=400, detail=f"Failed to download video from URL: {e}")
            else:
                logger.info(f"Video already exists at {local_path}, skipping download.")
                
            video_path = local_path

        fps = request.fps
        batch_size = request.batch_size
        
        logger.info(f"Получен WebRTC Offer. Task ID: {task_id}, Video: {video_path}")
        
        if not os.path.exists(video_path):
            logger.error(f"Video path not found: {video_path}")
            raise HTTPException(status_code=400, detail=f"Видео файл или директория не найдены: {video_path}")
            
        if os.path.isfile(video_path) and os.path.getsize(video_path) == 0:
            logger.error(f"Video file is empty: {video_path}")
            raise HTTPException(status_code=400, detail=f"Видео файл пуст: {video_path}")
            
        if os.path.isdir(video_path):
            img_exts = {'.png', '.jpg', '.jpeg'}
            has_images = any(os.path.splitext(f)[1].lower() in img_exts for f in os.listdir(video_path))
            if not has_images:
                logger.error(f"Directory contains no images: {video_path}")
                raise HTTPException(status_code=400, detail=f"В директории нет изображений: {video_path}")
            
        offer_desc = RTCSessionDescription(sdp=sdp, type=type)
        
        # Создаем PC через менеджер
        logger.info(f"Creating connection for {task_id}...")
        try:
            pc = await connection_manager.create_connection(task_id)
        except Exception as e:
            logger.error(f"Failed to create connection: {e}", exc_info=True)
            raise e
        logger.info(f"Connection created: {pc}")
        
        # Создаем медиа треки через StreamService
        # Создаем медиа треки через StreamService
        
        video_track = None
        loop = asyncio.get_event_loop()
        if not request.audio_only:
            try:
                logger.info(f"Start creating video track for {video_path}...")
                start_time = time.time()
                video_track = stream_service.create_video_track(
                    fps=fps, 
                    loop=loop,
                    # video_width/height will be auto-detected from video_path
                    video_path=video_path
                )
                logger.info(f"Video track created in {time.time() - start_time:.3f}s")
                pc.addTrack(video_track)
            except Exception as e:
                logger.error(f"Ошибка при создании видео трека: {e}", exc_info=True)
                await connection_manager.close_connection(task_id)
                raise HTTPException(status_code=500, detail=f"Failed to create video track: {e}")
        else:
            logger.info("Audio only mode requested. Skipping video track creation.")
        
        # Аудио трек для воспроизведения ответов
        try:
            from aiortc.contrib.media import MediaPlayer
            # Пустой media player изначально, он будет заменен при генерации
            # В оригинале использовался костыль с DelayedMediaPlayerTrack
            logger.info("Creating audio track...")
            audio_track = stream_service.create_audio_track(None, video_track)
            logger.info(f"Audio track created: {audio_track}")
            sender = pc.addTrack(audio_track)
            logger.info(f"Audio track added to PC: {sender}")
        except Exception as e:
            logger.error(f"Failed to create/add audio track: {e}", exc_info=True)
            raise e
        
        # --- Welcome Video Logic ---
        welcome_text = request.video_hello_text if request.video_hello_text else os.getenv("VIDEO_HELLO_TEXT")
        # Use settings_id from request if available, otherwise fallback to env
        settings_id = request.settings_id if request.settings_id else os.getenv("TTS_SETTINGS_ID", "1765145591841-i6lqzzwui")
        
        # Check if this is a "welcome" scenario (no task_id provided or specific flag? User said "when connecting")
        # Assuming every new connection is a candidate for welcome video if env is set.
        
        did_schedule_welcome = False
        
        if welcome_text:
            try:
                # Calculate hash
                welcome_hash = get_welcome_video_hash(video_path, welcome_text, settings_id)
                
                # Check for cached VIDEO using the hash (and video ext from video_path)
                video_ext = os.path.splitext(video_path)[1]
                cached_video_name = f"cached_{welcome_hash}.mp4" # Force mp4 for output usually
                cached_video_path = os.path.join("data/video/cached", cached_video_name)
                os.makedirs("data/video/cached", exist_ok=True) # Ensure dir
                
                # Check Audio
                welcome_audio_path = await ensure_welcome_audio(welcome_text, welcome_hash, config.audio_dir, settings_id)
                
                if welcome_audio_path:
                    if video_track:
                        if os.path.exists(cached_video_path) and os.path.getsize(cached_video_path) > 0:
                            logger.info(f"HIT: Playing cached welcome video: {cached_video_path}")
                            
                            # Play VIDEO from file
                            # video_track is an instance of VideoStreamGenerator which now has play_video_file
                            async def delayed_play(path, fps, delay):
                                 if delay > 0:
                                     logger.info(f"Waiting for greeting delay: {delay}s")
                                     await asyncio.sleep(delay)
                                 
                                 # Clear queue to remove idle frames that accumulated during delay
                                 # This ensures the video starts immediately with the audio
                                 if video_track:
                                     q_size = video_track.frame_queue.qsize()
                                     if q_size > 0:
                                         logger.info(f"Clearing {q_size} idle frames from queue before playing welcome video")
                                         while not video_track.frame_queue.empty():
                                             try:
                                                 video_track.frame_queue.get_nowait()
                                             except:
                                                 break
                                                 
                                 await video_track.play_video_file(path, fps=fps)

                            asyncio.create_task(delayed_play(cached_video_path, fps, config.greeting_delay))
                            
                            # Play AUDIO
                            # reset_audio_track with new path plays it
                            stream_service.reset_audio_track(audio_track, new_audio_path=welcome_audio_path)
                            
                            did_schedule_welcome = True
                            
                        else:
                            logger.info(f"MISS: Generating welcome video to cache: {cached_video_path}")
                            
                            # Start Generation with CAPTURE
                            # using task_id (we need a unique task id for generation process)
                            # But we also want to cache it.
                            
                            # Play AUDIO
                            stream_service.reset_audio_track(audio_track, new_audio_path=welcome_audio_path)
                            
                            await inference_service.start_generation(
                                task_id=task_id,
                                video_track=video_track,
                                audio_track=audio_track,
                                video_path=video_path,
                                audio_path=welcome_audio_path, # TTS audio
                                fps=fps,
                                batch_size=batch_size,
                                version=config.musetalk_version,
                                has_active_generation=False,
                                capture_video_path=cached_video_path, # Capture for next time
                                initial_delay=config.greeting_delay
                            )
                            did_schedule_welcome = True
                    else:
                        # Audio Only Logic
                        logger.info("Audio only mode: Playing welcome audio.")
                        
                        async def delayed_audio(path, delay):
                             if delay > 0:
                                 logger.info(f"Waiting for greeting delay: {delay}s")
                                 await asyncio.sleep(delay)
                             stream_service.reset_audio_track(audio_track, new_audio_path=path)
                        
                        asyncio.create_task(delayed_audio(welcome_audio_path, config.greeting_delay))
                        did_schedule_welcome = True
                else:
                    logger.warning("Could not get welcome audio, skipping welcome video.")
                    
            except Exception as e:
                logger.error(f"Error checking/starting welcome video: {e}", exc_info=True)
        
        # ---------------------------

        
        # Сохраняем информацию о треках
        connection_manager.update_connection_info(task_id, {
            "video_track": video_track,
            "audio_track": audio_track,
            "video_path": video_path,
            "fps": fps,
            "batch_size": batch_size
        })

        @pc.on("iceconnectionstatechange")
        async def on_ice_connection_state_change():
            logger.info(f"ICE connection state is {pc.iceConnectionState} for task {task_id}")
            if pc.iceConnectionState in ["failed", "closed"]:
                await cleanup_resources(task_id)
        
        @pc.on("connectionstatechange")
        async def on_connection_state_change():
            logger.info(f"Connection state is {pc.connectionState} for task {task_id}")
            if pc.connectionState in ["failed", "closed"]:
                await cleanup_resources(task_id)

        try:
            await pc.setRemoteDescription(offer_desc)
            logger.info("Remote description set")
            
            answer = await pc.createAnswer()
            logger.info("Answer created")
            
            await pc.setLocalDescription(answer)
            logger.info("Local description set")
        except Exception as e:
            logger.error(f"Error during SDP negotiation: {e}", exc_info=True)
            raise e
        
        # Ожидание сбора кандидатов (в aiortc это происходит автоматически после setLocalDescription,
        # но нужно дать немного времени или проверить состояние, если мы хотим отправить полный SDP)
        # В aiortc для trickle ICE candidates не нужно ждать, но если клиент не поддерживает trickle
        # или мы хотим максимальную совместимость, лучше подождать.
        # Однако aiortc по умолчанию собирает кандидатов. 
        # Проверим, есть ли кандидаты в SDP.
        
        # Простой способ подождать завершения сбора (или таймаута)
        # В aiortc нет явного события 'icegatheringstatechange' которое можно await так просто как в JS,
        # но мы можем подождать немного.
        
        # Более надежный способ - проверить iceGatheringState
        # Но в текущей версии aiortc может не быть удобного способа ждать. 
        # Просто дадим 1-2 секунды на сбор, если это публичные айпи.
        
        logger.info(f"Waiting for ICE gathering to complete for task {task_id}...")
        # Ждем пока состояние станет complete или пройдет время
        wait_time = 0
        while pc.iceGatheringState != "complete" and wait_time < 2.0:
            await asyncio.sleep(0.1)
            wait_time += 0.1
            
        logger.info(f"ICE gathering finished (state={pc.iceGatheringState}) in {wait_time:.1f}s")
        
        response = {
            "sdp": pc.localDescription.sdp,
            "type": pc.localDescription.type,
            "task_id": task_id
        }
        return response
        
    except Exception as e:
        logger.error(f"Ошибка в WebRTC offer: {e}", exc_info=True)
        # Если что-то пошло не так, пытаемся почистить
        if 'task_id' in locals():
            await cleanup_resources(task_id)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/webrtc/restart_generation")
async def restart_generation(
    request: RestartGenerationRequest
):
    """
    Перезапускает генерацию видео для существующей сессии.
    """
    task_id = request.task_id
    logger.info(f"Запрос на перезапуск генерации для task_id={task_id}")
    
    # Получаем информацию о соединении
    conn_info = connection_manager.get_connection_info(task_id)
    if not conn_info:
        raise HTTPException(status_code=404, detail="Task not found or connection closed")
    
    pc = connection_manager.get_connection(task_id)
    if not pc:
        # Может быть соединение закрыто, но info осталось?
        raise HTTPException(status_code=404, detail="WebRTC connection not found")

    # Останавливаем текущую генерацию, если есть
    await inference_service.stop_generation(task_id)
    
    # Очищаем очередь видео трека
    video_track = conn_info.get("video_track")
    if video_track:
        while not video_track.frame_queue.empty():
            try:
                video_track.frame_queue.get_nowait()
            except:
                break
        
        # Сброс флагов синхронизации
        video_track.running = True
        with video_track.first_real_frame_lock:
            video_track.first_real_frame_sent = False
        with video_track.generation_completed_lock:
            video_track.generation_completed = False
        with video_track.content_frames_lock:
            video_track.content_frames_sent = 0


    # Сбрасываем аудио
    audio_track = conn_info.get("audio_track")
    stream_service.reset_audio_track(audio_track)
    
    # Получаем параметры для запуска
    video_path = conn_info.get("video_path")
    fps = conn_info.get("fps", 25)
    batch_size = conn_info.get("batch_size", 4)
    # Используем какой-то дефолтный аудио файл для restart? 
    # В оригинале использовался audio_path из info, но если restart вызывается без нового аудио,
    # то что мы генерируем? Original logic used "data/audio/sun.wav" as fallback or reused `audio_path`.
    audio_path = conn_info.get("audio_path", "data/audio/sun.wav")
    
    # Запуск новой генерации
    # Запуск новой генерации
    if video_track:
        await inference_service.start_generation(
            task_id=task_id,
            video_track=video_track,
            audio_track=audio_track,
            video_path=video_path,
            audio_path=audio_path,
            fps=fps,
            batch_size=batch_size,
            version=config.musetalk_version,
            has_active_generation=False # Это новый запуск после стопа
        )
    else:
        logger.info(f"Audio only restart for task_id={task_id}, skipping video generation.")
    
    return {"status": "ok", "message": "Generation restarted", "task_id": task_id}

@app.post("/api/webrtc/stop")
async def stop_generation_endpoint(
    request: StopGenerationRequest
):
    """
    Останавливает текущую генерацию, очищает очереди и сбрасывает состояние,
    сохраняя WebRTC соединение активным.
    """
    task_id = request.task_id
    logger.info(f"Запрос на остановку генерации для task_id={task_id}")
    
    # Получаем информацию о соединении
    conn_info = connection_manager.get_connection_info(task_id)
    if not conn_info:
        raise HTTPException(status_code=404, detail="Task not found or connection closed")
    
    # Останавливаем генерацию в сервисе (отправляет STOP в Redis)
    await inference_service.stop_generation(task_id)
    
    # Очищаем очередь генерации в Redis (опционально, если нужно убрать ожидающие задачи)
    # Сейчас inference_service.stop_generation отправляет STOP команду в конец очереди.
    # Если мы хотим очистить очередь задач для этого task_id, это сложнее сделать атомарно в Redis без Lua скрипта,
    # но отправка STOP обычно достаточна, чтобы демон пропустил или прервал текущую.
    
    # Очищаем локальные очереди треков
    video_track = conn_info.get("video_track")
    if video_track:
        # Очистка очереди кадров
        while not video_track.frame_queue.empty():
            try:
                video_track.frame_queue.get_nowait()
            except:
                break
        
        # Сброс флагов синхронизации для возврата в idle
        video_track.running = True # Должен быть True чтобы крутился idle
        with video_track.first_real_frame_lock:
            video_track.first_real_frame_sent = False
        with video_track.generation_completed_lock:
            video_track.generation_completed = False
        with video_track.content_frames_lock:
            video_track.content_frames_sent = 0

    # Сбрасываем аудио (останавливаем воспроизведение)
    audio_track = conn_info.get("audio_track")
    if audio_track:
        stream_service.reset_audio_track(audio_track)
        
    return {"status": "ok", "message": "Generation stopped", "task_id": task_id}

@app.post("/api/webrtc/upload_audio_and_generate")
async def upload_audio_and_generate(
    task_id: str = Form(...),
    audio_file: UploadFile = File(...),
):
    """
    Загружает аудио файл и запускает генерацию видео.
    """
    logger.info(f"Загрузка аудио и старт генерации для task_id={task_id}")
    
    # Проверки соединения
    pc = connection_manager.get_connection(task_id)
    conn_info = connection_manager.get_connection_info(task_id)
    
    if not pc or not conn_info:
        raise HTTPException(status_code=404, detail="WebRTC connection not found")
        
    ice_state = pc.iceConnectionState
    conn_state = pc.connectionState
    
    if ice_state not in ["connected", "checking"] and conn_state not in ["connected", "connecting"]:
        raise HTTPException(
            status_code=400, 
            detail=f"WebRTC соединение не активно (ICE: {ice_state}, Connection: {conn_state})"
        )
    
    # Сохранение файла
    # Сохранение файла с высокой точностью времени во избежание коллизий
    timestamp = int(time.time() * 1000000) # Microseconds
    file_extension = os.path.splitext(audio_file.filename)[1] if audio_file.filename else ".wav"
    audio_filename = f"{task_id}_{timestamp}{file_extension}"
    generated_dir = os.path.join(config.audio_dir, "generated")
    os.makedirs(generated_dir, exist_ok=True)
    new_audio_path = os.path.join(generated_dir, audio_filename)
    
    try:
        content = await audio_file.read()
        with open(new_audio_path, "wb") as f:
            f.write(content)
            
        # Append 2 seconds of silence
        try:
            audio_segment = AudioSegment.from_file(new_audio_path)
            silence = AudioSegment.silent(duration=0)
            padded_audio = audio_segment + silence
            
            # Export maintaining original extension if possible or default to wav
            export_format = file_extension.lstrip('.') if file_extension else 'wav'
            # pydub handles m4a/mp3 using ffmpeg
            padded_audio.export(new_audio_path, format=export_format)
            logger.info(f"Appended 2 seconds of silence to uploaded audio: {new_audio_path}")
        except Exception as pad_err:
            logger.error(f"Failed to pad uploaded audio: {pad_err}")

    except Exception as e:
        logger.error(f"Ошибка при сохранении аудио файла: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to save audio file")
    
    # Останавливаем предыдущую генерацию
    has_active = await inference_service.is_generation_active(task_id)
    await inference_service.stop_generation(task_id)
    
    # Подготовка треков
    video_track = conn_info.get("video_track")
    if video_track:
        while not video_track.frame_queue.empty():
            try:
                video_track.frame_queue.get_nowait()
            except:
                break
        
        video_track.running = True
        with video_track.first_real_frame_lock:
            video_track.first_real_frame_sent = False
        with video_track.generation_completed_lock:
            video_track.generation_completed = False
        with video_track.content_frames_lock:
            video_track.content_frames_sent = 0

            
    audio_track = conn_info.get("audio_track")
    stream_service.reset_audio_track(audio_track, new_audio_path=new_audio_path)
    
    # Обновление инфо (для удаления старого файла)
    # Удаляем старый файл СНАЧАЛА (если есть), потом обновляем путь
    connection_manager.remove_audio_file(task_id) # Удаляет тот, что был записан в info
    
    connection_manager.update_connection_info(task_id, {
        "audio_path": new_audio_path,
        "uploaded_audio": True
    })
    
    # Параметры из info
    video_path = conn_info.get("video_path")
    fps = conn_info.get("fps", 25)
    batch_size = conn_info.get("batch_size", 4)
    
    # Запуск генерации
    # Запуск генерации
    if video_track:
        await inference_service.start_generation(
            task_id=task_id,
            video_track=video_track,
            audio_track=audio_track,
            video_path=video_path,
            audio_path=new_audio_path,
            fps=fps,
            batch_size=batch_size,
            version=config.musetalk_version,
            has_active_generation=has_active
        )
    else:
        logger.info(f"Audio only generation request for task_id={task_id}, skipping video generation.")
    


@app.get("/api/files/avatars")
async def list_avatars():
    """Список доступных видео-аватаров"""
    files = []
    video_dir = "data/video"
    results_avatars_dir = "results/v15/avatars"
    
    valid_avatars = set()
    if os.path.exists(results_avatars_dir):
        valid_avatars = set(os.listdir(results_avatars_dir))

    if os.path.exists(video_dir):
        for f in os.listdir(video_dir):
            if f.lower().endswith(('.mp4', '.avi', '.mov')):
                # Check if corresponding folder exists in results/v15/avatars
                name_without_ext = os.path.splitext(f)[0]
                if name_without_ext in valid_avatars:
                    files.append(f)
    return {"files": sorted(files)}

@app.get("/api/files/audio")
async def list_audio():
    """Список доступных аудио файлов"""
    files = []
    audio_dir = "data/audio"
    if os.path.exists(audio_dir):
        for f in os.listdir(audio_dir):
            if f.lower().endswith(('.wav', '.mp3', '.m4a')):
                files.append(f)
    return {"files": sorted(files)}

class GenerationRequest(BaseModel):
    video_path: str
    audio_path: str
    bbox_shift: int = 0
    extra_margin: int = 10
    parsing_mode: str = "jaw"
    left_cheek_width: int = 90
    right_cheek_width: int = 90
    fps: int = 25
    batch_size: int = 8

@app.post("/api/generate_offline")
async def generate_offline(req: GenerationRequest):
    """Запуск оффлайн генерации с параметрами"""
    task_id = f"gen_{int(time.time()*1000)}"
    
    # Запуск генерации через сервис
    # Используем start_generation но с доп параметрами
    # inference_service нужно обновить чтобы принимать **kwargs или specific params
    await inference_service.start_generation(
        task_id=task_id,
        video_track=None, # Нет треков для оффлайн
        audio_track=None,
        video_path=os.path.join("data/video", req.video_path),
        audio_path=os.path.join("data/audio", req.audio_path),
        fps=req.fps,
        batch_size=req.batch_size,
        version=config.musetalk_version,
        # Передаем доп параметры через kwargs если изменим сервис, или пока просто как props
        # FIXME: Update InferenceService signature
        additional_params={
            "bbox_shift": req.bbox_shift,
            "extra_margin": req.extra_margin,
            "parsing_mode": req.parsing_mode,
            "left_cheek_width": req.left_cheek_width,
            "right_cheek_width": req.right_cheek_width,
            "use_preprocessed": False # Force raw mode for testing parameters
        }
    )
    
    return {"task_id": task_id, "status": "started"}

app.mount("/results", StaticFiles(directory="results"), name="results")

@app.get("/api/result/{task_id}")
async def get_result_status(task_id: str):
    """Проверка статуса генерации"""
    # Check multiple possible locations
    possible_paths = [
        f"results/v1.5/{task_id}.mp4",
        f"results/v15/{task_id}.mp4",
        f"results/realtime/v15/{task_id}.mp4",
        f"results/realtime/v1.5/{task_id}.mp4"
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            # URL is relative to the mounted /results endpoint
            # path is like results/subdir/file.mp4 -> url /results/subdir/file.mp4
            # We assume app.mount("/results", ...) is serving the "results" folder
            relative_path = os.path.relpath(path, "results").replace("\\", "/")
            return {"status": "completed", "url": f"/results/{relative_path}"}
    
    # Check if active
    is_active = await inference_service.is_generation_active(task_id)
    if is_active:
        return {"status": "processing"}
        
    return {"status": "not_found_or_failed"}

async def cleanup_resources(task_id):
    """Очистка ресурсов"""
    logger.info(f"Очистка ресурсов для task_id={task_id}")
    
    # Стоп генерация
    await inference_service.stop_generation(task_id)
    
    # Удаление аудио
    connection_manager.remove_audio_file(task_id)
    
    # Получаем треки чтобы остановить их
    conn_info = connection_manager.get_connection_info(task_id)
    if conn_info:
        vt = conn_info.get("video_track")
        if vt and hasattr(vt, 'stop'):
            vt.stop()
        at = conn_info.get("audio_track")
        if at and hasattr(at, 'stop'):
            at.stop()
            
    # Закрытие соединения
    await connection_manager.close_connection(task_id)

@app.on_event("startup")
async def startup_event():
    logger.info(f"Запуск MuseTalk Streaming API...")
    logger.info(f"Конфигурация: Host={config.host}, Port={config.port}, Mode={config.musetalk_mode}")
    # Start heartbeat
    asyncio.create_task(heartbeat_watchdog())

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Остановка сервера...")
    # Останавливаем все генерации
    active_count = await inference_service.get_active_count()
    if active_count > 0:
        logger.info(f"Остановка {active_count} активных генераций...")
        # Логика cleanup_all должна быть в сервисе?
        # В inference_service нет cleanup_all, но connection_manager.cleanup_all закроет соединения
        # А треды генерации надо стопнуть.
        pass # ToDo: Add cleanup_all to inference service if needed, logic relies on task_id knowledge
    
    await connection_manager.cleanup_all()
    logger.info("Сервер остановлен")

@app.get("/health")
async def health_check():
    """Проверка здоровья сервиса"""
    # Active connections count is hard to get efficiently without exposing internals or adding method
    # Let's add methods to manager/service
    # For now, just return ok
    return {
        "status": "ok",
        "webrtc_available": WEBRTC_AVAILABLE,
        "active_generations": await inference_service.get_active_count()
    }

if __name__ == "__main__":
    import argparse
    
    default_version = config.musetalk_version
    default_mode = config.musetalk_mode
    default_gpu_id = config.gpu_id
    
    parser = argparse.ArgumentParser(description="WebRTC Streaming API для MuseTalk")
    parser.add_argument("--host", type=str, default=config.host, help="Хост для сервера")
    parser.add_argument("--port", type=int, default=config.port, help="Порт для сервера")
    parser.add_argument("--version", type=str, default=default_version, help="Версия модели (v1.0 или v1.5)")
    parser.add_argument("--mode", type=str, default=default_mode, help="Режим работы (normal или realtime)")
    parser.add_argument("--gpu_id", type=int, default=default_gpu_id, help="ID GPU")
    parser.add_argument("--use_float16", action="store_true", help="Использовать float16")
    
    args = parser.parse_args()
    
    if config.use_float16:
         args.use_float16 = True
    
    if not WEBRTC_AVAILABLE:
        logger.error("ОШИБКА: aiortc не установлен!")
        sys.exit(1)
    
    version_for_check = args.version
    if version_for_check == "v15":
        version_for_check = "v1.5"
    
    logger.info(f"Проверка демон-сервиса (версия: {version_for_check}, режим: {args.mode})...")
    is_running, message = check_daemon_service(version=version_for_check, mode=args.mode)
    
    if not is_running:
        logger.error("=" * 50)
        logger.error("ОШИБКА: Демон-сервис не запущен!")
        logger.error("=" * 50)
        logger.error(f"Причина: {message}")
        logger.error("")
        logger.error("Запустите демон-сервис перед запуском API:")
        logger.error(f"  sh start_model_service.sh {version_for_check} {args.mode} {args.gpu_id}")
        logger.error("")
        sys.exit(1)
    
    logger.info(f"✓ Демон-сервис запущен (PID: {message})")
    
    logger.info(f"Запуск WebRTC сервера на {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)