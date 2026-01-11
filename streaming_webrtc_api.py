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
from typing import Optional, Dict

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
        audio_filename = f"welcome_{hash_str}.wav"
        audio_path = os.path.join(audio_dir, audio_filename)
        
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
        fps = request.fps
        batch_size = request.batch_size
        
        logger.info(f"Получен WebRTC Offer. Task ID: {task_id}, Video: {video_path}")
        
        offer_desc = RTCSessionDescription(sdp=sdp, type=type)
        
        # Создаем PC через менеджер
        pc = await connection_manager.create_connection(task_id)
        
        # Создаем медиа треки через StreamService
        # Для начала только видео, аудио добавим если есть файл воспроизведения (здесь пока нет)
        loop = asyncio.get_event_loop()
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
        except Exception as e:
            logger.error(f"Ошибка при создании видео трека: {e}", exc_info=True)
            await connection_manager.close_connection(task_id)
            raise HTTPException(status_code=500, detail=f"Failed to create video track: {e}")
            
        pc.addTrack(video_track)
        
        # Аудио трек для воспроизведения ответов
        from aiortc.contrib.media import MediaPlayer
        # Пустой media player изначально, он будет заменен при генерации
        # В оригинале использовался костыль с DelayedMediaPlayerTrack
        audio_track = stream_service.create_audio_track(None, video_track)
        pc.addTrack(audio_track)
        
        # --- Welcome Video Logic ---
        welcome_text = os.getenv("VIDEO_HELLO_TEXT")
        settings_id = os.getenv("TTS_SETTINGS_ID", "1765145591841-i6lqzzwui")
        
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
                cached_video_path = os.path.join("data/video", cached_video_name)
                os.makedirs("data/video", exist_ok=True) # Ensure dir
                
                # Check Audio
                welcome_audio_path = await ensure_welcome_audio(welcome_text, welcome_hash, config.audio_dir, settings_id)
                
                if welcome_audio_path:
                    if os.path.exists(cached_video_path) and os.path.getsize(cached_video_path) > 0:
                        logger.info(f"HIT: Playing cached welcome video: {cached_video_path}")
                        
                        # Play VIDEO from file
                        # video_track is an instance of VideoStreamGenerator which now has play_video_file
                        asyncio.create_task(video_track.play_video_file(cached_video_path, fps=fps))
                        
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
                            capture_video_path=cached_video_path # Capture for next time
                        )
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

        await pc.setRemoteDescription(offer_desc)
        
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
        
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
    task_id: str = Form(...),
):
    """
    Перезапускает генерацию видео для существующей сессии.
    """
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
    
    return {"status": "ok", "message": "Generation restarted", "task_id": task_id}

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
    new_audio_path = os.path.join(config.audio_dir, audio_filename)
    
    try:
        content = await audio_file.read()
        with open(new_audio_path, "wb") as f:
            f.write(content)
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
    
    return {
        "status": "ok",
        "message": f"Аудио файл загружен и генерация для task_id={task_id} запущена",
        "task_id": task_id,
        "audio_path": new_audio_path,
        "audio_size": len(content)
    }

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