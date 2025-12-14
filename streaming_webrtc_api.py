#!/usr/bin/env python3
"""
FastAPI приложение для стриминга видео через WebRTC.
Использует aiortc для создания медиа потока из генерируемых кадров.
"""
import os
import sys
import json
import asyncio
import threading
import subprocess
import shlex
import shutil
import tempfile
import yaml
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import cv2
import numpy as np
from av import VideoFrame
from fractions import Fraction
import time
import traceback

# Совместимость с разными версиями Python для InvalidStateError
try:
    from asyncio.exceptions import InvalidStateError
except ImportError:
    # Для Python < 3.8
    InvalidStateError = asyncio.InvalidStateError

from dotenv import load_dotenv
load_dotenv()  # Загружает переменные из .env файла

# Добавляем путь к скриптам
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scripts.mmap_frame_buffer import MmapFrameReader

try:
    from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack, AudioStreamTrack, RTCConfiguration, RTCIceServer
    from aiortc.contrib.media import MediaPlayer, MediaRelay
    from av import AudioFrame
    from aiortc.rtcrtpsender import RTCRtpSender
    WEBRTC_AVAILABLE = True
except ImportError:
    WEBRTC_AVAILABLE = False
    print("[StreamingAPI] aiortc не установлен. Установите: pip install aiortc")

app = FastAPI()

# Добавляем CORS middleware для разрешения запросов с разных источников
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Управление активными WebRTC соединениями
active_connections = {}

# Управление активными генерациями (по task_id)
active_generations = {}
generation_lock = threading.Lock()

# Управление информацией о соединениях по task_id (сохраняется даже после завершения генерации)
connection_info_by_task = {}
connection_info_lock = threading.Lock()

# --- Вспомогательные функции ---

def _safe_close_media_player(media_player):
    """Безопасно останавливает и закрывает MediaPlayer"""
    if media_player:
        try:
            if hasattr(media_player, 'audio') and media_player.audio:
                media_player.audio.stop()
        except Exception as e:
            print(f"[StreamingAPI] Ошибка при остановке аудио: {e}")
        
        try:
            if hasattr(media_player, 'close'):
                media_player.close()
        except Exception as e:
            print(f"[StreamingAPI] Ошибка при закрытии MediaPlayer: {e}")

def _reset_audio_track(audio_track, new_audio_path=None):
    """Сбрасывает состояние аудио трека для перезапуска"""
    if not audio_track:
        return

    print(f"[StreamingAPI] Сброс состояния аудио трека...")
    
    if hasattr(audio_track, 'audio_started'):
        audio_track.audio_started = False
        print(f"[StreamingAPI] Флаг audio_started сброшен")

    if hasattr(audio_track, 'silence_frames_sent'):
        audio_track.silence_frames_sent = 0
    if hasattr(audio_track, '_recv_call_count'):
        audio_track._recv_call_count = 0
    if hasattr(audio_track, 'media_player_ended'):
        audio_track.media_player_ended = False
    if hasattr(audio_track, 'error_count'):
        audio_track.error_count = 0
    if hasattr(audio_track, 'last_error_log_time'):
        audio_track.last_error_log_time = 0
    
    # Перезапуск MediaPlayer для DelayedMediaPlayerTrack
    if hasattr(audio_track, 'media_player') and audio_track.media_player:
        old_media_player = audio_track.media_player
        _safe_close_media_player(old_media_player)
        
        path_to_use = new_audio_path if new_audio_path is not None else old_media_player._path
        
        try:
            new_media_player = MediaPlayer(path_to_use)
            if new_media_player and new_media_player.audio:
                audio_track.media_player = new_media_player
                if new_audio_path is not None:
                    audio_track.media_player._path = path_to_use
                print(f"[StreamingAPI] ✓ MediaPlayer перезапущен с аудио: {path_to_use}")
            else:
                print(f"[StreamingAPI] ⚠️ Не удалось создать новый MediaPlayer")
        except Exception as e:
            print(f"[StreamingAPI] Ошибка при перезапуске MediaPlayer: {e}")
            traceback.print_exc()

async def log_rtcp_stats(sender: RTCRtpSender, kind: str, interval: float = 2.0):
    """
    Периодически собирает и логирует статистику RTCP Sender Reports.
    """
    
    await asyncio.sleep(5.0)
    
    counter = 0
    while sender.transport.state != 'closed':
        try:
            await asyncio.sleep(interval)
            
            stats = await sender.getStats()
            
            sender_report_data = None
            for s in stats.values():
                if s.type == 'sender-report':
                    sender_report_data = s
                    break
            
            if sender_report_data:
                ntp_time = sender_report_data.ntpTimestamp
                rtp_time = sender_report_data.rtpTimestamp
                packets_sent = sender_report_data.packetsSent
                bytes_sent = sender_report_data.bytesSent
                
                if counter % 5 == 0:
                    print(f"[{kind.upper()} RTCP SR] NTP Time: {ntp_time}, RTP Time: {rtp_time}, Packets Sent: {packets_sent}, Bytes Sent: {bytes_sent}")
            else:
                if counter % 10 == 0:
                    print(f"[{kind.upper()} RTCP SR] Sender Report не найден в статистике.")
            
            counter += 1
            
        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"[{kind.upper()} RTCP SR] Ошибка при сборе статистики: {e}")
            break
            
# --- Треки ---

class DelayedMediaPlayerTrack(AudioStreamTrack):
    """Обертка для MediaPlayer с задержкой начала воспроизведения"""
    
    def __init__(self, media_player, video_track):
        super().__init__()
        self.media_player = media_player
        self.video_track = video_track
        self.audio_started = False
        self.silence_frames_sent = 0
        self.sample_rate = 48000
        self.time_base = Fraction(1, 48000)
        self.media_player_ended = False
        self.error_count = 0
        self.last_error_log_time = 0
    
    def _create_silence_frame(self, samples=480):
        """Создает кадр тишины с нулевыми данными"""
        silence = AudioFrame(format='s16', layout='stereo', samples=samples)
        silence.sample_rate = self.sample_rate
        silence.time_base = self.time_base
        silence.pts = self.silence_frames_sent * samples
        
        import numpy as np
        silence_data = np.zeros(samples * 2, dtype=np.int16)
        silence.planes[0].update(silence_data.tobytes())
        
        self.silence_frames_sent += 1
        return silence
        
    async def recv(self):
        """Получает следующий аудио кадр с задержкой до первого реального кадра"""
        if not hasattr(self, '_recv_call_count'):
            self._recv_call_count = 0
        self._recv_call_count += 1
        
        if not self.audio_started:
            if self.video_track:
                with self.video_track.first_real_frame_lock:
                    first_real_sent = self.video_track.first_real_frame_sent
                
                if first_real_sent:
                    self.audio_started = True
                    self.silence_frames_sent = 0  
                    print(f"[DelayedMediaPlayerTrack] ✓ Аудио синхронизировано с видео (первый реальный кадр отправлен, silence_frames_reset=0)")
                else:
                    await asyncio.sleep(0.01)
                    return self._create_silence_frame(480)
            else:
                await asyncio.sleep(0.01)
                return self._create_silence_frame(480)
        
        if self.media_player_ended:
            return self._create_silence_frame(480)
        
        if self.media_player and self.media_player.audio:
            try:
                frame = await self.media_player.audio.recv()
                self.error_count = 0
                
                try:
                    import numpy as np
                    audio_data = frame.to_ndarray()
                    actual_samples = frame.samples
                    
                    audio_frame = AudioFrame(format=frame.format.name, 
                                            layout=frame.layout.name, 
                                            samples=actual_samples)
                    audio_frame.sample_rate = self.sample_rate
                    audio_frame.time_base = self.time_base
                    
                    audio_frame.planes[0].update(audio_data.tobytes())
                    audio_frame.pts = frame.pts
                    
                    return audio_frame
                except Exception as copy_error:
                    return frame
            except Exception as e:
                self.error_count += 1
                error_str = str(e).lower()
                if 'end' in error_str or 'eof' in error_str or 'finished' in error_str:
                    self.media_player_ended = True
                    if self.error_count == 1:
                        print(f"[DelayedMediaPlayerTrack] MediaPlayer закончил воспроизведение (нормально)")
                    return self._create_silence_frame(480)
                
                current_time = time.time()
                if current_time - self.last_error_log_time >= 1.0:
                    print(f"[DelayedMediaPlayerTrack] Ошибка при получении кадра из MediaPlayer (ошибок: {self.error_count}): {e}")
                    self.last_error_log_time = current_time
                
                if self.error_count >= 10:
                    self.media_player_ended = True
                    print(f"[DelayedMediaPlayerTrack] Слишком много ошибок ({self.error_count}), считаем MediaPlayer завершенным")
                
                return self._create_silence_frame(480)
        else:
            return self._create_silence_frame(480)
    
    def stop(self):
        """Останавливает трек"""
        _safe_close_media_player(self.media_player)
        print(f"[DelayedMediaPlayerTrack] Аудио трек остановлен")


class VideoStreamGenerator(VideoStreamTrack):
    """Генератор видео потока из кадров"""
    
    def __init__(self, fps=25, loop=None, video_width=640, video_height=480, video_path=None):
        super().__init__()
        self.fps = fps
        self.frame_queue = asyncio.Queue(maxsize=100)
        self.running = True
        self.frame_time = 1.0 / fps
        self.start_time = None
        self.frame_count = 0
        self.loop = loop
        self._recv_called = False
        self.last_frame = None
        self.last_frame_lock = threading.Lock()
        self.keepalive_task = None
        self.original_video_task = None
        self.rtcp_task = None 
        self.video_width = video_width
        self.video_height = video_height
        self.video_path = video_path
        self.original_video_frames = []
        self.original_frame_index = 0
        self.first_real_frame_sent = False
        self.first_real_frame_lock = threading.Lock()
        self.generation_completed = False
        self.generation_completed_lock = threading.Lock()
        
        self.time_base = Fraction(1, 90000)
        
        self.conversion_times = []
        self.queue_wait_times = []
        self.last_metrics_log = time.time()
        self.metrics_lock = threading.Lock()
        
        print(f"[VideoStreamGenerator] Инициализирован: fps={fps}, loop={loop is not None}, размер={video_width}x{video_height}, video_path={video_path}")
        
        if video_path and os.path.exists(video_path):
            self._load_original_video_frames()
        
        if loop:
            async def send_test():
                await self._send_test_frame()
            asyncio.run_coroutine_threadsafe(send_test(), loop)
            
            def schedule_keepalive():
                self.keepalive_task = loop.create_task(self._start_keepalive())
            loop.call_soon_threadsafe(schedule_keepalive)
            
            def schedule_original_video():
                self.original_video_task = loop.create_task(self._play_original_video())
            loop.call_soon_threadsafe(schedule_original_video)
    
    def _load_original_video_frames(self):
        """Загружает все кадры из оригинального видео для использования в тестовых кадрах (с зацикливанием)"""
        try:
            video_cap = cv2.VideoCapture(self.video_path)
            if not video_cap.isOpened():
                print(f"[VideoStreamGenerator] ⚠️ Не удалось открыть видео для загрузки кадров: {self.video_path}")
                return
            
            total_frames = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            video_fps = video_cap.get(cv2.CAP_PROP_FPS)
            
            frame_count = 0
            
            while True:
                ret, frame = video_cap.read()
                if not ret:
                    break
                
                if frame.shape[1] != self.video_width or frame.shape[0] != self.video_height:
                    frame = cv2.resize(frame, (self.video_width, self.video_height))
                
                self.original_video_frames.append(frame.copy())
                frame_count += 1
                
            video_cap.release()
            
            if len(self.original_video_frames) > 0:
                print(f"[VideoStreamGenerator] ✓ Загружено {len(self.original_video_frames)} кадров из оригинального видео")
                with self.last_frame_lock:
                    self.last_frame = self.original_video_frames[0].copy()
        except Exception as e:
            print(f"[VideoStreamGenerator] Ошибка при загрузке кадров из оригинального видео: {e}")
            traceback.print_exc()
    
    async def _send_test_frame(self):
        """Отправляет тестовый кадр из оригинального видео (или серый, если видео не загружено)"""
        if len(self.original_video_frames) > 0:
            test_frame = self.original_video_frames[self.original_frame_index % len(self.original_video_frames)].copy()
            self.original_frame_index += 1
            await self.frame_queue.put(test_frame)
            with self.last_frame_lock:
                self.last_frame = test_frame.copy()
        else:
            test_frame = np.full((self.video_height, self.video_width, 3), 128, dtype=np.uint8)
            await self.frame_queue.put(test_frame)
            with self.last_frame_lock:
                self.last_frame = test_frame.copy()
    
    async def _play_original_video(self):
        """Воспроизводит оригинальное видео циклически, когда генерация не активна"""
        while len(self.original_video_frames) == 0 and self.running:
            await asyncio.sleep(0.1)
        
        if not self.running or len(self.original_video_frames) == 0:
            return
        
        frame_interval = 1.0 / self.fps
        
        while self.running:
            try:
                with self.generation_completed_lock:
                    gen_completed = self.generation_completed
                
                with self.first_real_frame_lock:
                    first_sent = self.first_real_frame_sent
                
                should_play = not first_sent or gen_completed
                
                if should_play:
                    queue_size = self.frame_queue.qsize()
                    if not self.frame_queue.full():
                        frame = self.original_video_frames[self.original_frame_index % len(self.original_video_frames)].copy()
                        self.original_frame_index += 1
                        
                        await self.frame_queue.put(frame)
                        
                        with self.last_frame_lock:
                            self.last_frame = frame.copy()
                        
                    else:
                        if gen_completed:
                            cleared = 0
                            while not self.frame_queue.empty() and cleared < 10:
                                try:
                                    self.frame_queue.get_nowait()
                                    cleared += 1
                                except:
                                    break
                            
                            if not self.frame_queue.full():
                                frame = self.original_video_frames[self.original_frame_index % len(self.original_video_frames)].copy()
                                self.original_frame_index += 1
                                await self.frame_queue.put(frame)
                                with self.last_frame_lock:
                                    self.last_frame = frame.copy()
                
                await asyncio.sleep(frame_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"[VideoStreamGenerator] Ошибка при воспроизведении оригинального видео: {e}")
                await asyncio.sleep(1.0)
        
        print(f"[VideoStreamGenerator] Задача воспроизведения оригинального видео остановлена")
    
    async def _start_keepalive(self):
        """Запускает задачу keepalive для периодической отправки последнего кадра"""
        await asyncio.sleep(2.0)
        
        keepalive_interval = 0.5
        
        while self.running:
            try:
                await asyncio.sleep(keepalive_interval)
                
                if not self.running:
                    break
                
                queue_size = self.frame_queue.qsize()
                
                if queue_size == 0:
                    try:
                        if not self.frame_queue.full():
                            keepalive_frame = None
                            
                            if self.last_frame is not None:
                                keepalive_frame = self.last_frame.copy()
                            elif len(self.original_video_frames) > 0:
                                keepalive_frame = self.original_video_frames[self.original_frame_index % len(self.original_video_frames)].copy()
                                self.original_frame_index += 1
                            else:
                                keepalive_frame = np.full((self.video_height, self.video_width, 3), 128, dtype=np.uint8)
                            
                            if keepalive_frame is not None:
                                await self.frame_queue.put(keepalive_frame)
                    except Exception as e:
                        pass
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"[VideoStreamGenerator] Keepalive: неожиданная ошибка: {e}")
                await asyncio.sleep(1.0)
        
        print(f"[VideoStreamGenerator] Keepalive задача остановлена")
    
    def _log_performance_metrics(self):
        """Логирует метрики производительности для диагностики"""
        with self.metrics_lock:
            if len(self.conversion_times) > 0:
                avg_conv = sum(self.conversion_times) / len(self.conversion_times)
                max_conv = max(self.conversion_times)
                min_conv = min(self.conversion_times)
            else:
                avg_conv = max_conv = min_conv = 0
            
            if len(self.queue_wait_times) > 0:
                avg_wait = sum(self.queue_wait_times) / len(self.queue_wait_times)
                max_wait = max(self.queue_wait_times)
            else:
                avg_wait = max_wait = 0
            
            queue_size = self.frame_queue.qsize()
            queue_capacity = self.frame_queue.maxsize
            queue_usage = (queue_size / queue_capacity * 100) if queue_capacity > 0 else 0
            
            if self.start_time:
                elapsed = time.time() - self.start_time
                current_fps = self.frame_count / elapsed if elapsed > 0 else 0
            else:
                current_fps = 0
            
            print(f"[VideoStreamGenerator] 📊 МЕТРИКИ ПРОИЗВОДИТЕЛЬНОСТИ:")
            print(f"  - Очередь: {queue_size}/{queue_capacity} ({queue_usage:.1f}%)")
            print(f"  - Конвертация кадров: avg={avg_conv*1000:.2f}ms, min={min_conv*1000:.2f}ms, max={max_conv*1000:.2f}ms")
            print(f"  - Ожидание в очереди: avg={avg_wait*1000:.2f}ms, max={max_wait*1000:.2f}ms")
            print(f"  - Текущий FPS: {current_fps:.2f} (целевой: {self.fps})")
            print(f"  - Всего кадров отправлено: {self.frame_count}")
    
    def add_frames_batch(self, frame_arrays):
        """
        Добавляет несколько кадров в очередь батчем (вызывается из другого потока).
        """
        if not self.running or not self.loop:
            return
        
        if not frame_arrays or len(frame_arrays) == 0:
            return
        
        valid_frames = []
        for frame_array in frame_arrays:
            if frame_array is None or len(frame_array.shape) != 3 or frame_array.shape[2] != 3:
                continue
            
            if frame_array.dtype != np.uint8:
                if frame_array.max() > 1.0:
                    frame_array = (frame_array / 255.0).clip(0, 1)
                frame_array = (frame_array * 255).astype(np.uint8)
            
            valid_frames.append(frame_array.copy())
        
        if len(valid_frames) == 0:
            return
        
        frames_to_add = valid_frames
        
        def add_batch_non_blocking():
            added_count = 0
            frames_skipped = 0
            for frame in frames_to_add:
                try:
                    self.frame_queue.put_nowait(frame)
                    added_count += 1
                except asyncio.QueueFull:
                    frames_skipped += 1
                    break
            
            if added_count > 0:
                with self.first_real_frame_lock:
                    if not self.first_real_frame_sent:
                        self.first_real_frame_sent = True
                
                with self.last_frame_lock:
                    self.last_frame = frames_to_add[added_count-1].copy()
            
            if frames_skipped > 0:
                print(f"[VideoStreamGenerator] ⚠️ Пропущено {frames_skipped} кадров из-за переполнения очереди!")
        
        try:
            self.loop.call_soon_threadsafe(add_batch_non_blocking)
        except Exception as e:
            print(f"[VideoStreamGenerator] Ошибка при добавлении батча кадров: {e}")
    
    def __repr__(self):
        return f"VideoStreamGenerator(fps={self.fps}, running={self.running}, queue_size={self.frame_queue.qsize()})"
        
    async def recv(self):
        """Получает следующий кадр для отправки"""
        self._recv_called = True
        
        if not self.running:
            raise Exception("Stream stopped")
        
        if self.start_time is None:
            self.start_time = time.time()
        
        try:
            queue_wait_start = time.time()
            frame_array = await asyncio.wait_for(self.frame_queue.get(), timeout=0.5)
            queue_wait_time = time.time() - queue_wait_start
            
            with self.metrics_lock:
                self.queue_wait_times.append(queue_wait_time)
                if len(self.queue_wait_times) > 100:
                    self.queue_wait_times.pop(0)
            
            if frame_array is None:
                raise Exception("None frame received")
            
            with self.last_frame_lock:
                self.last_frame = frame_array.copy()
            
            h, w = frame_array.shape[:2]
            if h % 2 != 0 or w % 2 != 0:
                h = (h // 2) * 2
                w = (w // 2) * 2
                frame_array = cv2.resize(frame_array, (w, h))
            
            conversion_start = time.time()
            try:
                frame = VideoFrame.from_ndarray(frame_array, format="bgr24")
                frame = frame.reformat(format="yuv420p")
                conversion_time = time.time() - conversion_start
                
                with self.metrics_lock:
                    self.conversion_times.append(conversion_time)
                    if len(self.conversion_times) > 100:
                        self.conversion_times.pop(0)
            except Exception as e:
                try:
                    frame = VideoFrame.from_ndarray(frame_array, format="rgb24")
                    frame = frame.reformat(format="yuv420p")
                except Exception as e2:
                    raise
            
            self.frame_count += 1
            pts = int(self.frame_count * 90000 / self.fps)
            frame.pts = pts
            frame.time_base = self.time_base
            
            current_time = time.time()
            if current_time - self.last_metrics_log > 5.0:
                self._log_performance_metrics()
                self.last_metrics_log = current_time
            
            return frame
        except asyncio.TimeoutError:
            with self.generation_completed_lock:
                gen_completed = self.generation_completed
            
            if gen_completed and len(self.original_video_frames) > 0:
                frame_array = self.original_video_frames[self.original_frame_index % len(self.original_video_frames)].copy()
                self.original_frame_index += 1
            else:
                with self.last_frame_lock:
                    if self.last_frame is not None:
                        frame_array = self.last_frame.copy()
                    elif len(self.original_video_frames) > 0:
                        frame_array = self.original_video_frames[self.original_frame_index % len(self.original_video_frames)].copy()
                        self.original_frame_index += 1
                    else:
                        frame_array = np.full((self.video_height, self.video_width, 3), 128, dtype=np.uint8)
            
            h, w = frame_array.shape[:2]
            if h % 2 != 0 or w % 2 != 0:
                h = (h // 2) * 2
                w = (w // 2) * 2
                frame_array = cv2.resize(frame_array, (w, h))
            
            try:
                frame = VideoFrame.from_ndarray(frame_array, format="bgr24")
                frame = frame.reformat(format="yuv420p")
            except Exception as e:
                frame = VideoFrame.from_ndarray(np.full((self.video_height, self.video_width, 3), 128, dtype=np.uint8), format="bgr24")
                frame = frame.reformat(format="yuv420p")
            
            self.frame_count += 1
            pts = int(self.frame_count * 90000 / self.fps)
            frame.pts = pts
            frame.time_base = self.time_base
            return frame
        except Exception as e:
            print(f"[VideoStreamGenerator] recv: КРИТИЧЕСКАЯ ОШИБКА: {e}")
            traceback.print_exc()
            try:
                frame = VideoFrame.from_ndarray(np.zeros((self.video_height, self.video_width, 3), dtype=np.uint8), format="bgr24")
                self.frame_count += 1
                pts = int(self.frame_count * 90000 / self.fps)
                frame.pts = pts
                frame.time_base = self.time_base
                return frame
            except:
                raise
    
    def add_frame(self, frame_array):
        """Добавляет кадр в очередь (вызывается из другого потока)"""
        if not self.running or not self.loop or frame_array is None:
            return
        
        if len(frame_array.shape) != 3 or frame_array.shape[2] != 3:
            return
        
        try:
            if frame_array.dtype != np.uint8:
                if frame_array.max() > 1.0:
                    frame_array = (frame_array / 255.0).clip(0, 1)
                frame_array = (frame_array * 255).astype(np.uint8)
            
            frame_final = frame_array.copy()
            
            def put_frame_non_blocking():
                try:
                    self.frame_queue.put_nowait(frame_final)
                    
                    with self.first_real_frame_lock:
                        if not self.first_real_frame_sent:
                            self.first_real_frame_sent = True
                    
                    with self.last_frame_lock:
                        self.last_frame = frame_final.copy()
                        
                except asyncio.QueueFull:
                    try:
                        self.frame_queue.get_nowait()
                        self.frame_queue.put_nowait(frame_final)
                    except:
                        pass
                except Exception as e:
                    print(f"[VideoStreamGenerator] Ошибка при добавлении кадра в очередь: {e}")

            self.loop.call_soon_threadsafe(put_frame_non_blocking)
            
        except Exception as e:
            print(f"[VideoStreamGenerator] Ошибка при обработке кадра: {e}")
            traceback.print_exc()
    
    def stop(self):
        """Останавливает поток и очищает очередь"""
        self.running = False
        
        if self.keepalive_task and not self.keepalive_task.done():
            if self.loop:
                def cancel_task():
                    self.keepalive_task.cancel()
                self.loop.call_soon_threadsafe(cancel_task)
        
        if self.original_video_task and not self.original_video_task.done():
            if self.loop:
                def cancel_original_task():
                    self.original_video_task.cancel()
                self.loop.call_soon_threadsafe(cancel_original_task)
        
        if self.rtcp_task and not self.rtcp_task.done():
            if self.loop:
                def cancel_rtcp_task():
                    self.rtcp_task.cancel()
                self.loop.call_soon_threadsafe(cancel_rtcp_task)
            self.rtcp_task = None
        
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except:
                break
        
        with self.last_frame_lock:
            self.last_frame = None
        
        print(f"[VideoStreamGenerator] Трек остановлен, очередь очищена, keepalive остановлен")


# --- Основная логика FastAPI ---

def check_daemon_service(version="v1.5", mode="realtime"):
    """Проверяет, запущен ли демон-сервис через PID файл"""
    if version == "v15":
        version = "v1.5"
    
    pid_file = f"./logs/model_service_{version}_{mode}.pid"
    
    if not os.path.exists(pid_file):
        return False, f"PID файл не найден: {pid_file}"
    
    try:
        with open(pid_file, 'r') as f:
            pid = int(f.read().strip())
        
        try:
            os.kill(pid, 0)
        except OSError:
            return False, f"Процесс с PID {pid} не найден"
        except AttributeError:
            import platform
            if platform.system() == "Windows":
                result = subprocess.run(
                    ["tasklist", "/FI", f"PID eq {pid}"],
                    capture_output=True,
                    text=True
                )
                if str(pid) not in result.stdout:
                    return False, f"Процесс с PID {pid} не найден"
            else:
                result = subprocess.run(
                    ["ps", "-p", str(pid)],
                    capture_output=True
                )
                if result.returncode != 0:
                    return False, f"Процесс с PID {pid} не найден"
        
        return True, pid
    except (ValueError, IOError) as e:
        return False, f"Ошибка при чтении PID файла: {e}"


def cleanup_resources(task_id: str, conn_info: dict):
    """Очищает WebRTC соединение, потоки инференса и временные файлы"""
    pc = conn_info.get("pc")
    video_track = conn_info.get("video_track")
    audio_track = conn_info.get("audio_track")
    audio_path = conn_info.get("audio_path")
    uploaded_audio = conn_info.get("uploaded_audio", False)
    
    # 1. Останавливаем треки
    if video_track:
        try:
            video_track.stop()
        except Exception as e:
            print(f"[StreamingAPI] Ошибка при остановке видео трека: {e}")
    if audio_track:
        try:
            audio_track.stop()
        except Exception as e:
            print(f"[StreamingAPI] Ошибка при остановке аудио трека: {e}")
    
    # 2. Останавливаем генерацию и удаляем из active_generations
    with generation_lock:
        if task_id in active_generations:
            gen_info = active_generations[task_id]
            print(f"[StreamingAPI] Остановка генерации {task_id} из-за закрытия соединения")
            gen_info["stop_flag"].set()
            del active_generations[task_id]
    
    # 3. Удаляем информацию о соединении
    with connection_info_lock:
        if task_id in connection_info_by_task:
            del connection_info_by_task[task_id]
            print(f"[StreamingAPI] Удаление информации о соединении {task_id}")
    
    # 4. Закрываем Peer Connection с безопасной обработкой ошибок aioice
    if pc and id(pc) in active_connections:
        del active_connections[id(pc)]
    if pc:
        try:
            # Проверяем состояние соединения перед закрытием
            try:
                state = pc.connectionState
                if state == "closed":
                    return  # Уже закрыто
            except:
                pass
            
            # Даем небольшую задержку для завершения STUN транзакций
            import time
            time.sleep(0.1)
            
            # Закрываем соединение с обработкой всех возможных ошибок
            loop = asyncio.get_event_loop()
            if loop.is_running():
                future = asyncio.run_coroutine_threadsafe(pc.close(), loop)
                try:
                    future.result(timeout=5)
                except (InvalidStateError, RuntimeError, Exception) as e:
                    # Игнорируем ошибки InvalidStateError из aioice при таймаутах STUN
                    if isinstance(e, InvalidStateError) or "invalid state" in str(e).lower():
                        print(f"[StreamingAPI] Предупреждение: Игнорируем InvalidStateError при закрытии соединения (это нормально при таймаутах STUN)")
                    else:
                        print(f"[StreamingAPI] Ошибка при закрытии соединения: {e}")
            else:
                # Если loop не запущен, запускаем синхронно
                asyncio.run(pc.close())
        except (InvalidStateError, RuntimeError, Exception) as e:
            # Игнорируем ошибки InvalidStateError из aioice
            if isinstance(e, InvalidStateError) or "invalid state" in str(e).lower():
                print(f"[StreamingAPI] Предупреждение: Игнорируем InvalidStateError при закрытии соединения (это нормально при таймаутах STUN)")
            else:
                print(f"[StreamingAPI] Ошибка при закрытии соединения: {e}")
    
    # 5. Удаляем временный аудио файл, если он был загружен
    # ВАЖНО: Удаление здесь часто приводит к гонке (race condition) с демоном,
    # который может пытаться прочитать файл для извлечения признаков.
    # Временно закомментировано, чтобы предотвратить "NoneType" ошибку в демоне.
    if uploaded_audio and audio_path and os.path.exists(audio_path):
        try:
            # os.remove(audio_path) 
            print(f"[StreamingAPI] ВНИМАНИЕ: Аудио файл {audio_path} НЕ был удален при очистке (предотвращение ошибки демона).")
            print(f"[StreamingAPI] Требуется периодическая очистка папки 'data/audio'.")
        except Exception as e:
            print(f"[StreamingAPI] Ошибка при удалении аудио файла {audio_path}: {e}")

@app.on_event("startup")
async def startup_event():
    """Инициализация при старте приложения"""
    if not WEBRTC_AVAILABLE:
        print("[StreamingAPI] ВНИМАНИЕ: aiortc не установлен. WebRTC стриминг недоступен.")
        return
    print("[StreamingAPI] WebRTC API готов к работе через файловую систему с демоном")

@app.on_event("shutdown")
async def shutdown_event():
    """Корректное завершение работы: остановка потоков и очистка ресурсов"""
    print("[StreamingAPI] Запуск процедуры завершения работы...")
    
    tasks_to_cancel = []
    
    with generation_lock:
        for task_id, gen_info in list(active_generations.items()):
            print(f"[StreamingAPI] Остановка генерации {task_id} перед завершением работы")
            gen_info["stop_flag"].set()
    
    with connection_info_lock:
        for task_id, conn_info in list(connection_info_by_task.items()):
            if conn_info["video_track"] and conn_info["video_track"].rtcp_task:
                tasks_to_cancel.append(conn_info["video_track"].rtcp_task)

            cleanup_resources(task_id, conn_info)
    
    for task in tasks_to_cancel:
        if not task.done():
            task.cancel()

@app.get("/")
async def get_index():
    """Возвращает HTML страницу для тестирования"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>MuseTalk WebRTC Streaming Test</title>
    </head>
    <body>
        <h1>MuseTalk WebRTC Streaming Test</h1>
        <p>Откройте <a href="/test">streaming_webrtc_test.html</a> для тестирования WebRTC стриминга</p>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@app.post("/api/webrtc/offer")
async def webrtc_offer(request: dict):
    """Обрабатывает WebRTC offer и возвращает answer"""
    if not WEBRTC_AVAILABLE:
        raise HTTPException(status_code=503, detail="WebRTC не доступен. Установите aiortc.")
    
    try:
        if "sdp" not in request or "type" not in request:
            raise HTTPException(status_code=400, detail="Отсутствуют поля sdp или type в запросе")
        
        if not request.get("sdp") or not request.get("type"):
            raise HTTPException(status_code=400, detail="Поля sdp или type пусты")
        
        offer = RTCSessionDescription(sdp=request["sdp"], type=request["type"])
        
        task_id = request.get("task_id")
        video_path = request.get("video_path")
        audio_path = request.get("audio_path")
        version = request.get("version", "v15")
        batch_size = request.get("batch_size", 8)
        fps = request.get("fps", 25)
        
        missing_params = []
        if not task_id:
            missing_params.append("task_id")
        if not video_path:
            missing_params.append("video_path")
        if not audio_path:
            missing_params.append("audio_path")
        
        if missing_params:
            raise HTTPException(
                status_code=400, 
                detail=f"Отсутствуют обязательные параметры: {', '.join(missing_params)}"
            )
        
        if not os.path.exists(video_path):
            raise HTTPException(status_code=400, detail=f"Видео файл не найден: {video_path}")
        
        if not os.path.exists(audio_path):
            print(f"[StreamingAPI] ⚠️ Предупреждение: Аудио файл не найден: {audio_path}. Будет использоваться заглушка.")
        
        loop = asyncio.get_event_loop()
        
        video_width = 640
        video_height = 480
        try:
            video_cap = cv2.VideoCapture(video_path)
            if video_cap.isOpened():
                video_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                video_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                video_cap.release()
        except Exception as e:
            print(f"[StreamingAPI] ⚠️ Ошибка при получении размеров видео: {e}, используем значения по умолчанию: {video_width}x{video_height}")
        
        video_track = VideoStreamGenerator(fps=fps, loop=loop, video_width=video_width, video_height=video_height, video_path=video_path)
        print(f"[StreamingAPI] Видео трек создан: {video_track}")
        
        # Настройка ICE серверов
        ice_servers = []
        ice_servers.append(RTCIceServer(urls=["stun:stun.l.google.com:19302"]))
        ice_servers.append(RTCIceServer(urls=["stun:stun1.l.google.com:19302"]))
        ice_servers.append(RTCIceServer(urls=["stun:stun2.l.google.com:19302"]))
        
        turn_url = os.getenv("TURN_SERVER_URL")
        turn_username = os.getenv("TURN_SERVER_USERNAME", "")
        turn_credential = os.getenv("TURN_SERVER_CREDENTIAL", "")
        use_public_turn = os.getenv("USE_PUBLIC_TURN", "true").lower() == "true"
        
        turn_servers_added = 0
        
        if turn_url and turn_username and turn_credential:
            turn_urls = [turn_url]
            if "?transport=" not in turn_url:
                turn_urls.append(f"{turn_url}?transport=tcp")
            
            ice_servers.append(
                RTCIceServer(
                    urls=turn_urls,
                    username=turn_username,
                    credential=turn_credential
                )
            )
            turn_servers_added += 1
            print(f"[StreamingAPI] ✓ Собственный TURN сервер настроен: {turn_url}")
        
        if use_public_turn and turn_servers_added == 0:
            print(f"[StreamingAPI] Используются публичные TURN серверы")
            ice_servers.append(
                RTCIceServer(
                    urls=[
                        "turn:openrelay.metered.ca:80",
                        "turn:openrelay.metered.ca:443",
                        "turn:openrelay.metered.ca:443?transport=tcp"
                    ],
                    username="openrelayproject",
                    credential="openrelayproject"
                )
            )
            ice_servers.append(
                RTCIceServer(
                    urls=[
                        "turn:relay.metered.ca:80",
                        "turn:relay.metered.ca:443",
                        "turn:relay.metered.ca:443?transport=tcp"
                    ],
                    username="openrelayproject",
                    credential="openrelayproject"
                )
            )
            turn_servers_added += 2
        
        pc = RTCPeerConnection(
            configuration=RTCConfiguration(iceServers=ice_servers),
        )
        
        connection_id = id(pc)
        active_connections[connection_id] = pc
        
        recv_check_started = False
        ice_candidates_count = {'host': 0, 'srflx': 0, 'relay': 0, 'prflx': 0}
        
        async def check_recv_called_after_connection():
            nonlocal recv_check_started
            if recv_check_started:
                return
            
            recv_check_started = True
            max_wait_time = 20
            check_interval = 1.0
            waited = 0
            
            while waited < max_wait_time:
                await asyncio.sleep(check_interval)
                waited += check_interval
                
                ice_state_current = pc.iceConnectionState
                conn_state_current = pc.connectionState
                
                if video_track._recv_called:
                    print(f"[StreamingAPI] ===== ПРОВЕРКА ЗАВЕРШЕНА: recv() был вызван через {waited:.1f}с =====")
                    return
                
                if ice_state_current in ["failed", "disconnected", "closed"] or conn_state_current in ["failed", "disconnected", "closed"]:
                    print(f"[StreamingAPI] ===== ПРОВЕРКА ПРЕРВАНА: Соединение потеряно =====")
                    return
                
                if (ice_state_current == "connected" or conn_state_current == "connected") and waited >= 4:
                    break
        
        def safe_cleanup():
            tid_to_cleanup = None
            for tid, info in list(connection_info_by_task.items()):
                if info.get("pc") == pc:
                    tid_to_cleanup = tid
                    break
            
            if tid_to_cleanup:
                conn_info = connection_info_by_task.get(tid_to_cleanup)
                if conn_info:
                    cleanup_resources(tid_to_cleanup, conn_info)
            else:
                try:
                    video_track.stop()
                    if audio_track:
                        audio_track.stop()
                    if connection_id in active_connections:
                        del active_connections[connection_id]
                    
                    # Безопасное закрытие с обработкой ошибок aioice
                    try:
                        state = pc.connectionState
                        if state == "closed":
                            return
                    except:
                        pass
                    
                    import time
                    time.sleep(0.1)  # Небольшая задержка для завершения STUN транзакций
                    
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        future = asyncio.run_coroutine_threadsafe(pc.close(), loop)
                        try:
                            future.result(timeout=5)
                        except (InvalidStateError, RuntimeError, Exception) as e:
                            if isinstance(e, InvalidStateError) or "invalid state" in str(e).lower():
                                pass  # Игнорируем ошибки InvalidStateError из aioice
                            else:
                                print(f"[StreamingAPI] Ошибка при закрытии соединения в safe_cleanup: {e}")
                    else:
                        asyncio.run(pc.close())
                except (InvalidStateError, RuntimeError, Exception) as e:
                    if isinstance(e, InvalidStateError) or "invalid state" in str(e).lower():
                        pass  # Игнорируем ошибки InvalidStateError из aioice
                    else:
                        print(f"[StreamingAPI] Ошибка в safe_cleanup: {e}")

        @pc.on("iceconnectionstatechange")
        async def on_ice_connection_state_change():
            try:
                state = pc.iceConnectionState
                print(f"[StreamingAPI] ICE connection state: {state}")
                
                if state == "connected":
                    if not recv_check_started:
                        asyncio.create_task(check_recv_called_after_connection())
                
                if state in ["failed", "closed"]:
                    print(f"[StreamingAPI] Очистка ресурсов из-за состояния ICE: {state}")
                    # Небольшая задержка перед очисткой, чтобы дать STUN транзакциям завершиться
                    await asyncio.sleep(0.1)
                    safe_cleanup()
            except (InvalidStateError, RuntimeError, Exception) as e:
                # Игнорируем ошибки InvalidStateError из aioice при таймаутах STUN
                if isinstance(e, InvalidStateError) or "invalid state" in str(e).lower():
                    print(f"[StreamingAPI] Предупреждение: Игнорируем InvalidStateError в обработчике ICE (это нормально при таймаутах STUN)")
                else:
                    print(f"[StreamingAPI] Ошибка в обработчике ICE connection state: {e}")
            
        @pc.on("icegatheringstatechange")
        async def on_ice_gathering_state_change():
            try:
                state = pc.iceGatheringState
                if state == "complete" and pc.localDescription:
                    sdp = pc.localDescription.sdp
                    relay_count = sdp.count('typ relay')
                    ice_candidates_count['relay'] = max(ice_candidates_count.get('relay', 0), relay_count)
                    
                    if relay_count > 0:
                        print(f"[StreamingAPI] ✓✓✓ TURN сервер работает! Найдено {relay_count} relay кандидатов в SDP ✓✓✓")
                    else:
                        print(f"[StreamingAPI] ⚠️ TURN сервер не используется - нет relay кандидатов в SDP")
            except (InvalidStateError, RuntimeError, Exception) as e:
                # Игнорируем ошибки InvalidStateError из aioice при таймаутах STUN
                if isinstance(e, InvalidStateError) or "invalid state" in str(e).lower():
                    pass  # Игнорируем
                else:
                    print(f"[StreamingAPI] Ошибка в обработчике ICE gathering state: {e}")

        @pc.on("icecandidate")
        async def on_ice_candidate(candidate):
            try:
                if candidate:
                    candidate_type = getattr(candidate, 'type', 'unknown')
                    if candidate_type in ice_candidates_count:
                        ice_candidates_count[candidate_type] += 1
            except (InvalidStateError, RuntimeError, Exception) as e:
                # Игнорируем ошибки InvalidStateError из aioice при таймаутах STUN
                if isinstance(e, InvalidStateError) or "invalid state" in str(e).lower():
                    pass  # Игнорируем
                else:
                    print(f"[StreamingAPI] Ошибка в обработчике ICE candidate: {e}")
                
        @pc.on("connectionstatechange")
        async def on_connection_state_change():
            try:
                state = pc.connectionState
                print(f"[StreamingAPI] Connection state: {state}")
                
                if state == "connected":
                    if not recv_check_started:
                        asyncio.create_task(check_recv_called_after_connection())
                
                if state in ["failed", "closed"]:
                    print(f"[StreamingAPI] ⚠️ Очистка ресурсов из-за состояния соединения: {state}")
                    # Небольшая задержка перед очисткой, чтобы дать STUN транзакциям завершиться
                    await asyncio.sleep(0.1)
                    safe_cleanup()
            except (InvalidStateError, RuntimeError, Exception) as e:
                # Игнорируем ошибки InvalidStateError из aioice при таймаутах STUN
                if isinstance(e, InvalidStateError) or "invalid state" in str(e).lower():
                    print(f"[StreamingAPI] Предупреждение: Игнорируем InvalidStateError в обработчике connection state (это нормально при таймаутах STUN)")
                else:
                    print(f"[StreamingAPI] Ошибка в обработчике connection state: {e}")
        
        @pc.on("track")
        async def on_track(track):
            try:
                print(f"[StreamingAPI] Track received: {track.kind}, id={track.id}")
                
                sender = None
                if track.kind == 'audio' or track.kind == 'video':
                    for s in pc.getSenders():
                        if s.track and s.track.kind == track.kind:
                            sender = s
                            break
                
                if sender and track.kind == 'video':
                    print(f"[StreamingAPI] Найдено исходящий Sender для {track.kind}. Запуск сбора статистики RTCP...")
                    video_track.rtcp_task = pc.loop.create_task(log_rtcp_stats(sender, track.kind))
            except (InvalidStateError, RuntimeError, Exception) as e:
                # Игнорируем ошибки InvalidStateError из aioice при таймаутах STUN
                if isinstance(e, InvalidStateError) or "invalid state" in str(e).lower():
                    pass  # Игнорируем
                else:
                    print(f"[StreamingAPI] Ошибка в обработчике track: {e}")
        
        # Добавляем треки
        video_sender = pc.addTrack(video_track)
        print(f"[StreamingAPI] Видео трек добавлен в peer connection.")
        
        audio_track = None
        is_uploaded_audio = False
        try:
            if os.path.exists(audio_path):
                print(f"[StreamingAPI] Использование MediaPlayer для аудио с синхронизацией (DelayedMediaPlayerTrack)")
                audio_player = MediaPlayer(audio_path)
                if audio_player and audio_player.audio:
                    audio_track = DelayedMediaPlayerTrack(audio_player, video_track)
                    audio_sender = pc.addTrack(audio_track)
                    print(f"[StreamingAPI] ✓ Аудио поток добавлен через MediaPlayer.")
                else:
                    print(f"[StreamingAPI] ⚠️ Предупреждение: не удалось создать аудио поток через MediaPlayer")
        except Exception as e:
            print(f"[StreamingAPI] Критическая ошибка при создании аудио потока: {e}")
            traceback.print_exc()
        
        try:
            await pc.setRemoteDescription(offer)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Ошибка при установке remote description: {str(e)}")
        
        try:
            answer = await pc.createAnswer()
            await pc.setLocalDescription(answer)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Ошибка при создании answer: {str(e)}")
        
        if not pc.localDescription:
            raise HTTPException(status_code=500, detail="Не удалось создать local description")
        
        # Сохраняем информацию о соединении
        with connection_info_lock:
            connection_info_by_task[task_id] = {
                "pc": pc,
                "video_track": video_track,
                "audio_track": audio_track,
                "video_path": video_path,
                "audio_path": audio_path,
                "version": version,
                "batch_size": batch_size,
                "fps": fps,
                "uploaded_audio": is_uploaded_audio
            }
        
        response = {
            "sdp": pc.localDescription.sdp,
            "type": pc.localDescription.type
        }
        
        if turn_url and turn_username and turn_credential:
            response["turn_server"] = {
                "url": turn_url,
                "username": turn_username,
                "credential": turn_credential
            }
        
        return response
        
    except Exception as e:
        print(f"[StreamingAPI] Ошибка в WebRTC offer: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/webrtc/restart_generation")
async def restart_generation(request: dict):
    """Запускает или перезапускает генерацию видео для существующего WebRTC соединения"""
    if not WEBRTC_AVAILABLE:
        raise HTTPException(status_code=503, detail="WebRTC не доступен. Установите aiortc.")
    
    try:
        task_id = request.get("task_id")
        if not task_id:
            raise HTTPException(status_code=400, detail="Отсутствует task_id")
        
        with connection_info_lock:
            if task_id not in connection_info_by_task:
                raise HTTPException(status_code=404, detail=f"Соединение для task_id={task_id} не найдено")
            
            conn_info = connection_info_by_task[task_id]
            pc = conn_info["pc"]
            video_track = conn_info["video_track"]
            audio_track = conn_info.get("audio_track")
            
            video_path = conn_info["video_path"]
            audio_path = conn_info["audio_path"]
            version = conn_info["version"]
            batch_size = conn_info["batch_size"]
            fps = conn_info["fps"]
        
        try:
            ice_state = pc.iceConnectionState
            conn_state = pc.connectionState
        except:
            raise HTTPException(status_code=400, detail="WebRTC соединение не доступно")
        
        if ice_state not in ["connected", "checking"] and conn_state not in ["connected", "connecting"]:
            raise HTTPException(status_code=400, detail=f"WebRTC соединение не активно (ICE: {ice_state}, Connection: {conn_state})")
        
        with generation_lock:
            has_active_generation = task_id in active_generations
            
            stop_flag = threading.Event()
            if has_active_generation:
                print(f"[StreamingAPI] Остановка текущей генерации для task_id={task_id}")
                gen_info = active_generations[task_id]
                gen_info["stop_flag"].set()
                
                inference_thread = gen_info["inference_thread"]
                if inference_thread.is_alive():
                    inference_thread.join(timeout=5.0)
                
                del active_generations[task_id]
            else:
                print(f"[StreamingAPI] Генерация для task_id={task_id} уже завершена, запускаем новую")
            
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
            
            _reset_audio_track(audio_track, new_audio_path=audio_path)
            
        
        def run_inference():
            try:
                print(f"[StreamingAPI] {'Перезапуск' if has_active_generation else 'Запуск'} генерации видео для task_id={task_id} (через демон, mmap)")
                
                config_data = {
                    "webrtc_mode": True,
                    "batch_size": batch_size,
                    "fps": fps,
                    task_id: {
                        "video_path": video_path,
                        "audio_path": audio_path
                    }
                }
                
                request_config_path = os.path.join("./requests", f"webrtc_{task_id}_{int(time.time())}.yaml")
                os.makedirs("./requests", exist_ok=True)
                
                with open(request_config_path, 'w') as f:
                    yaml.dump(config_data, f, default_flow_style=False)
                os.chmod(request_config_path, 0o644)
                
                time.sleep(2)
                
                mmap_reader = None
                from scripts.mmap_frame_buffer import get_mmap_path
                mmap_file_path = get_mmap_path(task_id)
                
                print(f"[StreamingAPI] Ожидание создания mmap файла: {mmap_file_path}", flush=True)
                
                max_wait_for_mmap = 60
                wait_start = time.time()
                mmap_created = False
                last_log_time = wait_start
                log_interval = 5  # Логируем каждые 5 секунд
                
                while time.time() - wait_start < max_wait_for_mmap:
                    current_time = time.time()
                    elapsed = current_time - wait_start
                    
                    # Логируем прогресс каждые 5 секунд
                    if current_time - last_log_time >= log_interval:
                        print(f"[StreamingAPI] Ожидание mmap файла... прошло {elapsed:.1f} секунд из {max_wait_for_mmap}", flush=True)
                        last_log_time = current_time
                    
                    if os.path.exists(mmap_file_path):
                        file_size = os.path.getsize(mmap_file_path)
                        if file_size > 0:
                            try:
                                mmap_reader = MmapFrameReader(task_id)
                                # Проверяем статус - если ошибка, выдаем понятное сообщение
                                status = mmap_reader.get_status()
                                if status.get("status") == "error":
                                    error_msg = f"Демон сообщил об ошибке при обработке задачи {task_id}. Проверьте логи демона для деталей."
                                    print(f"[StreamingAPI] ОШИБКА: {error_msg}", flush=True)
                                    print(f"[StreamingAPI] Статус из mmap: {status}", flush=True)
                                    raise RuntimeError(error_msg)
                                mmap_created = True
                                print(f"[StreamingAPI] Mmap файл успешно создан и открыт: {mmap_file_path} (размер: {file_size} байт)", flush=True)
                                print(f"[StreamingAPI] Статус: {status}", flush=True)
                                break
                            except RuntimeError:
                                # Пробрасываем RuntimeError (ошибка статуса)
                                raise
                            except (ValueError, FileNotFoundError) as e:
                                print(f"[StreamingAPI] Предупреждение: mmap файл существует, но не может быть открыт: {e}", flush=True)
                                time.sleep(0.5)
                        else:
                            # Файл существует, но пуст - возможно, это индикатор ошибки
                            if elapsed > 10:  # После 10 секунд предупреждаем
                                print(f"[StreamingAPI] Предупреждение: mmap файл существует, но пуст (размер: {file_size} байт). Возможно, демон еще инициализируется или произошла ошибка.", flush=True)
                            time.sleep(0.5)
                    else:
                        # Проверяем, существует ли директория для mmap файлов
                        mmap_dir = os.path.dirname(mmap_file_path)
                        if not os.path.exists(mmap_dir):
                            if elapsed > 5:  # После 5 секунд предупреждаем
                                print(f"[StreamingAPI] Предупреждение: директория для mmap файлов не существует: {mmap_dir}", flush=True)
                        time.sleep(0.5)
                
                if not mmap_created:
                    error_details = []
                    error_details.append(f"Mmap файл не был создан демоном в течение {max_wait_for_mmap} секунд")
                    error_details.append(f"Путь к файлу: {mmap_file_path}")
                    
                    # Проверяем, существует ли файл (даже если пустой)
                    if os.path.exists(mmap_file_path):
                        file_size = os.path.getsize(mmap_file_path)
                        # Пытаемся проверить статус, если файл имеет правильный размер
                        if file_size >= 32:  # Минимальный размер заголовка
                            try:
                                mmap_reader = MmapFrameReader(task_id)
                                status = mmap_reader.get_status()
                                if status.get("status") == "error":
                                    error_details.append(f"Файл существует с размером {file_size} байт и содержит статус ошибки")
                                    error_details.append(f"Статус: {status}")
                                    error_msg = "\n".join(error_details)
                                    print(f"[StreamingAPI] ОШИБКА: {error_msg}", flush=True)
                                    raise RuntimeError(error_msg)
                            except (ValueError, FileNotFoundError):
                                pass  # Игнорируем ошибки чтения, продолжаем с обычной обработкой
                        error_details.append(f"Файл существует, но размер: {file_size} байт (возможно, демон не смог инициализировать mmap)")
                    else:
                        error_details.append("Файл не существует")
                        mmap_dir = os.path.dirname(mmap_file_path)
                        if os.path.exists(mmap_dir):
                            error_details.append(f"Директория существует: {mmap_dir}")
                            # Проверяем другие файлы в директории
                            try:
                                other_files = os.listdir(mmap_dir)
                                if other_files:
                                    error_details.append(f"В директории есть другие файлы: {other_files[:5]}")
                            except:
                                pass
                        else:
                            error_details.append(f"Директория не существует: {mmap_dir}")
                    
                    error_details.append("Возможные причины:")
                    error_details.append("1. Демон не запущен или не обрабатывает запрос")
                    error_details.append("2. Аватар не был подготовлен (frame_list_cycle пуст)")
                    error_details.append("3. Ошибка при создании mmap файла (недостаточно памяти или прав доступа)")
                    error_details.append("4. Демон обрабатывает другой запрос и еще не дошел до этого")
                    
                    error_msg = "\n".join(error_details)
                    print(f"[StreamingAPI] ОШИБКА: {error_msg}", flush=True)
                    raise RuntimeError(error_msg)
                
                last_frame_index = -1
                max_wait_time = 300
                start_time = time.time()
                
                try:
                    frame_batch = []
                    batch_size_send = 8
                    
                    while True:
                        if stop_flag.is_set():
                            break
                        
                        if time.time() - start_time > max_wait_time:
                            break
                        
                        status = mmap_reader.get_status()
                        
                        current_frame_index = last_frame_index + 1
                        max_frames_to_read = status.get("frames_generated", 0)
                        
                        while current_frame_index < max_frames_to_read:
                            try:
                                frame_array = mmap_reader.read_frame(current_frame_index)
                                if frame_array is not None:
                                    frame_batch.append(frame_array)
                                    last_frame_index = current_frame_index
                                    current_frame_index += 1
                                    
                                    if len(frame_batch) >= batch_size_send:
                                        video_track.add_frames_batch(frame_batch)
                                        frame_batch = []
                                else:
                                    break
                            except Exception as e:
                                break
                        
                        if status.get("status") == "completed":
                            if frame_batch:
                                video_track.add_frames_batch(frame_batch)
                            
                            while current_frame_index < status.get("total_frames", 0):
                                try:
                                    frame_array = mmap_reader.read_frame(current_frame_index)
                                    if frame_array is not None:
                                        video_track.add_frame(frame_array)
                                        last_frame_index = current_frame_index
                                    else:
                                        break
                                except:
                                    break
                                current_frame_index += 1
                            break
                        elif status.get("status") in ["stopped", "error"]:
                            if frame_batch:
                                video_track.add_frames_batch(frame_batch)
                            break
                        
                        if len(frame_batch) > 0 and (time.time() - start_time) % 0.1 < 0.01:
                            video_track.add_frames_batch(frame_batch)
                            frame_batch = []
                        
                        time.sleep(0.01)
                finally:
                    if mmap_reader:
                        mmap_reader.close()
                        try:
                            mmap_reader.cleanup()
                        except Exception as e:
                            pass
                
                with video_track.generation_completed_lock:
                    video_track.generation_completed = True
            except Exception as e:
                traceback.print_exc()
            finally:
                with generation_lock:
                    if task_id in active_generations:
                        del active_generations[task_id]
        
        new_inference_thread = threading.Thread(target=run_inference, daemon=True)
        new_inference_thread.start()
        
        with generation_lock:
            active_generations[task_id] = {
                "video_track": video_track,
                "audio_track": audio_track,
                "inference_thread": new_inference_thread,
                "stop_flag": stop_flag,
                "pc": pc,
                "task_id": task_id,
                "video_path": video_path,
                "audio_path": audio_path,
                "version": version,
                "batch_size": batch_size,
                "fps": fps
            }
        
        action = "перезапущена" if has_active_generation else "запущена"
        return {
            "status": "ok",
            "message": f"Генерация для task_id={task_id} {action}",
            "task_id": task_id,
            "was_running": has_active_generation
        }
        
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/webrtc/upload_audio_and_generate")
async def upload_audio_and_generate(
    task_id: str = Form(...),
    audio_file: UploadFile = File(...)
):
    """
    Принимает аудио файл и запускает генерацию в трансляцию для существующего WebRTC соединения.
    """
    if not WEBRTC_AVAILABLE:
        raise HTTPException(status_code=503, detail="WebRTC не доступен. Установите aiortc.")
    
    try:
        with connection_info_lock:
            if task_id not in connection_info_by_task:
                raise HTTPException(
                    status_code=404, 
                    detail=f"Соединение для task_id={task_id} не найдено."
                )
            
            conn_info = connection_info_by_task[task_id]
            pc = conn_info["pc"]
            video_track = conn_info["video_track"]
            audio_track = conn_info.get("audio_track")
            video_path = conn_info["video_path"]
            
            version = conn_info.get("version", "v15")
            batch_size = conn_info.get("batch_size", 8)
            fps = conn_info.get("fps", 25)
        
        try:
            ice_state = pc.iceConnectionState
            conn_state = pc.connectionState
        except:
            raise HTTPException(status_code=400, detail="WebRTC соединение не доступно")
        
        if ice_state not in ["connected", "checking"] and conn_state not in ["connected", "connecting"]:
            raise HTTPException(
                status_code=400, 
                detail=f"WebRTC соединение не активно (ICE: {ice_state}, Connection: {conn_state})"
            )
        
        audio_dir = "data/audio"
        os.makedirs(audio_dir, exist_ok=True)
        
        timestamp = int(time.time())
        file_extension = os.path.splitext(audio_file.filename)[1] if audio_file.filename else ".wav"
        audio_filename = f"{task_id}_{timestamp}{file_extension}"
        new_audio_path = os.path.join(audio_dir, audio_filename)
        
        content = await audio_file.read()
        with open(new_audio_path, "wb") as f:
            f.write(content)
        
        with generation_lock:
            has_active_generation = task_id in active_generations
            
            stop_flag = threading.Event()
            if has_active_generation:
                gen_info = active_generations[task_id]
                gen_info["stop_flag"].set()
                inference_thread = gen_info["inference_thread"]
                if inference_thread.is_alive():
                    inference_thread.join(timeout=5.0)
                del active_generations[task_id]
            
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
            
            _reset_audio_track(audio_track, new_audio_path=new_audio_path)
        
        with connection_info_lock:
            # Удаляем ТОЛЬКО старый файл, если он был загружен, чтобы не засорять диск
            old_audio_path = conn_info.get("audio_path")
            if conn_info.get("uploaded_audio") and old_audio_path and os.path.exists(old_audio_path):
                try:
                    os.remove(old_audio_path)
                    print(f"[StreamingAPI] Удален старый временный аудио файл: {old_audio_path}")
                except Exception as e:
                    print(f"[StreamingAPI] Ошибка при удалении старого аудио файла {old_audio_path}: {e}")
            
            conn_info["audio_path"] = new_audio_path
            conn_info["uploaded_audio"] = True
            
        def run_inference():
            try:
                config_data = {
                    "webrtc_mode": True,
                    "batch_size": batch_size,
                    "fps": fps,
                    task_id: {
                        "video_path": video_path,
                        "audio_path": new_audio_path
                    }
                }
                
                request_config_path = os.path.join("./requests", f"webrtc_{task_id}_{int(time.time())}.yaml")
                os.makedirs("./requests", exist_ok=True)
                
                with open(request_config_path, 'w') as f:
                    yaml.dump(config_data, f, default_flow_style=False)
                os.chmod(request_config_path, 0o644)
                
                time.sleep(2)
                
                mmap_reader = None
                from scripts.mmap_frame_buffer import get_mmap_path
                mmap_file_path = get_mmap_path(task_id)
                
                print(f"[StreamingAPI] Ожидание создания mmap файла: {mmap_file_path}", flush=True)
                
                max_wait_for_mmap = 60
                wait_start = time.time()
                mmap_created = False
                last_log_time = wait_start
                log_interval = 5  # Логируем каждые 5 секунд
                
                while time.time() - wait_start < max_wait_for_mmap:
                    current_time = time.time()
                    elapsed = current_time - wait_start
                    
                    # Логируем прогресс каждые 5 секунд
                    if current_time - last_log_time >= log_interval:
                        print(f"[StreamingAPI] Ожидание mmap файла... прошло {elapsed:.1f} секунд из {max_wait_for_mmap}", flush=True)
                        last_log_time = current_time
                    
                    if os.path.exists(mmap_file_path):
                        file_size = os.path.getsize(mmap_file_path)
                        if file_size > 0:
                            try:
                                mmap_reader = MmapFrameReader(task_id)
                                # Проверяем статус - если ошибка, выдаем понятное сообщение
                                status = mmap_reader.get_status()
                                if status.get("status") == "error":
                                    error_msg = f"Демон сообщил об ошибке при обработке задачи {task_id}. Проверьте логи демона для деталей."
                                    print(f"[StreamingAPI] ОШИБКА: {error_msg}", flush=True)
                                    print(f"[StreamingAPI] Статус из mmap: {status}", flush=True)
                                    raise RuntimeError(error_msg)
                                mmap_created = True
                                print(f"[StreamingAPI] Mmap файл успешно создан и открыт: {mmap_file_path} (размер: {file_size} байт)", flush=True)
                                print(f"[StreamingAPI] Статус: {status}", flush=True)
                                break
                            except RuntimeError:
                                # Пробрасываем RuntimeError (ошибка статуса)
                                raise
                            except (ValueError, FileNotFoundError) as e:
                                print(f"[StreamingAPI] Предупреждение: mmap файл существует, но не может быть открыт: {e}", flush=True)
                                time.sleep(0.5)
                        else:
                            # Файл существует, но пуст - возможно, это индикатор ошибки
                            if elapsed > 10:  # После 10 секунд предупреждаем
                                print(f"[StreamingAPI] Предупреждение: mmap файл существует, но пуст (размер: {file_size} байт). Возможно, демон еще инициализируется или произошла ошибка.", flush=True)
                            time.sleep(0.5)
                    else:
                        # Проверяем, существует ли директория для mmap файлов
                        mmap_dir = os.path.dirname(mmap_file_path)
                        if not os.path.exists(mmap_dir):
                            if elapsed > 5:  # После 5 секунд предупреждаем
                                print(f"[StreamingAPI] Предупреждение: директория для mmap файлов не существует: {mmap_dir}", flush=True)
                        time.sleep(0.5)
                
                if not mmap_created:
                    error_details = []
                    error_details.append(f"Mmap файл не был создан демоном в течение {max_wait_for_mmap} секунд")
                    error_details.append(f"Путь к файлу: {mmap_file_path}")
                    
                    # Проверяем, существует ли файл (даже если пустой)
                    if os.path.exists(mmap_file_path):
                        file_size = os.path.getsize(mmap_file_path)
                        # Пытаемся проверить статус, если файл имеет правильный размер
                        if file_size >= 32:  # Минимальный размер заголовка
                            try:
                                mmap_reader = MmapFrameReader(task_id)
                                status = mmap_reader.get_status()
                                if status.get("status") == "error":
                                    error_details.append(f"Файл существует с размером {file_size} байт и содержит статус ошибки")
                                    error_details.append(f"Статус: {status}")
                                    error_msg = "\n".join(error_details)
                                    print(f"[StreamingAPI] ОШИБКА: {error_msg}", flush=True)
                                    raise RuntimeError(error_msg)
                            except (ValueError, FileNotFoundError):
                                pass  # Игнорируем ошибки чтения, продолжаем с обычной обработкой
                        error_details.append(f"Файл существует, но размер: {file_size} байт (возможно, демон не смог инициализировать mmap)")
                    else:
                        error_details.append("Файл не существует")
                        mmap_dir = os.path.dirname(mmap_file_path)
                        if os.path.exists(mmap_dir):
                            error_details.append(f"Директория существует: {mmap_dir}")
                            # Проверяем другие файлы в директории
                            try:
                                other_files = os.listdir(mmap_dir)
                                if other_files:
                                    error_details.append(f"В директории есть другие файлы: {other_files[:5]}")
                            except:
                                pass
                        else:
                            error_details.append(f"Директория не существует: {mmap_dir}")
                    
                    error_details.append("Возможные причины:")
                    error_details.append("1. Демон не запущен или не обрабатывает запрос")
                    error_details.append("2. Аватар не был подготовлен (frame_list_cycle пуст)")
                    error_details.append("3. Ошибка при создании mmap файла (недостаточно памяти или прав доступа)")
                    error_details.append("4. Демон обрабатывает другой запрос и еще не дошел до этого")
                    
                    error_msg = "\n".join(error_details)
                    print(f"[StreamingAPI] ОШИБКА: {error_msg}", flush=True)
                    raise RuntimeError(error_msg)
                
                last_frame_index = -1
                max_wait_time = 300
                start_time = time.time()
                
                try:
                    frame_batch = []
                    batch_size_send = 8
                    
                    while True:
                        if stop_flag.is_set():
                            break
                        
                        if time.time() - start_time > max_wait_time:
                            break
                        
                        status = mmap_reader.get_status()
                        
                        current_frame_index = last_frame_index + 1
                        max_frames_to_read = status.get("frames_generated", 0)
                        
                        while current_frame_index < max_frames_to_read:
                            try:
                                frame_array = mmap_reader.read_frame(current_frame_index)
                                if frame_array is not None:
                                    frame_batch.append(frame_array)
                                    last_frame_index = current_frame_index
                                    current_frame_index += 1
                                    
                                    if len(frame_batch) >= batch_size_send:
                                        video_track.add_frames_batch(frame_batch)
                                        frame_batch = []
                                else:
                                    break
                            except Exception as e:
                                break
                        
                        if status.get("status") == "completed":
                            if frame_batch:
                                video_track.add_frames_batch(frame_batch)
                            
                            while current_frame_index < status.get("total_frames", 0):
                                try:
                                    frame_array = mmap_reader.read_frame(current_frame_index)
                                    if frame_array is not None:
                                        video_track.add_frame(frame_array)
                                        last_frame_index = current_frame_index
                                    else:
                                        break
                                except:
                                    break
                                current_frame_index += 1
                            break
                        elif status.get("status") in ["stopped", "error"]:
                            if frame_batch:
                                video_track.add_frames_batch(frame_batch)
                            break
                        
                        if len(frame_batch) > 0 and (time.time() - start_time) % 0.1 < 0.01:
                            video_track.add_frames_batch(frame_batch)
                            frame_batch = []
                        
                        time.sleep(0.01)
                finally:
                    if mmap_reader:
                        mmap_reader.close()
                        try:
                            mmap_reader.cleanup()
                        except Exception as e:
                            pass
                
                with video_track.generation_completed_lock:
                    video_track.generation_completed = True
            except Exception as e:
                traceback.print_exc()
            finally:
                with generation_lock:
                    if task_id in active_generations:
                        del active_generations[task_id]
        
        new_inference_thread = threading.Thread(target=run_inference, daemon=True)
        new_inference_thread.start()
        
        with generation_lock:
            active_generations[task_id] = {
                "video_track": video_track,
                "audio_track": audio_track,
                "inference_thread": new_inference_thread,
                "stop_flag": stop_flag,
                "pc": pc,
                "task_id": task_id,
                "video_path": video_path,
                "audio_path": new_audio_path,
                "version": version,
                "batch_size": batch_size,
                "fps": fps
            }
        
        return {
            "status": "ok",
            "message": f"Аудио файл загружен и генерация для task_id={task_id} запущена",
            "task_id": task_id,
            "audio_path": new_audio_path,
            "audio_size": len(content)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Проверка здоровья сервиса"""
    return {
        "status": "ok",
        "webrtc_available": WEBRTC_AVAILABLE,
        "active_connections": len(active_connections),
        "active_generations": len(active_generations)
    }


if __name__ == "__main__":
    import argparse
    
    default_version = os.getenv("MUSETALK_VERSION", "v1.5")
    default_mode = os.getenv("MUSETALK_MODE", "realtime")
    default_gpu_id = int(os.getenv("MUSETALK_GPU_ID", "0"))
    default_use_float16 = os.getenv("MUSETALK_USE_FLOAT16", "true").lower() == "true"
    
    parser = argparse.ArgumentParser(description="WebRTC Streaming API для MuseTalk")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Хост для сервера")
    parser.add_argument("--port", type=int, default=8009, help="Порт для сервера")
    parser.add_argument("--version", type=str, default=default_version, help="Версия модели (v1.0 или v1.5)")
    parser.add_argument("--mode", type=str, default=default_mode, help="Режим работы (normal или realtime)")
    parser.add_argument("--gpu_id", type=int, default=default_gpu_id, help="ID GPU")
    parser.add_argument("--use_float16", action="store_true", help="Использовать float16")
    
    args = parser.parse_args()
    
    if not args.use_float16:
        args.use_float16 = default_use_float16
    
    if not WEBRTC_AVAILABLE:
        print("[StreamingAPI] ОШИБКА: aiortc не установлен!")
        sys.exit(1)
    
    if args.version not in ["v1.0", "v1.5", "v15"]:
        print("[StreamingAPI] ОШИБКА: Неверная версия. Используйте v1.0 или v1.5")
        sys.exit(1)
    
    if args.mode not in ["normal", "realtime"]:
        print("[StreamingAPI] ОШИБКА: Неверный режим. Используйте normal или realtime")
        sys.exit(1)
    
    print("[StreamingAPI] WebRTC API работает через файловую систему с демоном.")
    
    version_for_check = args.version
    if version_for_check == "v15":
        version_for_check = "v1.5"
    
    print(f"[StreamingAPI] Проверка демон-сервиса (версия: {version_for_check}, режим: {args.mode})...")
    is_running, message = check_daemon_service(version=version_for_check, mode=args.mode)
    
    if not is_running:
        print("")
        print("=" * 50)
        print("ОШИБКА: Демон-сервис не запущен!")
        print("=" * 50)
        print(f"Причина: {message}")
        print("")
        print("Запустите демон-сервис перед запуском API:")
        print(f"  sh start_model_service.sh {version_for_check} {args.mode} {args.gpu_id}")
        print("")
        sys.exit(1)
    
    print(f"[StreamingAPI] ✓ Демон-сервис запущен (PID: {message})")
    print("")
    
    print(f"[StreamingAPI] Запуск WebRTC сервера на {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)