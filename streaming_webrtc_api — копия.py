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

from dotenv import load_dotenv
load_dotenv()  # Загружает переменные из .env файла

# Добавляем путь к скриптам
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scripts.model_daemon_service import ModelDaemonService

try:
    from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack, AudioStreamTrack, RTCConfiguration, RTCIceServer, MediaStreamError
    from aiortc.contrib.media import MediaPlayer, MediaRelay
    from av import AudioFrame
    WEBRTC_AVAILABLE = True
except ImportError:
    WEBRTC_AVAILABLE = False
    print("[StreamingAPI] aiortc не установлен. Установите: pip install aiortc")


class MediaClock:
    """Общий клок для синхронизации аудио и видео"""
    
    def __init__(self):
        self.t0 = None
    
    def start(self):
        """Стартует клок (вызывается аудио-треком при первой отправке кадра)"""
        if self.t0 is None:
            self.t0 = time.time()
            print(f"[MediaClock] Клок запущен: t0={self.t0}")
    
    def now(self) -> float:
        """Возвращает текущее время воспроизведения в секундах"""
        if self.t0 is None:
            return 0.0
        return time.time() - self.t0

app = FastAPI()

# Добавляем CORS middleware для разрешения запросов с разных источников
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Для теста разрешаем все источники
    # Для продакшена укажите конкретные домены:
    # allow_origins=["http://localhost:5501", "http://127.0.0.1:5501", "http://51.250.22.55:8002"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Глобальный экземпляр сервиса
daemon_service = None

# Управление активными WebRTC соединениями
active_connections = {}

# Управление активными генерациями (по task_id)
# Структура: {task_id: {"video_track": VideoStreamGenerator, "inference_thread": Thread, "stop_flag": threading.Event, "pc": RTCPeerConnection}}
active_generations = {}
generation_lock = threading.Lock()

# Управление информацией о соединениях по task_id (сохраняется даже после завершения генерации)
# Структура: {task_id: {"pc": RTCPeerConnection, "video_track": VideoStreamGenerator, "audio_track": AudioStreamTrack, 
#                        "video_path": str, "audio_path": str, "version": str, "batch_size": int, "fps": int}}
connection_info_by_task = {}
connection_info_lock = threading.Lock()


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
        
        # Заполняем данные нулями (тишина)
        # Для s16 стерео: samples * 2 канала * 2 байта на сэмпл
        import numpy as np
        silence_data = np.zeros(samples * 2, dtype=np.int16)
        silence.planes[0].update(silence_data.tobytes())
        
        # Логируем периодически для диагностики заикания
        if hasattr(self, '_recv_call_count') and (self._recv_call_count <= 5 or self._recv_call_count % 200 == 0):
            frame_duration = samples / self.sample_rate if samples > 0 else 0
            print(f"[DelayedMediaPlayerTrack] Создан кадр тишины: samples={samples}, pts={silence.pts}, duration={frame_duration*1000:.2f}ms, silence_frames={self.silence_frames_sent}")
        
        self.silence_frames_sent += 1
        return silence
        
    async def recv(self):
        """Получает следующий аудио кадр с задержкой до первого реального кадра"""
        # Логируем первые несколько вызовов для диагностики
        if not hasattr(self, '_recv_call_count'):
            self._recv_call_count = 0
        self._recv_call_count += 1
        if self._recv_call_count <= 5 or self._recv_call_count % 100 == 0:
            print(f"[DelayedMediaPlayerTrack] recv вызван #{self._recv_call_count}, audio_started={self.audio_started}")
        
        # Проверяем, был ли отправлен первый реальный кадр
        if not self.audio_started:
            if self.video_track:
                with self.video_track.first_real_frame_lock:
                    first_real_sent = self.video_track.first_real_frame_sent
                
                if self._recv_call_count <= 5 or self._recv_call_count % 100 == 0:
                    print(f"[DelayedMediaPlayerTrack] Проверка флага: first_real_sent={first_real_sent}, audio_started={self.audio_started}")
                
                if first_real_sent:
                    self.audio_started = True
                    # Для DelayedMediaPlayerTrack MediaPlayer сам управляет PTS, но нужно убедиться, что он начинается с правильного времени
                    # Сбрасываем счетчик тишины, чтобы PTS начинался с правильного значения
                    # Синхронизируем с видео кадром
                    video_frame_count = self.video_track.frame_count if self.video_track else 0
                    video_pts_90k = int(video_frame_count * 90000 / (self.video_track.fps if self.video_track else 25))
                    # Конвертируем из 90kHz в сэмплы аудио (48kHz)
                    audio_pts_samples = int(video_pts_90k * self.sample_rate / 90000)
                    # ВАЖНО: Начинаем с нулевого PTS для плавного старта
                    # MediaPlayer сам управляет PTS, но мы сбрасываем счетчик тишины
                    self.silence_frames_sent = 0  # Начинаем с нуля для плавного старта
                    print(f"[DelayedMediaPlayerTrack] ✓ Аудио синхронизировано с видео (первый реальный кадр отправлен, вызов #{self._recv_call_count}, video_frame={video_frame_count}, silence_frames_reset=0)")
                else:
                    # Возвращаем тишину до отправки первого реального кадра
                    # Логируем периодически для диагностики
                    if self.silence_frames_sent % 100 == 0 or self._recv_call_count <= 5:  # Каждые 100 кадров тишины или первые 5 вызовов
                        print(f"[DelayedMediaPlayerTrack] Ожидание первого реального кадра (first_real_frame_sent={first_real_sent}, silence_frames={self.silence_frames_sent}, вызов #{self._recv_call_count})")
                    await asyncio.sleep(0.01)
                    return self._create_silence_frame(480)
            else:
                if self._recv_call_count <= 5:
                    print(f"[DelayedMediaPlayerTrack] ⚠️ video_track отсутствует!")
                await asyncio.sleep(0.01)
                return self._create_silence_frame(480)
        
        # После начала воспроизведения используем оригинальный трек
        if self.media_player_ended:
            # MediaPlayer закончил воспроизведение, возвращаем тишину без логирования
            return self._create_silence_frame(480)
        
        if self.media_player and self.media_player.audio:
            try:
                frame = await self.media_player.audio.recv()
                # Сбрасываем счетчик ошибок при успешном получении кадра
                self.error_count = 0
                
                # ВАЖНО: Копируем кадр для безопасной отправки и правильной синхронизации
                # MediaPlayer может возвращать кадры с неправильными PTS или проблемами синхронизации
                try:
                    import numpy as np
                    # Получаем данные как numpy array
                    audio_data = frame.to_ndarray()
                    actual_samples = frame.samples
                    
                    # Создаем новый кадр с правильными параметрами
                    audio_frame = AudioFrame(format=frame.format.name, 
                                            layout=frame.layout.name, 
                                            samples=actual_samples)
                    audio_frame.sample_rate = self.sample_rate
                    audio_frame.time_base = self.time_base
                    
                    # Копируем аудио данные
                    audio_frame.planes[0].update(audio_data.tobytes())
                    
                    # Устанавливаем PTS из оригинального кадра (MediaPlayer управляет PTS)
                    audio_frame.pts = frame.pts
                    
                    # Логируем периодически для диагностики
                    if self._recv_call_count <= 5 or self._recv_call_count % 100 == 0:
                        frame_duration = actual_samples / self.sample_rate if actual_samples > 0 else 0
                        print(f"[DelayedMediaPlayerTrack] Получен кадр из MediaPlayer (вызов #{self._recv_call_count}, samples={actual_samples}, pts={frame.pts}, duration={frame_duration*1000:.2f}ms)")
                    
                    return audio_frame
                except Exception as copy_error:
                    # Если не удалось скопировать, используем оригинальный кадр
                    if self._recv_call_count <= 5 or self._recv_call_count % 100 == 0:
                        print(f"[DelayedMediaPlayerTrack] ⚠️ Ошибка при копировании кадра: {copy_error}, используем оригинал")
                    return frame
            except Exception as e:
                self.error_count += 1
                # Проверяем, не закончилось ли воспроизведение
                error_str = str(e).lower()
                if 'end' in error_str or 'eof' in error_str or 'finished' in error_str:
                    # MediaPlayer закончил воспроизведение - это нормально
                    self.media_player_ended = True
                    # Логируем только один раз
                    if self.error_count == 1:
                        print(f"[DelayedMediaPlayerTrack] MediaPlayer закончил воспроизведение (нормально)")
                    return self._create_silence_frame(480)
                
                # Для других ошибок логируем реже (не чаще раза в секунду)
                current_time = time.time()
                if current_time - self.last_error_log_time >= 1.0:
                    print(f"[DelayedMediaPlayerTrack] Ошибка при получении кадра из MediaPlayer (ошибок: {self.error_count}): {e}")
                    self.last_error_log_time = current_time
                
                # Если слишком много ошибок подряд, считаем что MediaPlayer закончил
                if self.error_count >= 10:
                    self.media_player_ended = True
                    print(f"[DelayedMediaPlayerTrack] Слишком много ошибок ({self.error_count}), считаем MediaPlayer завершенным")
                
                return self._create_silence_frame(480)
        else:
            # Если трек недоступен, возвращаем тишину
            if self._recv_call_count <= 5 or self._recv_call_count % 100 == 0:
                print(f"[DelayedMediaPlayerTrack] ⚠️ MediaPlayer недоступен (media_player={self.media_player is not None}, audio={self.media_player.audio if self.media_player else None})")
            return self._create_silence_frame(480)
    
    def stop(self):
        """Останавливает трек"""
        if self.media_player:
            try:
                self.media_player.audio.stop()
            except:
                pass
        print(f"[DelayedMediaPlayerTrack] Аудио трек остановлен")


class SynchronizedAudioTrack(AudioStreamTrack):
    """Аудио трек - мастер-клок, стартует MediaClock и идет с естественным темпом"""
    
    def __init__(self, audio_path, clock: MediaClock, video_track=None, sample_rate=48000, channels=2):
        super().__init__()
        self.audio_path = audio_path
        self.clock = clock
        self.video_track = video_track  # Ссылка на видео трек для проверки первого кадра
        self.sample_rate = sample_rate
        self.channels = channels
        self.running = True
        self.audio_started = False  # Флаг: началось ли воспроизведение аудио
        self.min_frames_before_start = 5  # Минимум кадров перед началом аудио
        
        # Открываем аудио файл
        import av
        self.container = av.open(audio_path)
        self.stream = self.container.streams.audio[0]
        
        # Создаем resampler для преобразования в нужный формат
        target_layout = 'stereo' if channels == 2 else 'mono'
        self.resampler = av.AudioResampler(
            format='s16',
            layout=target_layout,
            rate=sample_rate
        )
        
        # Итератор для декодирования аудио
        self._audio_iter = self.container.decode(self.stream)
        
        # Позиция для PTS
        self.audio_position = 0
        
        # Time base для аудио
        from fractions import Fraction
        self.time_base = Fraction(1, sample_rate)
        
        print(f"[SynchronizedAudioTrack] Инициализация: audio_path={audio_path}, sample_rate={sample_rate}, channels={channels}, video_track={video_track is not None}")
    
    def reset(self):
        """Сбрасывает состояние трека для перезапуска"""
        self.audio_started = False
        self.audio_position = 0
        # Пересоздаем итератор для декодирования аудио
        import av
        if self.container:
            try:
                self.container.close()
            except:
                pass
        self.container = av.open(self.audio_path)
        self.stream = self.container.streams.audio[0]
        self._audio_iter = self.container.decode(self.stream)
        # Сбрасываем счетчик вызовов
        if hasattr(self, '_recv_call_count'):
            self._recv_call_count = 0
        if hasattr(self, '_frame_count'):
            self._frame_count = 0
        print(f"[SynchronizedAudioTrack] Состояние сброшено для перезапуска")
    
    async def _load_audio_old(self):
        """Загружает аудио файл с фиксированным размером кадров для предотвращения заикания"""
        try:
            import av
            self.audio_container = av.open(self.audio_path)
            self.audio_stream = self.audio_container.streams.audio[0]
            
            # Получаем информацию об исходном аудио файле
            original_sample_rate = self.audio_stream.rate
            original_channels = self.audio_stream.channels
            original_duration = float(self.audio_container.duration) / av.time_base if self.audio_container.duration else None
            
            print(f"[SynchronizedAudioTrack] Исходный аудио файл: sample_rate={original_sample_rate}, channels={original_channels}, duration={original_duration:.2f}s" if original_duration else f"[SynchronizedAudioTrack] Исходный аудио файл: sample_rate={original_sample_rate}, channels={original_channels}")
            print(f"[SynchronizedAudioTrack] Целевой формат: sample_rate={self.sample_rate}, channels={self.channels}")
            
            # Создаем resampler для преобразования аудио в нужный формат
            target_layout = 'stereo' if self.channels == 2 else 'mono'
            resampler = av.AudioResampler(
                format='s16',
                layout=target_layout,
                rate=self.sample_rate
            )
            
            # Фиксированный размер кадра для WebRTC (10ms при 48kHz)
            fixed_frame_size = 480  # 480 samples = 10ms при 48kHz
            
            print(f"[SynchronizedAudioTrack] Создан resampler: format=s16, layout={target_layout}, rate={self.sample_rate}")
            print(f"[SynchronizedAudioTrack] Используется фиксированный размер кадра: {fixed_frame_size} samples")
            
            # Вычисляем ожидаемую длительность после ресемплинга
            if original_duration:
                # Длительность не должна измениться при ресемплинге (только частота дискретизации)
                expected_duration = original_duration
                print(f"[SynchronizedAudioTrack] Ожидаемая длительность после ресемплинга: {expected_duration:.2f}s (не должна измениться)")
            
            # Буфер для накопления сэмплов
            sample_buffer = None
            
            # Читаем все аудио сэмплы
            frame_count = 0
            for frame in self.audio_container.decode(audio=0):
                if frame:
                    frame_count += 1
                    # Логируем первый кадр для диагностики
                    if frame_count == 1:
                        print(f"[SynchronizedAudioTrack] Первый кадр: format={frame.format.name}, sample_rate={frame.sample_rate}, layout={frame.layout.name}, samples={frame.samples}")
                    
                    # Конвертируем в нужный формат (48kHz, стерео, s16) через resampler
                    try:
                        # Используем resampler для преобразования кадра
                        resampled_frames = resampler.resample(frame)
                        
                        # resampler может вернуть несколько кадров или один кадр
                        if resampled_frames:
                            for resampled in resampled_frames:
                                # Проверяем результат реформатирования
                                if resampled.format.name != 's16':
                                    print(f"[SynchronizedAudioTrack] ⚠️ Предупреждение: формат после реформатирования {resampled.format.name}, ожидался s16")
                                if resampled.sample_rate != self.sample_rate:
                                    print(f"[SynchronizedAudioTrack] ⚠️ Предупреждение: sample_rate после реформатирования {resampled.sample_rate}, ожидался {self.sample_rate}")
                                
                                # Получаем данные кадра
                                audio_data = resampled.to_ndarray()
                                
                                # ВАЖНО: to_ndarray() возвращает данные в формате (samples, channels)
                                # Для стерео это (samples, 2), для моно (samples,)
                                # Нужно преобразовать в interleaved формат для planes[0]
                                if len(audio_data.shape) == 2:
                                    # Стерео: преобразуем (samples, 2) в interleaved (samples*2,)
                                    # Используем reshape для правильного порядка
                                    audio_data = audio_data.flatten('C')  # C-order: row-major (interleaved)
                                elif len(audio_data.shape) == 1:
                                    # Моно: уже правильный формат
                                    pass
                                else:
                                    print(f"[SynchronizedAudioTrack] ⚠️ Неожиданная форма audio_data: {audio_data.shape}")
                                    audio_data = audio_data.flatten('C')
                                
                                # Добавляем к буферу
                                if sample_buffer is None:
                                    sample_buffer = audio_data
                                else:
                                    # Объединяем с существующим буфером
                                    sample_buffer = np.concatenate([sample_buffer, audio_data], axis=0)
                                
                                # Для стерео: fixed_frame_size samples = fixed_frame_size * 2 элементов в буфере
                                # Для моно: fixed_frame_size samples = fixed_frame_size элементов в буфере
                                samples_per_frame = fixed_frame_size * self.channels
                                
                                # Разбиваем буфер на кадры фиксированного размера
                                while sample_buffer.shape[0] >= samples_per_frame:
                                    # Берем кадр фиксированного размера
                                    frame_data = sample_buffer[:samples_per_frame]
                                    sample_buffer = sample_buffer[samples_per_frame:]
                                    
                                    # Создаем новый кадр с фиксированным размером
                                    fixed_frame = AudioFrame(format='s16', 
                                                            layout=target_layout, 
                                                            samples=fixed_frame_size)
                                    fixed_frame.sample_rate = self.sample_rate
                                    fixed_frame.time_base = Fraction(1, self.sample_rate)
                                    
                                    # Копируем данные (уже в правильном interleaved формате)
                                    fixed_frame.planes[0].update(frame_data.tobytes())
                                    
                                    self.audio_samples.append(fixed_frame)
                        else:
                            print(f"[SynchronizedAudioTrack] ⚠️ Resampler не вернул кадры для кадра {frame_count}")
                            
                    except Exception as e:
                        print(f"[SynchronizedAudioTrack] Ошибка при реформатировании кадра {frame_count}: {e}")
                        import traceback
                        traceback.print_exc()
                        # Пробуем без реформатирования, если формат уже правильный
                        if frame.format.name == 's16' and frame.sample_rate == self.sample_rate:
                            # Проверяем layout
                            frame_layout = frame.layout.name if hasattr(frame.layout, 'name') else str(frame.layout)
                            if (self.channels == 2 and 'stereo' in frame_layout.lower()) or \
                               (self.channels == 1 and 'mono' in frame_layout.lower()):
                                print(f"[SynchronizedAudioTrack] Используем кадр без реформатирования (формат уже правильный)")
                                # Также разбиваем на фиксированный размер
                                audio_data = frame.to_ndarray()
                                
                                # Преобразуем в interleaved формат
                                if len(audio_data.shape) == 2:
                                    audio_data = audio_data.flatten('C')
                                elif len(audio_data.shape) == 1:
                                    pass
                                else:
                                    audio_data = audio_data.flatten('C')
                                
                                if sample_buffer is None:
                                    sample_buffer = audio_data
                                else:
                                    sample_buffer = np.concatenate([sample_buffer, audio_data], axis=0)
                                
                                samples_per_frame = fixed_frame_size * self.channels
                                while sample_buffer.shape[0] >= samples_per_frame:
                                    frame_data = sample_buffer[:samples_per_frame]
                                    sample_buffer = sample_buffer[samples_per_frame:]
                                    
                                    fixed_frame = AudioFrame(format='s16', 
                                                            layout=target_layout, 
                                                            samples=fixed_frame_size)
                                    fixed_frame.sample_rate = self.sample_rate
                                    fixed_frame.time_base = Fraction(1, self.sample_rate)
                                    fixed_frame.planes[0].update(frame_data.tobytes())
                                    self.audio_samples.append(fixed_frame)
                            else:
                                print(f"[SynchronizedAudioTrack] ⚠️ Пропускаем кадр {frame_count}: неправильный layout ({frame_layout})")
                        else:
                            print(f"[SynchronizedAudioTrack] ⚠️ Пропускаем кадр {frame_count}: неправильный формат (format={frame.format.name}, rate={frame.sample_rate})")
            
            # Обрабатываем остаток буфера (дополняем нулями до фиксированного размера)
            if sample_buffer is not None and sample_buffer.shape[0] > 0:
                remaining_elements = sample_buffer.shape[0]
                samples_per_frame = fixed_frame_size * self.channels
                
                if remaining_elements < samples_per_frame:
                    # Дополняем нулями до нужного размера
                    padding = np.zeros(samples_per_frame - remaining_elements, dtype=sample_buffer.dtype)
                    sample_buffer = np.concatenate([sample_buffer, padding], axis=0)
                    print(f"[SynchronizedAudioTrack] Остаток буфера дополнен нулями: {remaining_elements} -> {samples_per_frame} элементов")
                
                # Создаем последний кадр
                fixed_frame = AudioFrame(format='s16', 
                                        layout=target_layout, 
                                        samples=fixed_frame_size)
                fixed_frame.sample_rate = self.sample_rate
                fixed_frame.time_base = Fraction(1, self.sample_rate)
                fixed_frame.planes[0].update(sample_buffer.tobytes())
                self.audio_samples.append(fixed_frame)
                print(f"[SynchronizedAudioTrack] Добавлен последний кадр с остатком ({remaining_elements} элементов, дополнено до {samples_per_frame})")
            
            self.audio_loaded = True
            print(f"[SynchronizedAudioTrack] Аудио загружено: {len(self.audio_samples)} кадров")
            if len(self.audio_samples) > 0:
                first_frame = self.audio_samples[0]
                layout_str = first_frame.layout.name if hasattr(first_frame.layout, 'name') else str(first_frame.layout)
                print(f"[SynchronizedAudioTrack] Первый кадр после обработки: samples={first_frame.samples}, format={first_frame.format.name}, sample_rate={first_frame.sample_rate}, layout={layout_str}")
                
                # Вычисляем фактическую длительность загруженного аудио
                total_samples = sum(frame.samples for frame in self.audio_samples)
                actual_duration = total_samples / self.sample_rate
                print(f"[SynchronizedAudioTrack] Фактическая длительность загруженного аудио: {actual_duration:.2f}s ({total_samples} samples при {self.sample_rate} Hz)")
                
                # Проверяем размеры кадров для диагностики проблем с заиканием
                sample_sizes = [frame.samples for frame in self.audio_samples]
                min_samples = min(sample_sizes)
                max_samples = max(sample_sizes)
                avg_samples = sum(sample_sizes) / len(sample_sizes)
                
                if min_samples != max_samples:
                    print(f"[SynchronizedAudioTrack] ⚠️ ВНИМАНИЕ: Размеры кадров различаются! min={min_samples}, max={max_samples}, avg={avg_samples:.1f}")
                    print(f"[SynchronizedAudioTrack] Это может вызывать заикание. Рекомендуется использовать resampler с фиксированным размером кадра.")
                else:
                    print(f"[SynchronizedAudioTrack] ✓ Все кадры имеют одинаковый размер: {min_samples} samples")
            else:
                print(f"[SynchronizedAudioTrack] ⚠️ ВНИМАНИЕ: Не загружено ни одного аудио кадра!")
        except Exception as e:
            print(f"[SynchronizedAudioTrack] Ошибка при загрузке аудио: {e}")
            import traceback
            traceback.print_exc()
            self.running = False
    
    def _create_silence_frame(self, samples=480):
        """Создает кадр тишины с нулевыми данными"""
        # Создаем тишину с правильными параметрами
        silence = AudioFrame(format='s16', layout='stereo', samples=samples)
        silence.sample_rate = self.sample_rate
        silence.time_base = self.time_base
        silence.pts = self.audio_position
        
        # Заполняем данные нулями (тишина)
        # Для s16 стерео: samples * 2 канала * 2 байта на сэмпл
        import numpy as np
        silence_data = np.zeros(samples * 2, dtype=np.int16)
        silence.planes[0].update(silence_data.tobytes())
        
        # Обновляем позицию для следующего кадра
        self.audio_position += samples
        
        return silence
    
    def get_t_playback(self):
        """Возвращает текущее время воспроизведения в секундах (t_playback)"""
        if not self.audio_started:
            return 0.0
        # t_playback = количество отправленных сэмплов / sample_rate
        t_playback = self.audio_position / self.sample_rate
        # Логируем периодически для диагностики
        if hasattr(self, '_t_playback_call_count'):
            self._t_playback_call_count += 1
        else:
            self._t_playback_call_count = 1
        
        if self._t_playback_call_count % 100 == 0:
            print(f"[SynchronizedAudioTrack] get_t_playback: audio_position={self.audio_position}, sample_rate={self.sample_rate}, t_playback={t_playback:.3f}s, current_frame_index={self.current_frame_index}/{len(self.audio_samples) if self.audio_samples else 0}")
        
        return t_playback
    
    async def recv(self):
        """Получает следующий аудио кадр - первая отправка кадра стартует клок"""
        if not self.running:
            raise MediaStreamError("Audio track stopped")
        
        # Логируем первые несколько вызовов для диагностики
        if not hasattr(self, '_recv_call_count'):
            self._recv_call_count = 0
        self._recv_call_count += 1
        if self._recv_call_count <= 5 or self._recv_call_count % 100 == 0:
            print(f"[SynchronizedAudioTrack] recv вызван #{self._recv_call_count}, audio_started={self.audio_started}")
        
        # Проверяем, нужно ли ждать первых кадров видео перед началом аудио
        if not self.audio_started:
            if self.video_track:
                # Проверяем, был ли отправлен первый реальный кадр
                with self.video_track.first_real_frame_lock:
                    first_real_sent = self.video_track.first_real_frame_sent
                
                # Проверяем количество отправленных реальных кадров
                real_frames_count = self.video_track.real_frames_generated if hasattr(self.video_track, 'real_frames_generated') else 0
                
                if first_real_sent and real_frames_count >= self.min_frames_before_start:
                    self.audio_started = True
                    # Первая отправка кадра — стартуем клок
                    clock_was_started = self.clock.t0 is not None
                    self.clock.start()
                    if not clock_was_started:
                        print(f"[SynchronizedAudioTrack] ✓ Аудио запущено (отправлено {real_frames_count} реальных кадров, минимум {self.min_frames_before_start}), MediaClock запущен")
                else:
                    # Возвращаем тишину до отправки достаточного количества кадров
                    if self._recv_call_count % 100 == 0:
                        print(f"[SynchronizedAudioTrack] Ожидание {self.min_frames_before_start} реальных кадров (отправлено: {real_frames_count}, first_real_sent={first_real_sent})")
                    await asyncio.sleep(0.01)
                    silence = self._create_silence_frame(480)
                    return silence
            else:
                # Если нет видео трека, начинаем сразу
                self.audio_started = True
                clock_was_started = self.clock.t0 is not None
                self.clock.start()
                if not clock_was_started:
                    print(f"[SynchronizedAudioTrack] ✓ Аудио запущено (нет video_track), MediaClock запущен")
        else:
            # Аудио уже запущено, просто стартуем клок если еще не запущен
            clock_was_started = self.clock.t0 is not None
            self.clock.start()
        
        # Получаем следующий кадр из итератора
        try:
            frame = next(self._audio_iter, None)
        except StopIteration:
            frame = None
        except Exception as e:
            print(f"[SynchronizedAudioTrack] Ошибка при декодировании аудио: {e}")
            import traceback
            traceback.print_exc()
            frame = None
        
        if frame is None:
            # Аудио закончилось — можно сделать EOS или зациклить
            if not hasattr(self, '_audio_ended_logged'):
                print(f"[SynchronizedAudioTrack] Аудио закончилось, отправляем тишину")
                self._audio_ended_logged = True
            await asyncio.sleep(0.02)
            # Вместо ошибки возвращаем тишину
            silence = self._create_silence_frame(480)
            return silence
        
        # Ресемплируем кадр в нужный формат
        try:
            resampled_frames = self.resampler.resample(frame)
        except Exception as e:
            print(f"[SynchronizedAudioTrack] Ошибка при ресемплинге: {e}")
            import traceback
            traceback.print_exc()
            await asyncio.sleep(0.01)
            return await self.recv()
        
        if not resampled_frames:
            # Если resampler не вернул кадры, пропускаем
            await asyncio.sleep(0.01)
            return await self.recv()
        
        # Берем первый ресемплированный кадр (обычно resampler возвращает один кадр)
        audio_frame = resampled_frames[0]
        
        # Устанавливаем правильный PTS и time_base
        audio_frame.time_base = self.time_base
        audio_frame.pts = self.audio_position
        audio_frame.sample_rate = self.sample_rate
        
        # Вычисляем длительность кадра на основе количества сэмплов (правильный способ)
        frame_duration = audio_frame.samples / self.sample_rate
        
        # Обновляем позицию для следующего кадра
        self.audio_position += audio_frame.samples
        
        # Логируем периодически для диагностики
        if not hasattr(self, '_frame_count'):
            self._frame_count = 0
        self._frame_count += 1
        if self._frame_count % 100 == 0 or self._frame_count <= 5:
            print(f"[SynchronizedAudioTrack] Отправлен аудио кадр #{self._frame_count}, samples={audio_frame.samples}, pts={audio_frame.pts}, duration={frame_duration*1000:.2f}ms, sample_rate={audio_frame.sample_rate}")
        
        # aiortc само выровняет по времени с помощью sleep внутри,
        # но мы должны примерно выдерживать реальное время:
        # Используем длительность на основе количества сэмплов (правильный способ)
        await asyncio.sleep(frame_duration)
        
        return audio_frame
    
    def stop(self):
        """Останавливает аудио трек"""
        self.running = False
        if self.audio_container:
            try:
                self.audio_container.close()
            except:
                pass
        print(f"[SynchronizedAudioTrack] Аудио трек остановлен")


class VideoStreamGenerator(VideoStreamTrack):
    """Генератор видео потока из кадров"""
    
    def __init__(self, fps=25, loop=None, video_width=640, video_height=480, video_path=None, clock: MediaClock = None):
        super().__init__()
        self.fps = fps
        self.clock = clock  # MediaClock для синхронизации
        self.frame_queue = asyncio.Queue(maxsize=100)  # Очередь кадров: (frame_array, logical_timestamp) или просто frame_array
        self.running = True
        self.frame_time = 1.0 / fps
        self.frame_count = 0
        self.loop = loop  # Event loop для добавления кадров из других потоков
        self.last_frame = None  # Последний отправленный кадр (VideoFrame или numpy array)
        self.last_frame_lock = threading.Lock()  # Блокировка для доступа к last_frame
        self.video_width = video_width
        self.video_height = video_height
        self.video_path = video_path
        self.original_video_frames = []  # Кадры из оригинального видео для тестовых кадров
        self.original_frame_index = 0
        self.first_real_frame_sent = False  # Флаг: отправлен ли первый реальный кадр
        self.first_real_frame_lock = threading.Lock()
        self.generation_completed = False
        self.generation_completed_lock = threading.Lock()
        self.real_frames_generated = 0
        self.real_frames_lock = threading.Lock()
        
        # Стандартный RTP clock для видео (90kHz)
        self.time_base = Fraction(1, 90000)
        self.pts_per_second = 90000
        
        # Метрики производительности для диагностики
        self.conversion_times = []  # Время конвертации кадров
        self.queue_wait_times = []  # Время ожидания в очереди
        self.last_metrics_log = time.time()  # Время последнего логирования метрик
        self.metrics_lock = threading.Lock()  # Блокировка для метрик
        
        print(f"[VideoStreamGenerator] Инициализирован: fps={fps}, loop={loop is not None}, размер={video_width}x{video_height}, video_path={video_path}, clock={clock is not None}")
        
        # Загружаем кадры из оригинального видео для тестовых кадров
        if video_path and os.path.exists(video_path):
            self._load_original_video_frames()
    
    def _load_original_video_frames(self):
        """Загружает все кадры из оригинального видео для использования в тестовых кадрах (с зацикливанием)"""
        try:
            video_cap = cv2.VideoCapture(self.video_path)
            if not video_cap.isOpened():
                print(f"[VideoStreamGenerator] ⚠️ Не удалось открыть видео для загрузки кадров: {self.video_path}")
                return
            
            # Получаем общее количество кадров в видео
            total_frames = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            video_fps = video_cap.get(cv2.CAP_PROP_FPS)
            
            print(f"[VideoStreamGenerator] Загрузка всех кадров из оригинального видео: {total_frames} кадров, {video_fps:.2f} fps")
            
            frame_count = 0
            
            # Загружаем все кадры из видео
            while True:
                ret, frame = video_cap.read()
                if not ret:
                    break
                
                # Конвертируем BGR в правильный формат и изменяем размер если нужно
                if frame.shape[1] != self.video_width or frame.shape[0] != self.video_height:
                    frame = cv2.resize(frame, (self.video_width, self.video_height))
                
                self.original_video_frames.append(frame.copy())
                frame_count += 1
                
                # Логируем прогресс каждые 100 кадров
                if frame_count % 100 == 0:
                    print(f"[VideoStreamGenerator] Загружено {frame_count} кадров из оригинального видео...")
            
            video_cap.release()
            
            if len(self.original_video_frames) > 0:
                print(f"[VideoStreamGenerator] ✓ Загружено {len(self.original_video_frames)} кадров из оригинального видео для тестовых кадров (будет зациклено)")
                # Сохраняем первый кадр как последний по умолчанию
                with self.last_frame_lock:
                    self.last_frame = self.original_video_frames[0].copy()
            else:
                print(f"[VideoStreamGenerator] ⚠️ Не удалось загрузить кадры из оригинального видео")
        except Exception as e:
            print(f"[VideoStreamGenerator] Ошибка при загрузке кадров из оригинального видео: {e}")
            import traceback
            traceback.print_exc()
    
    async def _send_test_frame(self):
        """Отправляет тестовый кадр из оригинального видео (или серый, если видео не загружено)"""
        # Тестовые кадры имеют timestamp = -1 (специальное значение, не синхронизируется с аудио)
        test_timestamp = -1.0
        if len(self.original_video_frames) > 0:
            # Используем кадр из оригинального видео
            test_frame = self.original_video_frames[self.original_frame_index % len(self.original_video_frames)].copy()
            self.original_frame_index += 1
            await self.frame_queue.put((test_frame, test_timestamp))
            # Сохраняем тестовый кадр как последний
            with self.last_frame_lock:
                self.last_frame = test_frame.copy()
                self.last_frame_timestamp = test_timestamp
            print(f"[VideoStreamGenerator] Тестовый кадр из оригинального видео добавлен (кадр #{self.original_frame_index-1}, размер: {self.video_width}x{self.video_height})")
        else:
            # Fallback: серый кадр, если видео не загружено
            test_frame = np.full((self.video_height, self.video_width, 3), 128, dtype=np.uint8)  # Серый кадр в BGR
            await self.frame_queue.put((test_frame, test_timestamp))
            # Сохраняем тестовый кадр как последний
            with self.last_frame_lock:
                self.last_frame = test_frame.copy()
                self.last_frame_timestamp = test_timestamp
            print(f"[VideoStreamGenerator] Тестовый кадр (серый) добавлен для поддержания соединения (размер: {self.video_width}x{self.video_height})")
    
    async def _play_original_video(self):
        """Воспроизводит оригинальное видео циклически, когда генерация не активна"""
        # Ждем загрузки кадров
        while len(self.original_video_frames) == 0 and self.running:
            await asyncio.sleep(0.1)
        
        if not self.running or len(self.original_video_frames) == 0:
            return
        
        frame_interval = 1.0 / self.fps  # Интервал между кадрами в секундах
        
        while self.running:
            try:
                # Проверяем, нужно ли воспроизводить оригинальное видео
                # Воспроизводим до начала генерации или после её завершения
                should_play = False
                
                with self.generation_completed_lock:
                    gen_completed = self.generation_completed
                
                with self.first_real_frame_lock:
                    first_sent = self.first_real_frame_sent
                
                # Воспроизводим если: генерация еще не началась ИЛИ генерация завершена
                if not first_sent or gen_completed:
                    should_play = True
                
                if should_play:
                    # Проверяем, не переполнена ли очередь
                    queue_size = self.frame_queue.qsize()
                    if not self.frame_queue.full():
                        # Берем следующий кадр из оригинального видео (с зацикливанием)
                        frame = self.original_video_frames[self.original_frame_index % len(self.original_video_frames)].copy()
                        self.original_frame_index += 1
                        
                        # Оригинальное видео имеет timestamp = -1 (не синхронизируется с аудио)
                        original_timestamp = -1.0
                        await self.frame_queue.put((frame, original_timestamp))
                        
                        # Сохраняем кадр как последний
                        with self.last_frame_lock:
                            self.last_frame = frame.copy()
                            self.last_frame_timestamp = original_timestamp
                        
                        if self.original_frame_index % (self.fps * 2) == 0:  # Логируем каждые 2 секунды
                            status = "до начала генерации" if not first_sent else "после завершения генерации"
                            print(f"[VideoStreamGenerator] Воспроизведение оригинального видео ({status}, кадр #{self.original_frame_index-1}, очередь: {queue_size})")
                    else:
                        # Очередь переполнена - если генерация завершена, очищаем очередь и добавляем кадр
                        if gen_completed:
                            # Очищаем очередь от старых кадров, чтобы освободить место для оригинального видео
                            cleared = 0
                            while not self.frame_queue.empty() and cleared < 10:  # Очищаем до 10 кадров
                                try:
                                    self.frame_queue.get_nowait()
                                    cleared += 1
                                except:
                                    break
                            if cleared > 0:
                                print(f"[VideoStreamGenerator] Очищено {cleared} кадров из переполненной очереди для оригинального видео")
                            
                            # Теперь пробуем добавить кадр снова
                            if not self.frame_queue.full():
                                frame = self.original_video_frames[self.original_frame_index % len(self.original_video_frames)].copy()
                                self.original_frame_index += 1
                                original_timestamp = -1.0
                                await self.frame_queue.put((frame, original_timestamp))
                                with self.last_frame_lock:
                                    self.last_frame = frame.copy()
                                    self.last_frame_timestamp = original_timestamp
                        else:
                            # Очередь переполнена - логируем периодически
                            if self.original_frame_index % (self.fps * 5) == 0:  # Каждые 5 секунд
                                print(f"[VideoStreamGenerator] Очередь переполнена ({queue_size}), пропускаем кадр оригинального видео")
                else:
                    # Не нужно воспроизводить - логируем периодически для диагностики
                    if self.original_frame_index % (self.fps * 10) == 0:  # Каждые 10 секунд
                        print(f"[VideoStreamGenerator] Воспроизведение оригинального видео приостановлено (first_sent={first_sent}, gen_completed={gen_completed})")
                
                # Ждем перед следующим кадром
                await asyncio.sleep(frame_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"[VideoStreamGenerator] Ошибка при воспроизведении оригинального видео: {e}")
                await asyncio.sleep(1.0)
        
        print(f"[VideoStreamGenerator] Задача воспроизведения оригинального видео остановлена")
    
    async def _start_keepalive(self):
        """Запускает задачу keepalive для периодической отправки последнего кадра"""
        # Ждем немного перед началом keepalive, чтобы дать время для получения первых кадров
        await asyncio.sleep(2.0)
        
        keepalive_interval = 0.5  # Отправляем keepalive каждые 0.5 секунды (2 раза в секунду)
        
        while self.running:
            try:
                await asyncio.sleep(keepalive_interval)
                
                if not self.running:
                    break
                
                # Проверяем, есть ли кадры в очереди
                queue_size = self.frame_queue.qsize()
                
                # Если очередь пуста, добавляем кадр для keepalive
                if queue_size == 0:
                    try:
                        # Проверяем, не переполнена ли очередь
                        if not self.frame_queue.full():
                            keepalive_frame = None
                            
                            # Приоритет 1: последний отправленный кадр
                            keepalive_timestamp = -1.0  # Keepalive кадры не синхронизируются
                            if self.last_frame is not None:
                                keepalive_frame = self.last_frame.copy()
                                keepalive_timestamp = self.last_frame_timestamp
                            # Приоритет 2: кадр из оригинального видео
                            elif len(self.original_video_frames) > 0:
                                keepalive_frame = self.original_video_frames[self.original_frame_index % len(self.original_video_frames)].copy()
                                self.original_frame_index += 1
                                keepalive_timestamp = -1.0
                            # Приоритет 3: серый кадр (fallback)
                            else:
                                keepalive_frame = np.full((self.video_height, self.video_width, 3), 128, dtype=np.uint8)
                                keepalive_timestamp = -1.0
                            
                            if keepalive_frame is not None:
                                await self.frame_queue.put((keepalive_frame, keepalive_timestamp))
                                if self.frame_count % 60 == 0:  # Логируем каждые 60 keepalive кадров
                                    frame_source = "последний кадр" if self.last_frame is not None else ("кадр из оригинального видео" if len(self.original_video_frames) > 0 else "серый кадр")
                                    print(f"[VideoStreamGenerator] Keepalive: отправлен {frame_source} (очередь пуста)")
                    except Exception as e:
                        if self.frame_count % 60 == 0:
                            print(f"[VideoStreamGenerator] Keepalive: ошибка при отправке кадра: {e}")
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"[VideoStreamGenerator] Keepalive: неожиданная ошибка: {e}")
                await asyncio.sleep(1.0)  # Ждем перед повтором
        
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
            
            # Вычисляем текущий FPS
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
        Это более эффективно, чем вызов add_frame для каждого кадра отдельно.
        
        Args:
            frame_arrays: Список numpy массивов кадров в формате BGR
        """
        if not self.running or not self.loop:
            return
        
        if not frame_arrays or len(frame_arrays) == 0:
            return
        
        valid_frames = []
        for frame_array in frame_arrays:
            if frame_array is None:
                continue
            
            if len(frame_array.shape) != 3 or frame_array.shape[2] != 3:
                continue
            
            # Нормализуем формат кадра
            if frame_array.dtype != np.uint8:
                if frame_array.max() > 1.0:
                    frame_array = (frame_array / 255.0).clip(0, 1)
                frame_array = (frame_array * 255).astype(np.uint8)
            
            valid_frames.append(frame_array.copy())
        
        if len(valid_frames) == 0:
            return
        
        # Добавляем все кадры в очередь одним асинхронным вызовом
        async def add_batch():
            added_count = 0
            start_frame_index = 0
            
            # Получаем начальный индекс для timestamp
            with self.real_frames_lock:
                start_frame_index = self.real_frames_generated
            
            for i, frame in enumerate(valid_frames):
                try:
                    # Проверяем размер очереди перед добавлением
                    if self.frame_queue.qsize() < self.frame_queue.maxsize:
                        # Вычисляем логический timestamp для кадра
                        frame_index = start_frame_index + i
                        logical_timestamp = frame_index / self.fps
                        
                        # Добавляем кадр с timestamp
                        frame_with_timestamp = (frame, logical_timestamp)
                        await self.frame_queue.put(frame_with_timestamp)
                        added_count += 1
                    else:
                        # Очередь полная - пропускаем остальные кадры
                        break
                except:
                    break
            
            # Обновляем счетчик реальных кадров
            with self.real_frames_lock:
                self.real_frames_generated += added_count
            
            # Отмечаем первый реальный кадр
            if added_count > 0:
                with self.first_real_frame_lock:
                    if not self.first_real_frame_sent:
                        self.first_real_frame_sent = True
                        first_timestamp = start_frame_index / self.fps
                        print(f"[VideoStreamGenerator] ✓ Первый реальный сгенерированный кадр добавлен в очередь (батч из {added_count} кадров, timestamp={first_timestamp:.3f}s)")
                        print(f"[VideoStreamGenerator] Флаг first_real_frame_sent установлен в True - аудио должно начаться")
                
                # Сохраняем последний кадр с его timestamp
                last_timestamp = (start_frame_index + added_count - 1) / self.fps
                with self.last_frame_lock:
                    self.last_frame = valid_frames[-1].copy()
                    self.last_frame_timestamp = last_timestamp
            
            if added_count > 0 and (start_frame_index % 30 == 0 or start_frame_index < 5):
                print(f"[VideoStreamGenerator] Батч из {added_count}/{len(valid_frames)} кадров добавлен в очередь (очередь: {self.frame_queue.qsize()}/{self.frame_queue.maxsize}, timestamp: {start_frame_index/self.fps:.3f}s-{last_timestamp:.3f}s)")
        
        try:
            asyncio.run_coroutine_threadsafe(add_batch(), self.loop)
        except Exception as e:
            print(f"[VideoStreamGenerator] Ошибка при добавлении батча кадров: {e}")
    
    def __repr__(self):
        return f"VideoStreamGenerator(fps={self.fps}, running={self.running}, queue_size={self.frame_queue.qsize()})"
        
    async def recv(self):
        """Получает следующий кадр для отправки - синхронизируется с MediaClock"""
        if not self.running:
            raise MediaStreamError("Video track stopped")
        
        # Целевое время следующего кадра на основе MediaClock
        now = self.clock.now() if self.clock else 0.0
        clock_started = (self.clock is not None and self.clock.t0 is not None)
        
        # Если клок не запущен, используем frame_count для PTS (монотонное увеличение)
        if not clock_started:
            target_pts = int(self.frame_count * self.pts_per_second / self.fps)
        else:
            target_pts = int(now * self.pts_per_second)
        
        # Пытаемся взять самый свежий кадр, который не слишком старый
        frame_array = None
        skipped_old_frames = 0
        
        # Пытаемся получить кадр из очереди (не ждем бесконечно, чтобы не блокировать)
        # Берем самый свежий подходящий кадр
        best_frame = None
        best_timestamp = None
        
        queue_size_before = self.frame_queue.qsize()
        
        while True:
            try:
                candidate = await asyncio.wait_for(self.frame_queue.get(), timeout=0.0)
                
                # Проверяем формат: может быть (frame, timestamp) или просто frame
                if isinstance(candidate, tuple) and len(candidate) == 2:
                    candidate_frame, candidate_timestamp = candidate
                else:
                    candidate_frame = candidate
                    candidate_timestamp = -1  # Старый формат без timestamp
                
                # Кадры с timestamp = -1 (тестовые/оригинальные/старый формат) используем как есть
                if candidate_timestamp < 0:
                    frame_array = candidate_frame
                    break
                
                # Если клок не запущен, используем любой кадр из очереди (не проверяем на старость)
                if not clock_started:
                    # Берем первый подходящий кадр и используем его сразу
                    frame_array = candidate_frame
                    queue_size_after = self.frame_queue.qsize()
                    if self.frame_count % 30 == 0 or self.frame_count <= 5:
                        print(f"[VideoStreamGenerator] recv: ✓ Использован кадр из очереди (клок не запущен, timestamp={candidate_timestamp:.3f}s, очередь было={queue_size_before}, стало={queue_size_after})")
                    break
                
                # Если клок запущен, проверяем синхронизацию
                # Кадры с timestamp = -1 (тестовые/оригинальные) используем как есть
                if candidate_timestamp < 0:
                    frame_array = candidate_frame
                    break
                
                # Проверяем, не слишком ли старый кадр (старше 1.0 секунды - более мягкий порог)
                # Если кадр не слишком старый, используем его
                age = now - candidate_timestamp
                if age > 1.0:
                    # Кадр слишком старый - пропускаем, берем следующий
                    skipped_old_frames += 1
                    if skipped_old_frames <= 3 or skipped_old_frames % 10 == 0:
                        print(f"[VideoStreamGenerator] recv: Пропущен старый кадр (timestamp={candidate_timestamp:.3f}s, now={now:.3f}s, age={age:.3f}s)")
                    continue
                else:
                    # Кадр подходит - сохраняем как лучший, но продолжаем искать более свежий
                    if best_frame is None or candidate_timestamp > best_timestamp:
                        best_frame = candidate_frame
                        best_timestamp = candidate_timestamp
                    # Если кадр свежий (в пределах 0.2 секунды) или это первый подходящий кадр, используем его сразу
                    if age <= 0.2 or best_frame == candidate_frame:
                        frame_array = candidate_frame
                        break
                    # Иначе продолжаем искать более свежий кадр (но не более 3 итераций)
                    if skipped_old_frames < 3:
                        continue
                    else:
                        # Используем лучший найденный кадр
                        frame_array = best_frame
                        break
                    
            except asyncio.TimeoutError:
                # Новых кадров нет — используем лучший найденный или последний
                if best_frame is not None:
                    frame_array = best_frame
                    if self.frame_count % 30 == 0:
                        print(f"[VideoStreamGenerator] recv: Использован лучший найденный кадр (timestamp={best_timestamp:.3f}s, очередь={queue_size_before})")
                break
        
        # Если не нашли кадр, используем последний отправленный
        if frame_array is None:
            with self.last_frame_lock:
                if self.last_frame is not None:
                    # last_frame должен быть numpy array
                    frame_array = self.last_frame.copy()
                    if self.frame_count % 30 == 0:
                        print(f"[VideoStreamGenerator] recv: Использован last_frame (очередь={queue_size_before}, клок запущен={clock_started})")
                else:
                    # Если нет последнего кадра, используем кадр из оригинального видео или серый
                    if len(self.original_video_frames) > 0:
                        frame_array = self.original_video_frames[self.original_frame_index % len(self.original_video_frames)].copy()
                        self.original_frame_index += 1
                        if self.frame_count % 60 == 0:
                            print(f"[VideoStreamGenerator] recv: Нет кадров, используем оригинальное видео (кадр #{self.original_frame_index-1}, очередь={queue_size_before})")
                    else:
                        frame_array = np.full((self.video_height, self.video_width, 3), 128, dtype=np.uint8)
                        if self.frame_count % 60 == 0:
                            print(f"[VideoStreamGenerator] recv: Нет кадров, используем серый кадр (очередь={queue_size_before})")
        
        # Сохраняем кадр как последний (всегда numpy array)
        with self.last_frame_lock:
            self.last_frame = frame_array.copy()
            if self.frame_count % 30 == 0 or self.frame_count <= 5:
                print(f"[VideoStreamGenerator] recv: Кадр сохранен как last_frame (размер={frame_array.shape if hasattr(frame_array, 'shape') else 'unknown'})")
        
        # Проверяем размер кадра - WebRTC может требовать четные размеры
        h, w = frame_array.shape[:2]
        if h % 2 != 0 or w % 2 != 0:
            h = (h // 2) * 2
            w = (w // 2) * 2
            frame_array = cv2.resize(frame_array, (w, h))
        
        # Конвертируем numpy array в VideoFrame
        try:
            frame = VideoFrame.from_ndarray(frame_array, format="bgr24")
            frame = frame.reformat(format="yuv420p")
        except Exception as e:
            print(f"[VideoStreamGenerator] recv: Ошибка при создании VideoFrame: {e}")
            # Fallback: серый кадр
            frame = VideoFrame.from_ndarray(
                np.full((self.video_height, self.video_width, 3), 128, dtype=np.uint8),
                format="bgr24"
            )
            frame = frame.reformat(format="yuv420p")
        
        # Устанавливаем PTS на основе MediaClock или frame_count
        # Гарантируем монотонное увеличение PTS
        if not hasattr(self, 'last_pts'):
            self.last_pts = None
        
        if self.last_pts is not None and target_pts < self.last_pts:
            # Если PTS меньше предыдущего, используем предыдущий + минимальный шаг
            min_step = int(self.pts_per_second / self.fps)
            target_pts = self.last_pts + min_step
        
        self.last_pts = target_pts
        frame.pts = target_pts
        frame.time_base = self.time_base
        
        # Задаем темп выдачи кадров (реальный FPS будет <= заданного)
        await asyncio.sleep(1 / self.fps)
        
        self.frame_count += 1
        if self.frame_count % 30 == 0 or self.frame_count <= 5:
            queue_size = self.frame_queue.qsize()
            print(f"[VideoStreamGenerator] recv: Отправлен кадр #{self.frame_count}, pts={target_pts}, clock.now()={now:.3f}s, очередь={queue_size}, skipped_old={skipped_old_frames}")
        
        return frame
    
    def add_frame(self, frame_array):
        """Добавляет кадр в очередь (вызывается из другого потока)"""
        if not self.running:
            print(f"[VideoStreamGenerator] Поток остановлен, кадр не добавлен")
            return
        
        if not self.loop:
            print(f"[VideoStreamGenerator] Event loop не установлен, кадр не добавлен")
            return
        
        # Убеждаемся, что кадр в правильном формате (BGR для OpenCV)
        if frame_array is None:
            print(f"[VideoStreamGenerator] Получен None вместо кадра")
            return
        
        # Проверяем формат кадра
        if len(frame_array.shape) != 3 or frame_array.shape[2] != 3:
            print(f"[VideoStreamGenerator] Неправильный формат кадра: {frame_array.shape}")
            return
        
        # Если кадр в RGB, конвертируем в BGR
        # (обычно кадры из модели приходят в RGB)
        try:
            # Проверяем, нужно ли конвертировать (эвристика: если значения большие, это может быть RGB float)
            if frame_array.dtype != np.uint8:
                # Нормализуем и конвертируем в uint8
                if frame_array.max() > 1.0:
                    frame_array = (frame_array / 255.0).clip(0, 1)
                frame_array = (frame_array * 255).astype(np.uint8)
            
            # Проверяем формат кадра - кадры из OpenCV операций уже в BGR
            # Кадры из модели могут быть в RGB, но после get_image_blending они в BGR
            # Проверяем по эвристике: если кадр уже выглядит как BGR (из OpenCV), не конвертируем
            # Если кадр пришел из модели напрямую (RGB), конвертируем
            
            # Проверяем, нужно ли конвертировать RGB->BGR
            # Кадры из combine_frame уже в BGR (OpenCV формат)
            # Поэтому НЕ конвертируем, а используем как есть
            frame_final = frame_array.copy()
            
            # Проверяем размер очереди перед добавлением
            queue_size = self.frame_queue.qsize()
            queue_capacity = self.frame_queue.maxsize
            
            # Если очередь почти полная, логируем предупреждение
            if queue_size >= queue_capacity * 0.9:
                if self.frame_count % 30 == 0:
                    print(f"[VideoStreamGenerator] ⚠️ Очередь почти полная: {queue_size}/{queue_capacity} ({queue_size/queue_capacity*100:.1f}%)")
            
            # Вычисляем логический timestamp для кадра (frame_index / fps) ПЕРЕД использованием
            with self.real_frames_lock:
                self.real_frames_generated += 1
                frame_index = self.real_frames_generated - 1  # Индекс начинается с 0
                logical_timestamp = frame_index / self.fps  # Время в секундах
            
            # Добавляем кадр в очередь с логическим timestamp: (frame_array, logical_timestamp)
            frame_with_timestamp = (frame_final, logical_timestamp)
            
            # Добавляем кадр в очередь асинхронно через правильный loop
            # Используем put_nowait если очередь не полная, чтобы избежать блокировки
            try:
                if queue_size < queue_capacity:
                    # Очередь не полная - добавляем напрямую
                    asyncio.run_coroutine_threadsafe(
                        self.frame_queue.put(frame_with_timestamp),
                        self.loop
                    )
                else:
                    # Очередь полная - пытаемся добавить с проверкой
                    # В этом случае кадр может быть отброшен, но это лучше чем блокировка
                    if queue_size >= queue_capacity:
                        # Пытаемся удалить старый кадр из очереди (FIFO)
                        try:
                            # Создаем задачу для удаления старого кадра и добавления нового
                            async def replace_old_frame():
                                try:
                                    # Пытаемся удалить один старый кадр
                                    if not self.frame_queue.empty():
                                        await self.frame_queue.get()
                                    await self.frame_queue.put(frame_with_timestamp)
                                except:
                                    pass
                            asyncio.run_coroutine_threadsafe(replace_old_frame(), self.loop)
                        except:
                            # Если не удалось, просто пропускаем кадр
                            if self.real_frames_generated % 30 == 0:
                                print(f"[VideoStreamGenerator] ⚠️ Очередь переполнена, кадр пропущен")
            except Exception as e:
                print(f"[VideoStreamGenerator] Ошибка при добавлении кадра в очередь: {e}")
            
            # Отмечаем, что первый реальный сгенерированный кадр был добавлен
            # Это важно для синхронизации аудио - аудио должно начинаться только после реальных кадров
            with self.first_real_frame_lock:
                if not self.first_real_frame_sent:
                    self.first_real_frame_sent = True
                    print(f"[VideoStreamGenerator] ✓ Первый реальный сгенерированный кадр добавлен в очередь (размер: {frame_final.shape}, timestamp={logical_timestamp:.3f}s)")
                    print(f"[VideoStreamGenerator] Флаг first_real_frame_sent установлен в True - аудио должно начаться")
                else:
                    # Логируем, если флаг уже был установлен (при перезапуске)
                    if self.real_frames_generated % 30 == 0:
                        print(f"[VideoStreamGenerator] Кадр #{self.real_frames_generated} добавлен (timestamp={logical_timestamp:.3f}s, first_real_frame_sent уже True)")
            
            # Сохраняем кадр как последний с его timestamp
            with self.last_frame_lock:
                self.last_frame = frame_final.copy()
                self.last_frame_timestamp = logical_timestamp
            
            if self.frame_count % 30 == 0:  # Логируем периодически
                print(f"[VideoStreamGenerator] Кадр добавлен в очередь (размер: {frame_final.shape}, dtype: {frame_final.dtype}, очередь: {queue_size}/{queue_capacity})")
        except Exception as e:
            print(f"[VideoStreamGenerator] Ошибка при добавлении кадра: {e}")
            import traceback
            traceback.print_exc()
    
    def stop(self):
        """Останавливает поток и очищает очередь"""
        self.running = False
        
        # Останавливаем keepalive задачу
        if self.keepalive_task and not self.keepalive_task.done():
            if self.loop:
                def cancel_task():
                    self.keepalive_task.cancel()
                self.loop.call_soon_threadsafe(cancel_task)
        
        # Останавливаем задачу воспроизведения оригинального видео
        if self.original_video_task and not self.original_video_task.done():
            if self.loop:
                def cancel_original_task():
                    self.original_video_task.cancel()
                self.loop.call_soon_threadsafe(cancel_original_task)
        
        # Очищаем очередь кадров
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except:
                break
        
        # Очищаем последний кадр
        with self.last_frame_lock:
            self.last_frame = None
        
        print(f"[VideoStreamGenerator] Трек остановлен, очередь очищена, keepalive остановлен")


def init_daemon_service(version="v15", gpu_id=0, use_float16=True,
                       whisper_dir="./models/whisper", vae_type="sd-vae",
                       ffmpeg_path="./ffmpeg-4.4-amd64-static/",
                       left_cheek_width=90, right_cheek_width=90):
    """Инициализирует демон-сервис с предзагруженными моделями"""
    global daemon_service
    if daemon_service is None:
        print("[StreamingAPI] Инициализация ModelDaemonService...")
        daemon_service = ModelDaemonService(
            version=version,
            gpu_id=gpu_id,
            use_float16=use_float16,
            whisper_dir=whisper_dir,
            vae_type=vae_type,
            ffmpeg_path=ffmpeg_path,
            left_cheek_width=left_cheek_width,
            right_cheek_width=right_cheek_width,
            request_dir="./requests",
            result_dir="./results"
        )
        print("[StreamingAPI] ModelDaemonService инициализирован")
    return daemon_service


@app.on_event("startup")
async def startup_event():
    """Инициализация при старте приложения"""
    if not WEBRTC_AVAILABLE:
        print("[StreamingAPI] ВНИМАНИЕ: aiortc не установлен. WebRTC стриминг недоступен.")
        print("[StreamingAPI] Установите: pip install aiortc")
        return
    
    version = os.getenv("MUSETALK_VERSION", "v15")
    gpu_id = int(os.getenv("MUSETALK_GPU_ID", "0"))
    use_float16 = os.getenv("MUSETALK_USE_FLOAT16", "true").lower() == "true"
    
    init_daemon_service(
        version=version,
        gpu_id=gpu_id,
        use_float16=use_float16
    )


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
        # Проверяем наличие обязательных полей в запросе
        if "sdp" not in request or "type" not in request:
            raise HTTPException(status_code=400, detail="Отсутствуют поля sdp или type в запросе")
        
        if not request.get("sdp") or not request.get("type"):
            raise HTTPException(status_code=400, detail="Поля sdp или type пусты")
        
        offer = RTCSessionDescription(sdp=request["sdp"], type=request["type"])
        
        # Настройка диапазона портов для RTP/RTCP
        rtc_min_port = int(os.getenv("WEBRTC_RTC_MIN_PORT", "10000"))
        rtc_max_port = int(os.getenv("WEBRTC_RTC_MAX_PORT", "20000"))
        
        # Получаем параметры из запроса
        task_id = request.get("task_id")
        video_path = request.get("video_path")
        audio_path = request.get("audio_path")
        version = request.get("version", "v15")
        batch_size = request.get("batch_size", 8)
        fps = request.get("fps", 25)
        
        # Детальная проверка параметров
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
        
        # Проверка существования файлов
        if not os.path.exists(video_path):
            raise HTTPException(status_code=400, detail=f"Видео файл не найден: {video_path}")
        
        if not os.path.exists(audio_path):
            raise HTTPException(status_code=400, detail=f"Аудио файл не найден: {audio_path}")
        
        # ВАЖНО: Добавляем треки ДО установки remote description
        # Это нужно для правильной обработки направлений медиапотоков в SDP
        # Получаем event loop для использования в callback
        loop = asyncio.get_event_loop()
        
        # Получаем размеры оригинального видео
        video_width = 640
        video_height = 480
        try:
            video_cap = cv2.VideoCapture(video_path)
            if video_cap.isOpened():
                video_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                video_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                video_cap.release()
                print(f"[StreamingAPI] Размеры оригинального видео: {video_width}x{video_height}")
            else:
                print(f"[StreamingAPI] ⚠️ Не удалось открыть видео для получения размеров, используем значения по умолчанию: {video_width}x{video_height}")
        except Exception as e:
            print(f"[StreamingAPI] ⚠️ Ошибка при получении размеров видео: {e}, используем значения по умолчанию: {video_width}x{video_height}")
        
        # Создаем общий MediaClock для синхронизации
        media_clock = MediaClock()
        
        # Создаем видео поток с передачей event loop, размеров видео и пути к оригинальному видео
        # Передаем MediaClock для синхронизации
        video_track = VideoStreamGenerator(fps=fps, loop=loop, video_width=video_width, video_height=video_height, video_path=video_path, clock=media_clock)
        print(f"[StreamingAPI] Видео трек создан: {video_track}")
        
        # Создаем peer connection с настройками портов
        # Настройка ICE серверов
        ice_servers = []
        
        # STUN серверы (всегда добавляем)
        ice_servers.append(RTCIceServer(urls=["stun:stun.l.google.com:19302"]))
        ice_servers.append(RTCIceServer(urls=["stun:stun1.l.google.com:19302"]))
        ice_servers.append(RTCIceServer(urls=["stun:stun2.l.google.com:19302"]))
        
        # Проверяем, настроен ли собственный TURN сервер через переменные окружения
        turn_url = os.getenv("TURN_SERVER_URL")
        turn_username = os.getenv("TURN_SERVER_USERNAME", "")
        turn_credential = os.getenv("TURN_SERVER_CREDENTIAL", "")
        use_public_turn = os.getenv("USE_PUBLIC_TURN", "true").lower() == "true"
        
        # Диагностика переменных окружения
        print(f"[StreamingAPI] Проверка переменных окружения TURN:")
        print(f"[StreamingAPI]   TURN_SERVER_URL: {'установлен' if turn_url else 'НЕ установлен'}")
        if turn_url:
            print(f"[StreamingAPI]   TURN_SERVER_URL значение: {turn_url}")
        print(f"[StreamingAPI]   TURN_SERVER_USERNAME: {'установлен' if turn_username else 'НЕ установлен'}")
        print(f"[StreamingAPI]   TURN_SERVER_CREDENTIAL: {'установлен' if turn_credential else 'НЕ установлен'}")
        print(f"[StreamingAPI]   USE_PUBLIC_TURN: {use_public_turn}")
        
        turn_servers_added = 0
        
        # Добавляем собственный TURN сервер, если настроен
        if turn_url:
            turn_urls = [turn_url]
            # Добавляем TCP вариант, если не указан
            if "?transport=" not in turn_url:
                turn_urls.append(f"{turn_url}?transport=tcp")
            
            if turn_username and turn_credential:
                ice_servers.append(
                    RTCIceServer(
                        urls=turn_urls,
                        username=turn_username,
                        credential=turn_credential
                    )
                )
                turn_servers_added += 1
                print(f"[StreamingAPI] ✓ Собственный TURN сервер настроен: {turn_url}")
            else:
                print(f"[StreamingAPI] ⚠️ TURN_SERVER_URL указан, но нет USERNAME/CREDENTIAL")
        
        # Добавляем публичные TURN серверы, если не отключены и собственный не настроен
        if use_public_turn and turn_servers_added == 0:
            print(f"[StreamingAPI] Используются публичные TURN серверы (можно отключить через USE_PUBLIC_TURN=false)")
            # Публичные TURN серверы (openrelay.metered.ca - бесплатный)
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
            # Альтернативные TURN серверы
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
        elif not use_public_turn:
            print(f"[StreamingAPI] Публичные TURN серверы отключены (USE_PUBLIC_TURN=false)")
        
        print(f"[StreamingAPI] Настроено ICE серверов: {len(ice_servers)} (3 STUN + {turn_servers_added} TURN)")
        if turn_servers_added == 0:
            print(f"[StreamingAPI] ")
            print(f"[StreamingAPI] ⚠️⚠️⚠️ ВНИМАНИЕ: TURN серверы НЕ настроены! ⚠️⚠️⚠️")
            print(f"[StreamingAPI] Соединение НЕ СМОЖЕТ установиться через NAT/Firewall без TURN сервера!")
            print(f"[StreamingAPI] ")
            print(f"[StreamingAPI] Настройте TURN сервер через переменные окружения или .env файл:")
            print(f"[StreamingAPI] ")
            print(f"[StreamingAPI] Вариант 1: Переменные окружения")
            print(f"[StreamingAPI]   export TURN_SERVER_URL='turn:YOUR_IP_ADDRESS:3478'")
            print(f"[StreamingAPI]   export TURN_SERVER_USERNAME='username'")
            print(f"[StreamingAPI]   export TURN_SERVER_CREDENTIAL='password'")
            print(f"[StreamingAPI] ")
            print(f"[StreamingAPI] Вариант 2: .env файл (создайте .env в корне проекта)")
            print(f"[StreamingAPI]   TURN_SERVER_URL=turn:YOUR_IP_ADDRESS:3478")
            print(f"[StreamingAPI]   TURN_SERVER_USERNAME=username")
            print(f"[StreamingAPI]   TURN_SERVER_CREDENTIAL=password")
            print(f"[StreamingAPI] ")
            print(f"[StreamingAPI] Пример: TURN_SERVER_URL=turn:93.173.41.137:3478")
            print(f"[StreamingAPI] ")
        
        pc = RTCPeerConnection(
            configuration=RTCConfiguration(iceServers=ice_servers),
            # Настройка портов для RTP/RTCP
            # Примечание: aiortc может не поддерживать прямую настройку портов
            # Порты будут выбираться автоматически из доступного диапазона
        )
        
        connection_id = id(pc)
        active_connections[connection_id] = pc
        
        # Флаг для отслеживания, была ли запущена проверка recv()
        recv_check_started = False
        
        # Проверяем, был ли вызван recv() после установки соединения
        async def check_recv_called_after_connection():
            """Проверяет, вызывается ли recv() после установки соединения"""
            nonlocal recv_check_started
            if recv_check_started:
                print(f"[StreamingAPI] Проверка recv() уже запущена, пропускаем")
                return
            
            recv_check_started = True
            ice_state = pc.iceConnectionState
            conn_state = pc.connectionState
            
            print(f"[StreamingAPI] ===== НАЧАЛО ПРОВЕРКИ recv() =====")
            print(f"[StreamingAPI] Начальное состояние: ICE={ice_state}, Connection={conn_state}")
            print(f"[StreamingAPI] recv() должен вызываться автоматически WebRTC")
            
            # Ждем до 20 секунд, пока соединение установится, или пока recv() не будет вызван
            max_wait_time = 20  # секунд
            check_interval = 1.0  # секунд (проверяем каждую секунду)
            waited = 0
            last_logged_state = None
            
            while waited < max_wait_time:
                await asyncio.sleep(check_interval)
                waited += check_interval
                
                ice_state_current = pc.iceConnectionState
                conn_state_current = pc.connectionState
                state_str = f"ICE={ice_state_current}, Connection={conn_state_current}"
                
                # Логируем изменение состояния каждую секунду
                if state_str != last_logged_state or waited % 5 == 0:
                    print(f"[StreamingAPI] Проверка recv() [{waited:.0f}с]: {state_str}, frame_count={video_track.frame_count}")
                    last_logged_state = state_str
                
                # Если recv() был вызван (frame_count > 0), прекращаем проверку
                if video_track.frame_count > 0:
                    print(f"[StreamingAPI] ===== ПРОВЕРКА ЗАВЕРШЕНА: recv() был вызван через {waited:.1f}с =====")
                    print(f"[StreamingAPI] Финальное состояние: {state_str}")
                    return
                
                # Если соединение установлено, продолжаем ждать еще немного
                if ice_state_current == "connected" or conn_state_current == "connected":
                    if waited <= 2:
                        print(f"[StreamingAPI] ✓✓✓ Соединение установлено ({state_str})! ✓✓✓")
                        print(f"[StreamingAPI] Продолжаем проверку recv()...")
                    # Если соединение установлено более 2 секунд, ждем еще немного и завершаем
                    if waited >= 4:
                        print(f"[StreamingAPI] Соединение установлено более 4 секунд, завершаем проверку")
                        break
                
                # Если соединение упало, прекращаем проверку
                if ice_state_current in ["failed", "disconnected", "closed"] or conn_state_current in ["failed", "disconnected", "closed"]:
                    print(f"[StreamingAPI] ===== ПРОВЕРКА ПРЕРВАНА: Соединение потеряно =====")
                    print(f"[StreamingAPI] Состояние: {state_str}")
                    print(f"[StreamingAPI] recv() не был вызван за {waited:.1f}с")
                    
                    # Проверяем SDP еще раз для диагностики
                    if pc.localDescription:
                        sdp = pc.localDescription.sdp
                        relay_count = sdp.count('typ relay')
                        print(f"[StreamingAPI] Relay кандидатов в SDP при разрыве: {relay_count}")
                        if relay_count > 0:
                            print(f"[StreamingAPI] TURN сервер был настроен, но соединение все равно разорвалось")
                            print(f"[StreamingAPI] Возможные причины: проблемы с сетью, порты закрыты, TURN сервер недоступен")
                    return
            
            # Финальная проверка
            ice_state_final = pc.iceConnectionState
            conn_state_final = pc.connectionState
            ice_gathering_final = pc.iceGatheringState
            
            if video_track.frame_count == 0:
                queue_size = video_track.frame_queue.qsize()
                print(f"[StreamingAPI] ===== ПРЕДУПРЕЖДЕНИЕ: recv() не был вызван за {waited:.1f}с! =====")
                print(f"[StreamingAPI] Финальное состояние:")
                print(f"[StreamingAPI]   ICE connection: {ice_state_final}")
                print(f"[StreamingAPI]   Connection: {conn_state_final}")
                print(f"[StreamingAPI]   ICE gathering: {ice_gathering_final}")
                print(f"[StreamingAPI] Кадров в очереди: {queue_size}")
                
                # Проверяем статистику кандидатов
                relay_count = ice_candidates_count.get('relay', 0)
                host_count = ice_candidates_count.get('host', 0)
                srflx_count = ice_candidates_count.get('srflx', 0)
                
                print(f"[StreamingAPI] Статистика ICE кандидатов: host={host_count}, srflx={srflx_count}, relay={relay_count}")
                
                if ice_state_final not in ["connected"] and conn_state_final not in ["connected"]:
                    print(f"[StreamingAPI] ")
                    print(f"[StreamingAPI] ⚠️  ПРОБЛЕМА: WebRTC соединение не установлено!")
                    print(f"[StreamingAPI] ")
                    
                    if relay_count == 0:
                        print(f"[StreamingAPI] 🔴 КРИТИЧЕСКАЯ ПРОБЛЕМА: TURN сервер НЕ используется!")
                        print(f"[StreamingAPI]    Найдено relay кандидатов: {relay_count} (должно быть > 0)")
                        print(f"[StreamingAPI] ")
                        print(f"[StreamingAPI] БЕЗ TURN сервера соединение НЕ МОЖЕТ установиться через NAT/Firewall!")
                        print(f"[StreamingAPI] ")
                        print(f"[StreamingAPI] РЕШЕНИЕ: Настройте собственный TURN сервер:")
                        print(f"[StreamingAPI] ")
                        print(f"[StreamingAPI] 1. Установите coturn:")
                        print(f"[StreamingAPI]    sudo apt-get install coturn")
                        print(f"[StreamingAPI] ")
                        print(f"[StreamingAPI] 2. Настройте /etc/turnserver.conf:")
                        print(f"[StreamingAPI]    listening-port=3478")
                        print(f"[StreamingAPI]    external-ip=YOUR_PUBLIC_IP")
                        print(f"[StreamingAPI]    user=username:password")
                        print(f"[StreamingAPI] ")
                        print(f"[StreamingAPI] 3. Запустите coturn:")
                        print(f"[StreamingAPI]    sudo systemctl start coturn")
                        print(f"[StreamingAPI] ")
                        print(f"[StreamingAPI] 4. Настройте переменные окружения (можно использовать IP без домена):")
                        print(f"[StreamingAPI]    export TURN_SERVER_URL='turn:YOUR_PUBLIC_IP:3478'")
                        print(f"[StreamingAPI]    export TURN_SERVER_USERNAME='username'")
                        print(f"[StreamingAPI]    export TURN_SERVER_CREDENTIAL='password'")
                        print(f"[StreamingAPI]    export USE_PUBLIC_TURN='false'")
                        print(f"[StreamingAPI]    Пример: export TURN_SERVER_URL='turn:93.173.41.137:3478'")
                        print(f"[StreamingAPI] ")
                        print(f"[StreamingAPI] 5. Перезапустите сервер")
                        print(f"[StreamingAPI] ")
                    else:
                        print(f"[StreamingAPI] TURN сервер работает (relay кандидатов: {relay_count}), но соединение не устанавливается")
                        print(f"[StreamingAPI] Возможные причины:")
                        print(f"[StreamingAPI]   1. Порты UDP не открыты на сервере/роутере")
                        print(f"[StreamingAPI]   2. Firewall блокирует трафик")
                        print(f"[StreamingAPI]   3. Проблемы с сетью между клиентом и сервером")
                    
                    print(f"[StreamingAPI] ")
                    print(f"[StreamingAPI] Примечание: Кадры генерируются ({queue_size} в очереди), но не отправляются,")
                    print(f"[StreamingAPI] так как WebRTC не может установить соединение с клиентом.")
                else:
                    print(f"[StreamingAPI] Соединение установлено, но WebRTC не запрашивает кадры")
                    print(f"[StreamingAPI] Возможные причины: клиент не начал воспроизведение, проблемы с медиа потоком")
            else:
                print(f"[StreamingAPI] ===== ПРОВЕРКА УСПЕШНА: recv() был вызван =====")
                print(f"[StreamingAPI] Финальное состояние: ICE={ice_state_final}, Connection={conn_state_final}")
                print(f"[StreamingAPI] Трек активен и WebRTC запрашивает кадры")
        
        # Добавляем обработчики событий для диагностики и очистки
        @pc.on("iceconnectionstatechange")
        async def on_ice_connection_state_change():
            state = pc.iceConnectionState
            print(f"[StreamingAPI] ICE connection state: {state}")
            
            # Если соединение установлено, запускаем проверку recv()
            if state == "connected":
                if not recv_check_started:
                    print(f"[StreamingAPI] ICE соединение установлено, запускаем проверку recv()")
                    asyncio.create_task(check_recv_called_after_connection())
                else:
                    print(f"[StreamingAPI] ICE соединение установлено, но проверка уже запущена")
            
            # Закрываем соединение при ошибке или разрыве
            # Но не останавливаем поток при "disconnected" - это может быть временное состояние
            if state in ["failed", "closed"]:
                print(f"[StreamingAPI] Очистка ресурсов из-за состояния ICE: {state}")
                video_track.stop()
                if audio_track:
                    audio_track.stop()
                
                # Останавливаем генерацию и удаляем из active_generations
                with generation_lock:
                    # Находим task_id по pc
                    for tid, gen_info in list(active_generations.items()):
                        if gen_info["pc"] == pc:
                            print(f"[StreamingAPI] Остановка генерации {tid} из-за закрытия соединения")
                            gen_info["stop_flag"].set()
                            del active_generations[tid]
                            break
                
                # Удаляем информацию о соединении при закрытии
                with connection_info_lock:
                    # Находим task_id по pc
                    for tid in list(connection_info_by_task.keys()):
                        if connection_info_by_task[tid]["pc"] == pc:
                            print(f"[StreamingAPI] Удаление информации о соединении {tid} из-за закрытия")
                            del connection_info_by_task[tid]
                            break
                
                if connection_id in active_connections:
                    del active_connections[connection_id]
                try:
                    await pc.close()
                except:
                    pass
            elif state == "disconnected":
                print(f"[StreamingAPI] ICE соединение разорвано (disconnected), но не останавливаем поток - может восстановиться")
                # Не останавливаем поток при disconnected - соединение может восстановиться
        
        @pc.on("icegatheringstatechange")
        async def on_ice_gathering_state_change():
            state = pc.iceGatheringState
            print(f"[StreamingAPI] ICE gathering state: {state}")
            if state == "complete":
                print(f"[StreamingAPI] ICE gathering завершен - все кандидаты собраны")
                # Получаем информацию о трансиверах
                try:
                    transceivers = pc.getTransceivers()
                    print(f"[StreamingAPI] Количество трансиверов: {len(transceivers)}")
                    for idx, transceiver in enumerate(transceivers):
                        print(f"[StreamingAPI]   Transceiver {idx} ({transceiver.kind}): direction={transceiver.direction}, currentDirection={transceiver.currentDirection}")
                    
                    # Пытаемся получить информацию о локальном SDP для диагностики
                    if pc.localDescription:
                        sdp = pc.localDescription.sdp
                        # Проверяем наличие relay кандидатов в SDP
                        relay_count = sdp.count('typ relay')
                        host_count = sdp.count('typ host')
                        srflx_count = sdp.count('typ srflx')
                        prflx_count = sdp.count('typ prflx')
                        
                        # Обновляем счетчик кандидатов из SDP (так как события могут не срабатывать)
                        ice_candidates_count['host'] = max(ice_candidates_count['host'], host_count)
                        ice_candidates_count['srflx'] = max(ice_candidates_count['srflx'], srflx_count)
                        ice_candidates_count['relay'] = max(ice_candidates_count['relay'], relay_count)
                        ice_candidates_count['prflx'] = max(ice_candidates_count['prflx'], prflx_count)
                        
                        print(f"[StreamingAPI] Кандидаты в SDP (после ICE gathering): host={host_count}, srflx={srflx_count}, relay={relay_count}, prflx={prflx_count}")
                        if relay_count > 0:
                            print(f"[StreamingAPI] ✓✓✓ TURN сервер работает! Найдено {relay_count} relay кандидатов в SDP ✓✓✓")
                        else:
                            print(f"[StreamingAPI] ⚠️ TURN сервер не используется - нет relay кандидатов в SDP")
                            print(f"[StreamingAPI] ")
                            print(f"[StreamingAPI] ВАЖНО: Без TURN сервера соединение может не установиться через NAT/Firewall!")
                            print(f"[StreamingAPI] ")
                            print(f"[StreamingAPI] Возможные причины:")
                            print(f"[StreamingAPI]   1. TURN сервер недоступен (проверьте доступность)")
                            print(f"[StreamingAPI]   2. Неправильные учетные данные")
                            print(f"[StreamingAPI]   3. Порты закрыты (3478 UDP/TCP, 49152-65535 UDP)")
                            print(f"[StreamingAPI]   4. TURN сервер не запущен или не работает")
                            print(f"[StreamingAPI] ")
                            print(f"[StreamingAPI] РЕШЕНИЕ: Настройте собственный TURN сервер (coturn)")
                            print(f"[StreamingAPI]   Или используйте коммерческие TURN сервисы (Twilio, Xirsys)")
                except Exception as e:
                    print(f"[StreamingAPI] Ошибка при получении информации о трансиверах: {e}")
                    import traceback
                    traceback.print_exc()
        
        # Счетчик кандидатов для диагностики
        ice_candidates_count = {'host': 0, 'srflx': 0, 'relay': 0, 'prflx': 0}
        
        @pc.on("icecandidate")
        async def on_ice_candidate(candidate):
            if candidate:
                candidate_type = getattr(candidate, 'type', 'unknown')
                candidate_protocol = getattr(candidate, 'protocol', 'unknown')
                candidate_address = getattr(candidate, 'address', 'unknown')
                candidate_port = getattr(candidate, 'port', 'unknown')
                candidate_priority = getattr(candidate, 'priority', 'unknown')
                
                # Подсчитываем кандидаты по типам
                if candidate_type in ice_candidates_count:
                    ice_candidates_count[candidate_type] += 1
                
                print(f"[StreamingAPI] ICE candidate: {candidate_type} {candidate_protocol} {candidate_address}:{candidate_port} (priority: {candidate_priority})")
                
                # Проверяем, является ли это relay кандидатом (TURN)
                if candidate_type == 'relay':
                    print(f"[StreamingAPI] ✓✓✓ TURN сервер работает! Relay кандидат найден ✓✓✓")
            else:
                print(f"[StreamingAPI] ICE gathering завершен (все кандидаты собраны)")
                print(f"[StreamingAPI] Статистика кандидатов: host={ice_candidates_count['host']}, srflx={ice_candidates_count['srflx']}, relay={ice_candidates_count['relay']}, prflx={ice_candidates_count['prflx']}")
                if ice_candidates_count['relay'] == 0:
                    print(f"[StreamingAPI] ⚠️ ВНИМАНИЕ: Нет relay кандидатов - TURN сервер не используется!")
        
        @pc.on("track")
        async def on_track(track):
            print(f"[StreamingAPI] Track received: {track.kind}, id={track.id}")
            # Примечание: это событие срабатывает только для входящих треков, не для исходящих
        
        @pc.on("connectionstatechange")
        async def on_connection_state_change():
            state = pc.connectionState
            ice_state = pc.iceConnectionState
            print(f"[StreamingAPI] Connection state: {state} (ICE: {ice_state})")
            
            # Если соединение установлено, запускаем проверку recv()
            if state == "connected":
                if not recv_check_started:
                    print(f"[StreamingAPI] ✓✓✓ Connection установлено! Запускаем проверку recv() ✓✓✓")
                    asyncio.create_task(check_recv_called_after_connection())
                else:
                    print(f"[StreamingAPI] Connection установлено, но проверка уже запущена")
            
            # Закрываем соединение при ошибке или разрыве
            # Но не останавливаем поток при "disconnected" - это может быть временное состояние
            if state in ["failed", "closed"]:
                print(f"[StreamingAPI] ⚠️ Очистка ресурсов из-за состояния соединения: {state}")
                print(f"[StreamingAPI] ICE состояние: {ice_state}")
                print(f"[StreamingAPI] Статистика кандидатов: host={ice_candidates_count['host']}, srflx={ice_candidates_count['srflx']}, relay={ice_candidates_count['relay']}")
                if ice_candidates_count['relay'] == 0:
                    print(f"[StreamingAPI] 🔴 ПРОБЛЕМА: TURN сервер не использовался - нет relay кандидатов!")
                    print(f"[StreamingAPI] Это основная причина разрыва соединения через NAT/Firewall")
                video_track.stop()
                if audio_track:
                    audio_track.stop()
                
                # Останавливаем генерацию и удаляем из active_generations
                with generation_lock:
                    # Находим task_id по pc
                    for tid, gen_info in list(active_generations.items()):
                        if gen_info["pc"] == pc:
                            print(f"[StreamingAPI] Остановка генерации {tid} из-за закрытия соединения")
                            gen_info["stop_flag"].set()
                            del active_generations[tid]
                            break
                
                # Удаляем информацию о соединении при закрытии
                with connection_info_lock:
                    # Находим task_id по pc
                    for tid in list(connection_info_by_task.keys()):
                        if connection_info_by_task[tid]["pc"] == pc:
                            print(f"[StreamingAPI] Удаление информации о соединении {tid} из-за закрытия")
                            del connection_info_by_task[tid]
                            break
                
                if connection_id in active_connections:
                    del active_connections[connection_id]
                try:
                    await pc.close()
                except:
                    pass
            elif state == "disconnected":
                print(f"[StreamingAPI] Connection разорвано (disconnected), ICE: {ice_state}")
                print(f"[StreamingAPI] Не останавливаем поток - соединение может восстановиться")
                # Не останавливаем поток при disconnected - соединение может восстановиться
        
        # Добавляем трек в peer connection ДО установки remote description
        sender = pc.addTrack(video_track)
        print(f"[StreamingAPI] Видео трек добавлен в peer connection, sender={sender}")
        
        # Для аудио создаем синхронизированный поток
        # Используем кастомный трек для синхронизации с видео
        audio_track = None
        try:
            # Проверяем существование аудио файла
            if not os.path.exists(audio_path):
                print(f"[StreamingAPI] ⚠️ Предупреждение: аудио файл не найден: {audio_path}")
            else:
                # Проверяем, использовать ли кастомный трек или MediaPlayer
                # По умолчанию используем MediaPlayer с оберткой (лучшее качество звука + синхронизация)
                use_custom_audio = os.getenv("USE_CUSTOM_AUDIO_TRACK", "false").lower() == "true"
                
                if use_custom_audio:
                    # Создаем синхронизированный аудио трек
                    print(f"[StreamingAPI] Создание синхронизированного аудио потока из файла: {audio_path}")
                    try:
                        # Создаем аудио трек с MediaClock (мастер-клок)
                        audio_track = SynchronizedAudioTrack(
                            audio_path=audio_path,
                            clock=media_clock,
                            video_track=video_track,
                            sample_rate=48000,
                            channels=2
                        )
                        pc.addTrack(audio_track)
                        print(f"[StreamingAPI] ✓ Синхронизированный аудио поток добавлен успешно")
                        print(f"[StreamingAPI] Аудио трек будет синхронизирован с видео (мастер-клок)")
                    except Exception as e:
                        print(f"[StreamingAPI] Ошибка при создании кастомного аудио трека: {e}")
                        print(f"[StreamingAPI] Переключаемся на MediaPlayer...")
                        use_custom_audio = False
                
                if not use_custom_audio:
                    # Используем MediaPlayer с оберткой для синхронизации
                    print(f"[StreamingAPI] Использование MediaPlayer для аудио с синхронизацией (DelayedMediaPlayerTrack)")
                    print(f"[StreamingAPI] Аудио файл: {audio_path}")
                    audio_player = MediaPlayer(audio_path)
                    if audio_player and audio_player.audio:
                        # Используем обертку для задержки начала воспроизведения
                        audio_track = DelayedMediaPlayerTrack(audio_player, video_track)
                        # Для DelayedMediaPlayerTrack не можем использовать get_t_playback, 
                        # но можем попробовать добавить метод или использовать альтернативный подход
                        # Пока оставляем без установки audio_track в video_track для MediaPlayer
                        pc.addTrack(audio_track)
                        print(f"[StreamingAPI] ✓ Аудио поток добавлен через MediaPlayer с синхронизацией")
                        print(f"[StreamingAPI] Для использования кастомного трека установите USE_CUSTOM_AUDIO_TRACK=true")
                        print(f"[StreamingAPI] ⚠️ Примечание: синхронизация с t_playback работает только с USE_CUSTOM_AUDIO_TRACK=true")
                    else:
                        print(f"[StreamingAPI] ⚠️ Предупреждение: не удалось создать аудио поток через MediaPlayer")
        except Exception as e:
            print(f"[StreamingAPI] Критическая ошибка при создании аудио потока: {e}")
            import traceback
            traceback.print_exc()
            print(f"[StreamingAPI] Продолжаем без аудио")
        
        # Теперь устанавливаем remote description
        try:
            await pc.setRemoteDescription(offer)
            print(f"[StreamingAPI] Remote description установлен")
        except Exception as e:
            print(f"[StreamingAPI] Ошибка при установке remote description: {e}")
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Ошибка при установке remote description: {str(e)}")
        
        # Создаем answer ПОСЛЕ установки remote description и добавления треков
        try:
            answer = await pc.createAnswer()
            await pc.setLocalDescription(answer)
            print(f"[StreamingAPI] Answer создан и установлен")
            
            # Проверяем SDP на наличие TURN серверов и relay кандидатов
            if pc.localDescription:
                sdp = pc.localDescription.sdp
                # Проверяем наличие TURN серверов в SDP
                turn_in_sdp = "turn:" in sdp.lower() or "turns:" in sdp.lower()
                relay_in_sdp = "typ relay" in sdp.lower() or ("candidate:" in sdp.lower() and "relay" in sdp.lower())
                
                print(f"[StreamingAPI] Проверка SDP answer:")
                print(f"[StreamingAPI]   TURN серверы в SDP: {'да' if turn_in_sdp else 'нет'}")
                print(f"[StreamingAPI]   Relay кандидаты в SDP: {'да' if relay_in_sdp else 'нет'}")
                
                # Подсчитываем кандидаты в SDP (более точный подсчет)
                relay_count_sdp = sdp.count('typ relay')
                host_count_sdp = sdp.count('typ host')
                srflx_count_sdp = sdp.count('typ srflx')
                prflx_count_sdp = sdp.count('typ prflx')
                
                print(f"[StreamingAPI] Кандидаты в SDP answer: host={host_count_sdp}, srflx={srflx_count_sdp}, relay={relay_count_sdp}, prflx={prflx_count_sdp}")
                
                if relay_count_sdp > 0:
                    print(f"[StreamingAPI] ✓✓✓ TURN сервер работает! Найдено {relay_count_sdp} relay кандидатов в SDP ✓✓✓")
                else:
                    print(f"[StreamingAPI] ⚠️ ВНИМАНИЕ: Нет relay кандидатов в SDP answer!")
                    if turn_in_sdp:
                        print(f"[StreamingAPI] TURN серверы указаны в SDP, но relay кандидаты не сгенерированы")
                        print(f"[StreamingAPI] Возможные причины:")
                        print(f"[StreamingAPI]   1. TURN сервер недоступен (проверьте доступность)")
                        print(f"[StreamingAPI]   2. Неправильные учетные данные")
                        print(f"[StreamingAPI]   3. Порты закрыты (3478 UDP/TCP, 49152-65535 UDP)")
                        print(f"[StreamingAPI]   4. TURN сервер не запущен или не работает")
                    else:
                        print(f"[StreamingAPI] TURN серверы НЕ указаны в SDP - проверьте настройку ICE серверов")
        except Exception as e:
            print(f"[StreamingAPI] Ошибка при создании answer: {e}")
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Ошибка при создании answer: {str(e)}")
        
        # Проверяем, что answer создан
        if not pc.localDescription:
            raise HTTPException(status_code=500, detail="Не удалось создать local description")
        
        # Проверяем, что daemon_service инициализирован
        if daemon_service is None:
            raise HTTPException(status_code=500, detail="Daemon service не инициализирован")
        
        # НЕ запускаем генерацию автоматически - она будет запущена по запросу через /api/webrtc/restart_generation
        print(f"[StreamingAPI] WebRTC соединение установлено, генерация НЕ запущена автоматически")
        print(f"[StreamingAPI] Для запуска генерации используйте endpoint /api/webrtc/restart_generation")
        
        # Сохраняем информацию о соединении для возможности перезапуска после завершения генерации
        with connection_info_lock:
            connection_info_by_task[task_id] = {
                "pc": pc,
                "video_track": video_track,
                "audio_track": audio_track,
                "video_path": video_path,
                "audio_path": audio_path,
                "version": version,
                "batch_size": batch_size,
                "fps": fps
            }
            print(f"[StreamingAPI] Информация о соединении {task_id} сохранена")
        
        # Возвращаем answer и информацию о TURN сервере (если настроен)
        response = {
            "sdp": pc.localDescription.sdp,
            "type": pc.localDescription.type
        }
        
        # Добавляем информацию о TURN сервере для клиента (опционально)
        if turn_url and turn_username and turn_credential:
            response["turn_server"] = {
                "url": turn_url,
                "username": turn_username,
                "credential": turn_credential
            }
            print(f"[StreamingAPI] Информация о TURN сервере передана клиенту")
        
        return response
        
    except Exception as e:
        print(f"[StreamingAPI] Ошибка в WebRTC offer: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/webrtc/restart_generation")
async def restart_generation(request: dict):
    """Запускает или перезапускает генерацию видео для существующего WebRTC соединения"""
    if not WEBRTC_AVAILABLE:
        raise HTTPException(status_code=503, detail="WebRTC не доступен. Установите aiortc.")
    
    # Проверяем, что daemon_service инициализирован
    if daemon_service is None:
        raise HTTPException(status_code=500, detail="Daemon service не инициализирован")
    
    try:
        task_id = request.get("task_id")
        if not task_id:
            raise HTTPException(status_code=400, detail="Отсутствует task_id")
        
        # Получаем информацию о соединении (может быть даже если генерация завершена)
        with connection_info_lock:
            if task_id not in connection_info_by_task:
                raise HTTPException(status_code=404, detail=f"Соединение для task_id={task_id} не найдено")
            
            conn_info = connection_info_by_task[task_id]
            pc = conn_info["pc"]
            video_track = conn_info["video_track"]
            audio_track = conn_info.get("audio_track")
            
            # Получаем параметры из сохраненной информации о соединении
            video_path = conn_info["video_path"]
            audio_path = conn_info["audio_path"]
            version = conn_info["version"]
            batch_size = conn_info["batch_size"]
            fps = conn_info["fps"]
        
        # Проверяем, что соединение активно
        try:
            ice_state = pc.iceConnectionState
            conn_state = pc.connectionState
        except:
            raise HTTPException(status_code=400, detail="WebRTC соединение не доступно")
        
        if ice_state not in ["connected", "checking"] and conn_state not in ["connected", "connecting"]:
            raise HTTPException(status_code=400, detail=f"WebRTC соединение не активно (ICE: {ice_state}, Connection: {conn_state})")
        
        # Проверяем состояние video_track перед перезапуском
        print(f"[StreamingAPI] Состояние video_track перед перезапуском: running={video_track.running}, queue_size={video_track.frame_queue.qsize()}")
        
        # Проверяем, есть ли активная генерация
        with generation_lock:
            has_active_generation = task_id in active_generations
            
            if has_active_generation:
                # Останавливаем текущую генерацию
                print(f"[StreamingAPI] Остановка текущей генерации для task_id={task_id}")
                gen_info = active_generations[task_id]
                stop_flag = gen_info["stop_flag"]
                stop_flag.set()
                
                # Ждем завершения потока (с таймаутом)
                inference_thread = gen_info["inference_thread"]
                if inference_thread.is_alive():
                    inference_thread.join(timeout=5.0)
                    if inference_thread.is_alive():
                        print(f"[StreamingAPI] Предупреждение: поток генерации не завершился за 5 секунд")
                
                # Удаляем из активных генераций
                del active_generations[task_id]
            else:
                print(f"[StreamingAPI] Генерация для task_id={task_id} уже завершена, запускаем новую")
                # Создаем новый флаг остановки
                stop_flag = threading.Event()
            
            # Очищаем очередь кадров
            queue_cleared = 0
            while not video_track.frame_queue.empty():
                try:
                    video_track.frame_queue.get_nowait()
                    queue_cleared += 1
                except:
                    break
            if queue_cleared > 0:
                print(f"[StreamingAPI] Очищено {queue_cleared} кадров из очереди")
            
            # Убеждаемся, что video_track активен для новой генерации
            if not video_track.running:
                print(f"[StreamingAPI] ВНИМАНИЕ: video_track.running = False, устанавливаем в True для перезапуска")
                video_track.running = True
                # Также нужно сбросить флаг first_real_frame_sent для новой генерации
                with video_track.first_real_frame_lock:
                    video_track.first_real_frame_sent = False
                # Сбрасываем флаг завершения генерации для новой генерации
                with video_track.generation_completed_lock:
                    video_track.generation_completed = False
                print(f"[StreamingAPI] video_track активирован для новой генерации")
            else:
                print(f"[StreamingAPI] video_track уже активен (running=True)")
                # Все равно сбрасываем флаг first_real_frame_sent для новой генерации
                with video_track.first_real_frame_lock:
                    video_track.first_real_frame_sent = False
                # Сбрасываем флаг завершения генерации для новой генерации
                with video_track.generation_completed_lock:
                    video_track.generation_completed = False
            
            # Сбрасываем флаг audio_started в аудио треке для перезапуска аудио
            if audio_track:
                # Для SynchronizedAudioTrack используем метод reset()
                if hasattr(audio_track, 'reset'):
                    audio_track.reset()
                    print(f"[StreamingAPI] SynchronizedAudioTrack перезапущен через reset()")
                else:
                    # Для других типов треков используем старую логику
                    if hasattr(audio_track, 'audio_started'):
                        audio_track.audio_started = False
                        print(f"[StreamingAPI] Флаг audio_started сброшен в аудио треке для перезапуска")
                    # Для SynchronizedAudioTrack сбрасываем позицию
                    if hasattr(audio_track, 'current_frame_index'):
                        audio_track.current_frame_index = 0
                        if hasattr(audio_track, 'audio_position'):
                            audio_track.audio_position = 0
                        print(f"[StreamingAPI] Позиция аудио сброшена для перезапуска")
                    # Для DelayedMediaPlayerTrack сбрасываем счетчик тишины и перезапускаем MediaPlayer
                    if hasattr(audio_track, 'silence_frames_sent'):
                        audio_track.silence_frames_sent = 0
                        print(f"[StreamingAPI] Счетчик тишины сброшен для перезапуска")
                    # Сбрасываем счетчик вызовов recv для логирования
                    if hasattr(audio_track, '_recv_call_count'):
                        audio_track._recv_call_count = 0
                        print(f"[StreamingAPI] Счетчик вызовов recv сброшен для перезапуска")
                # Сбрасываем флаги и счетчики ошибок для DelayedMediaPlayerTrack
                if hasattr(audio_track, 'media_player_ended'):
                    audio_track.media_player_ended = False
                    print(f"[StreamingAPI] Флаг media_player_ended сброшен для перезапуска")
                if hasattr(audio_track, 'error_count'):
                    audio_track.error_count = 0
                if hasattr(audio_track, 'last_error_log_time'):
                    audio_track.last_error_log_time = 0
                # Для DelayedMediaPlayerTrack перезапускаем MediaPlayer
                if hasattr(audio_track, 'media_player') and audio_track.media_player:
                    try:
                        print(f"[StreamingAPI] Перезапуск MediaPlayer для DelayedMediaPlayerTrack")
                        # Останавливаем и закрываем текущий MediaPlayer
                        old_media_player = audio_track.media_player
                        try:
                            if hasattr(old_media_player, 'audio') and old_media_player.audio:
                                old_media_player.audio.stop()
                        except Exception as e:
                            print(f"[StreamingAPI] Ошибка при остановке старого MediaPlayer: {e}")
                        
                        try:
                            if hasattr(old_media_player, 'close'):
                                old_media_player.close()
                        except Exception as e:
                            print(f"[StreamingAPI] Ошибка при закрытии старого MediaPlayer: {e}")
                        
                        # Создаем новый MediaPlayer для перезапуска
                        from aiortc.contrib.media import MediaPlayer
                        new_media_player = MediaPlayer(audio_path)
                        if new_media_player and new_media_player.audio:
                            audio_track.media_player = new_media_player
                            # Сбрасываем флаги для нового MediaPlayer
                            if hasattr(audio_track, 'media_player_ended'):
                                audio_track.media_player_ended = False
                            if hasattr(audio_track, 'error_count'):
                                audio_track.error_count = 0
                            if hasattr(audio_track, 'last_error_log_time'):
                                audio_track.last_error_log_time = 0
                            print(f"[StreamingAPI] ✓ MediaPlayer перезапущен для перезапуска аудио (новый MediaPlayer создан)")
                        else:
                            print(f"[StreamingAPI] ⚠️ Не удалось создать новый MediaPlayer для перезапуска")
                    except Exception as e:
                        print(f"[StreamingAPI] Ошибка при перезапуске MediaPlayer: {e}")
                        import traceback
                        traceback.print_exc()
            
        # Запускаем новую генерацию
        def run_inference():
            try:
                print(f"[StreamingAPI] {'Перезапуск' if has_active_generation else 'Запуск'} генерации видео для task_id={task_id}")
                
                def frame_callback(frame_array, frame_index):
                    """Callback для добавления кадра в WebRTC поток"""
                    try:
                        # Проверяем флаг остановки
                        if stop_flag.is_set():
                            print(f"[StreamingAPI] Получен сигнал остановки генерации")
                            return
                        
                        # Используем правильный event loop для добавления кадра
                        video_track.add_frame(frame_array)
                        if frame_index % 30 == 0:  # Логируем каждые 30 кадров
                            print(f"[StreamingAPI] Отправлен кадр #{frame_index}")
                    except Exception as e:
                        print(f"[StreamingAPI] Ошибка при отправке кадра {frame_index}: {e}")
                        import traceback
                        traceback.print_exc()
                
                print(f"[StreamingAPI] Вызов process_request_webrtc для {'перезапуска' if has_active_generation else 'запуска'}...")
                daemon_service.process_request_webrtc(
                    task_id=task_id,
                    video_path=video_path,
                    audio_path=audio_path,
                    frame_callback=frame_callback,
                    version=version,
                    batch_size=batch_size,
                    fps=fps,
                    video_track=video_track,
                    stop_flag=stop_flag
                )
                print(f"[StreamingAPI] Генерация видео завершена ({'перезапуск' if has_active_generation else 'запуск'})")
                
                # Отмечаем, что генерация завершена - теперь можно использовать оригинальное видео
                with video_track.generation_completed_lock:
                    video_track.generation_completed = True
                    print(f"[StreamingAPI] ✓ Флаг generation_completed установлен - будет использоваться оригинальное видео")
                    print(f"[StreamingAPI] Состояние: first_real_frame_sent={video_track.first_real_frame_sent}, generation_completed={video_track.generation_completed}")
            except Exception as e:
                print(f"[StreamingAPI] Ошибка при {'перезапуске' if has_active_generation else 'запуске'} инференса: {e}")
                import traceback
                traceback.print_exc()
            finally:
                # Удаляем из активных генераций
                with generation_lock:
                    if task_id in active_generations:
                        del active_generations[task_id]
                        print(f"[StreamingAPI] Генерация {task_id} удалена из активных после завершения")
        
        # Запускаем новую генерацию в отдельном потоке
        new_inference_thread = threading.Thread(target=run_inference, daemon=True)
        new_inference_thread.start()
        
        # Добавляем информацию о генерации
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
        
        print(f"[StreamingAPI] Генерация {task_id} {'перезапущена' if has_active_generation else 'запущена'}")
        
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
        print(f"[StreamingAPI] Ошибка при перезапуске генерации: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/webrtc/upload_audio_and_generate")
async def upload_audio_and_generate(
    task_id: str = Form(...),
    audio_file: UploadFile = File(...),
    version: str = Form(default="v15"),
    batch_size: int = Form(default=8),
    fps: int = Form(default=25)
):
    """
    Принимает аудио файл и запускает генерацию в трансляцию для существующего WebRTC соединения.
    
    Args:
        task_id: ID задачи (должно существовать активное WebRTC соединение)
        audio_file: Аудио файл для обработки
        version: Версия модели (по умолчанию v15)
        batch_size: Размер батча для генерации (по умолчанию 8)
        fps: FPS для генерации (по умолчанию 25)
    
    Returns:
        JSON с результатом операции
    """
    if not WEBRTC_AVAILABLE:
        raise HTTPException(status_code=503, detail="WebRTC не доступен. Установите aiortc.")
    
    # Проверяем, что daemon_service инициализирован
    if daemon_service is None:
        raise HTTPException(status_code=500, detail="Daemon service не инициализирован")
    
    try:
        # Проверяем наличие соединения
        with connection_info_lock:
            if task_id not in connection_info_by_task:
                raise HTTPException(
                    status_code=404, 
                    detail=f"Соединение для task_id={task_id} не найдено. Сначала установите WebRTC соединение через /api/webrtc/offer"
                )
            
            conn_info = connection_info_by_task[task_id]
            pc = conn_info["pc"]
            video_track = conn_info["video_track"]
            video_path = conn_info["video_path"]
        
        # Проверяем, что соединение активно
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
        
        # Создаем директорию для аудио файлов, если её нет
        audio_dir = "data/audio"
        os.makedirs(audio_dir, exist_ok=True)
        
        # Сохраняем аудио файл
        # Используем task_id и timestamp для уникального имени файла
        timestamp = int(time.time())
        file_extension = os.path.splitext(audio_file.filename)[1] if audio_file.filename else ".wav"
        audio_filename = f"{task_id}_{timestamp}{file_extension}"
        audio_path = os.path.join(audio_dir, audio_filename)
        
        # Читаем и сохраняем файл
        with open(audio_path, "wb") as f:
            content = await audio_file.read()
            f.write(content)
        
        print(f"[StreamingAPI] Аудио файл сохранен: {audio_path} (размер: {len(content)} байт)")
        
        # Проверяем существование файла
        if not os.path.exists(audio_path):
            raise HTTPException(status_code=500, detail="Не удалось сохранить аудио файл")
        
        # Обновляем audio_path в connection_info_by_task
        with connection_info_lock:
            if task_id in connection_info_by_task:
                connection_info_by_task[task_id]["audio_path"] = audio_path
                print(f"[StreamingAPI] audio_path обновлен для task_id={task_id}: {audio_path}")
        
        # Проверяем состояние video_track перед запуском генерации
        print(f"[StreamingAPI] Состояние video_track перед запуском: running={video_track.running}, queue_size={video_track.frame_queue.qsize()}")
        
        # Проверяем, есть ли активная генерация
        with generation_lock:
            has_active_generation = task_id in active_generations
            
            if has_active_generation:
                # Останавливаем текущую генерацию
                print(f"[StreamingAPI] Остановка текущей генерации для task_id={task_id}")
                gen_info = active_generations[task_id]
                stop_flag = gen_info["stop_flag"]
                stop_flag.set()
                
                # Ждем завершения потока (с таймаутом)
                inference_thread = gen_info["inference_thread"]
                if inference_thread.is_alive():
                    inference_thread.join(timeout=5.0)
                    if inference_thread.is_alive():
                        print(f"[StreamingAPI] Предупреждение: поток генерации не завершился за 5 секунд")
                
                # Удаляем из активных генераций
                del active_generations[task_id]
            else:
                print(f"[StreamingAPI] Новая генерация для task_id={task_id}")
                # Создаем новый флаг остановки
                stop_flag = threading.Event()
            
            # Очищаем очередь кадров
            queue_cleared = 0
            while not video_track.frame_queue.empty():
                try:
                    video_track.frame_queue.get_nowait()
                    queue_cleared += 1
                except:
                    break
            if queue_cleared > 0:
                print(f"[StreamingAPI] Очищено {queue_cleared} кадров из очереди")
            
            # Убеждаемся, что video_track активен для новой генерации
            if not video_track.running:
                print(f"[StreamingAPI] ВНИМАНИЕ: video_track.running = False, устанавливаем в True для запуска")
                video_track.running = True
                # Также нужно сбросить флаг first_real_frame_sent для новой генерации
                with video_track.first_real_frame_lock:
                    video_track.first_real_frame_sent = False
                # Сбрасываем флаг завершения генерации для новой генерации
                with video_track.generation_completed_lock:
                    video_track.generation_completed = False
                print(f"[StreamingAPI] video_track активирован для новой генерации")
            else:
                print(f"[StreamingAPI] video_track уже активен (running=True)")
                # Все равно сбрасываем флаг first_real_frame_sent для новой генерации
                with video_track.first_real_frame_lock:
                    video_track.first_real_frame_sent = False
                # Сбрасываем флаг завершения генерации для новой генерации
                with video_track.generation_completed_lock:
                    video_track.generation_completed = False
        
        # Обновляем аудио трек, если он существует
        with connection_info_lock:
            if task_id in connection_info_by_task:
                audio_track = connection_info_by_task[task_id].get("audio_track")
                
                if audio_track:
                    # Для SynchronizedAudioTrack используем метод reset()
                    if hasattr(audio_track, 'reset'):
                        audio_track.reset()
                        print(f"[StreamingAPI] SynchronizedAudioTrack перезапущен через reset()")
                    else:
                        # Для других типов треков используем старую логику
                        if hasattr(audio_track, 'audio_started'):
                            audio_track.audio_started = False
                            print(f"[StreamingAPI] Флаг audio_started сброшен в аудио треке для перезапуска")
                        # Для SynchronizedAudioTrack сбрасываем позицию
                        if hasattr(audio_track, 'current_frame_index'):
                            audio_track.current_frame_index = 0
                            if hasattr(audio_track, 'audio_position'):
                                audio_track.audio_position = 0
                            print(f"[StreamingAPI] Позиция аудио сброшена для перезапуска")
                        # Для DelayedMediaPlayerTrack сбрасываем счетчик тишины и перезапускаем MediaPlayer
                        if hasattr(audio_track, 'silence_frames_sent'):
                            audio_track.silence_frames_sent = 0
                            print(f"[StreamingAPI] Счетчик тишины сброшен для перезапуска")
                        # Сбрасываем счетчик вызовов recv для логирования
                        if hasattr(audio_track, '_recv_call_count'):
                            audio_track._recv_call_count = 0
                            print(f"[StreamingAPI] Счетчик вызовов recv сброшен для перезапуска")
                    # Сбрасываем флаги и счетчики ошибок для DelayedMediaPlayerTrack
                    if hasattr(audio_track, 'media_player_ended'):
                        audio_track.media_player_ended = False
                        print(f"[StreamingAPI] Флаг media_player_ended сброшен для перезапуска")
                    if hasattr(audio_track, 'error_count'):
                        audio_track.error_count = 0
                    if hasattr(audio_track, 'last_error_log_time'):
                        audio_track.last_error_log_time = 0
                    # Для DelayedMediaPlayerTrack перезапускаем MediaPlayer
                    if hasattr(audio_track, 'media_player') and audio_track.media_player:
                        try:
                            print(f"[StreamingAPI] Перезапуск MediaPlayer для DelayedMediaPlayerTrack")
                            # Останавливаем и закрываем текущий MediaPlayer
                            old_media_player = audio_track.media_player
                            try:
                                if hasattr(old_media_player, 'audio') and old_media_player.audio:
                                    old_media_player.audio.stop()
                            except Exception as e:
                                print(f"[StreamingAPI] Ошибка при остановке старого MediaPlayer: {e}")
                            
                            try:
                                if hasattr(old_media_player, 'close'):
                                    old_media_player.close()
                            except Exception as e:
                                print(f"[StreamingAPI] Ошибка при закрытии старого MediaPlayer: {e}")
                            
                            # Создаем новый MediaPlayer для перезапуска
                            from aiortc.contrib.media import MediaPlayer
                            new_media_player = MediaPlayer(audio_path)
                            if new_media_player and new_media_player.audio:
                                audio_track.media_player = new_media_player
                                # Сбрасываем флаги для нового MediaPlayer
                                if hasattr(audio_track, 'media_player_ended'):
                                    audio_track.media_player_ended = False
                                if hasattr(audio_track, 'error_count'):
                                    audio_track.error_count = 0
                                if hasattr(audio_track, 'last_error_log_time'):
                                    audio_track.last_error_log_time = 0
                                print(f"[StreamingAPI] ✓ MediaPlayer перезапущен для нового аудио (новый MediaPlayer создан)")
                            else:
                                print(f"[StreamingAPI] ⚠️ Не удалось создать новый MediaPlayer для перезапуска")
                        except Exception as e:
                            print(f"[StreamingAPI] Ошибка при перезапуске MediaPlayer: {e}")
                            import traceback
                            traceback.print_exc()
        
        # Запускаем новую генерацию
        def run_inference():
            try:
                print(f"[StreamingAPI] Запуск генерации видео для task_id={task_id} с аудио {audio_path}")
                
                def frame_callback(frame_array, frame_index):
                    """Callback для добавления кадра в WebRTC поток"""
                    try:
                        # Проверяем флаг остановки
                        if stop_flag.is_set():
                            print(f"[StreamingAPI] Получен сигнал остановки генерации")
                            return
                        
                        # Используем правильный event loop для добавления кадра
                        video_track.add_frame(frame_array)
                        if frame_index % 30 == 0:  # Логируем каждые 30 кадров
                            print(f"[StreamingAPI] Отправлен кадр #{frame_index}")
                    except Exception as e:
                        print(f"[StreamingAPI] Ошибка при отправке кадра {frame_index}: {e}")
                        import traceback
                        traceback.print_exc()
                
                print(f"[StreamingAPI] Вызов process_request_webrtc для запуска генерации...")
                daemon_service.process_request_webrtc(
                    task_id=task_id,
                    video_path=video_path,
                    audio_path=audio_path,
                    frame_callback=frame_callback,
                    version=version,
                    batch_size=batch_size,
                    fps=fps,
                    video_track=video_track,
                    stop_flag=stop_flag
                )
                print(f"[StreamingAPI] Генерация видео завершена")
                
                # Отмечаем, что генерация завершена - теперь можно использовать оригинальное видео
                with video_track.generation_completed_lock:
                    video_track.generation_completed = True
                    print(f"[StreamingAPI] ✓ Флаг generation_completed установлен - будет использоваться оригинальное видео")
                    print(f"[StreamingAPI] Состояние: first_real_frame_sent={video_track.first_real_frame_sent}, generation_completed={video_track.generation_completed}")
            except Exception as e:
                print(f"[StreamingAPI] Ошибка при запуске инференса: {e}")
                import traceback
                traceback.print_exc()
            finally:
                # Удаляем из активных генераций
                with generation_lock:
                    if task_id in active_generations:
                        del active_generations[task_id]
                        print(f"[StreamingAPI] Генерация {task_id} удалена из активных после завершения")
        
        # Запускаем новую генерацию в отдельном потоке
        new_inference_thread = threading.Thread(target=run_inference, daemon=True)
        new_inference_thread.start()
        
        # Добавляем информацию о генерации
        with generation_lock:
            active_generations[task_id] = {
                "video_track": video_track,
                "audio_track": conn_info.get("audio_track"),
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
        
        print(f"[StreamingAPI] Генерация {task_id} запущена с новым аудио файлом")
        
        return {
            "status": "ok",
            "message": f"Аудио файл загружен и генерация для task_id={task_id} запущена",
            "task_id": task_id,
            "audio_path": audio_path,
            "audio_size": len(content)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"[StreamingAPI] Ошибка при загрузке аудио и запуске генерации: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Проверка здоровья сервиса"""
    return {
        "status": "ok",
        "daemon_initialized": daemon_service is not None,
        "webrtc_available": WEBRTC_AVAILABLE,
        "active_connections": len(active_connections),
        "active_generations": len(active_generations)
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="WebRTC Streaming API для MuseTalk")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Хост для сервера")
    parser.add_argument("--port", type=int, default=8009, help="Порт для сервера")
    parser.add_argument("--version", type=str, default="v15", help="Версия модели")
    parser.add_argument("--gpu_id", type=int, default=0, help="ID GPU")
    parser.add_argument("--use_float16", action="store_true", help="Использовать float16")
    
    args = parser.parse_args()
    
    if not WEBRTC_AVAILABLE:
        print("[StreamingAPI] ОШИБКА: aiortc не установлен!")
        print("[StreamingAPI] Установите: pip install aiortc")
        sys.exit(1)
    
    # Инициализируем сервис перед запуском
    init_daemon_service(
        version=args.version,
        gpu_id=args.gpu_id,
        use_float16=args.use_float16
    )
    
    print(f"[StreamingAPI] Запуск WebRTC сервера на {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)

