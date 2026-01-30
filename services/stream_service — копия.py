
import asyncio
import threading
import logging
import time
import os
import cv2
import numpy as np
from fractions import Fraction
import av
from av import VideoFrame, AudioFrame
from aiortc import VideoStreamTrack, AudioStreamTrack
from aiortc.contrib.media import MediaPlayer

logger = logging.getLogger("StreamingAPI")

class VideoFrameCache:
    """Глобальный кэш для кадров видео, чтобы не загружать их для каждого клиента"""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(VideoFrameCache, cls).__new__(cls)
            cls._instance.cache = {} # (path, width, height) -> list[frames]
            cls._instance.lock = threading.Lock()
        return cls._instance
    
    def get_frames(self, video_path, width, height):
        key = (video_path, width, height)
        with self.lock:
            if key in self.cache:
                logger.info(f"[VideoFrameCache] Hit for {video_path} ({width}x{height})")
                return self.cache[key]
        return None
        
    def set_frames(self, video_path, width, height, frames):
        key = (video_path, width, height)
        with self.lock:
            self.cache[key] = frames
            logger.info(f"[VideoFrameCache] Cached {len(frames)} frames for {video_path} ({width}x{height}). Total keys: {len(self.cache)}")

video_frame_cache = VideoFrameCache()


def _safe_close_media_player(media_player):
    """Безопасно останавливает и закрывает MediaPlayer"""
    if media_player:
        try:
            if hasattr(media_player, 'audio') and media_player.audio:
                media_player.audio.stop()
        except Exception as e:
            logger.error(f"Ошибка при остановке аудио: {e}")
        
        try:
            if hasattr(media_player, 'close'):
                media_player.close()
        except Exception as e:
            logger.error(f"Ошибка при закрытии MediaPlayer: {e}")

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
        
        # Для AV синхронизации
        self.audio_samples_read = 0
        self.sync_threshold_seconds = 0.2 # Допустимый рассинхрон (аудио может бежать вперед на 0.2с)
        
        # Monotonic PTS counter
        self.next_pts = 0
        
        # Buffer for resampled frames
        from collections import deque
        self.frame_buffer = deque()
        self.resampler = None
        
        # Pacing
        self.start_time = None
        self.samples_sent = 0 # To track time
        self.pacing_lock = threading.Lock()
        
        self.initial_video_frames = 0 # Baseline for AV sync

    
    def _create_silence_frame(self, samples=480):
        """Создает кадр тишины с нулевыми данными"""
        silence = AudioFrame(format='s16', layout='stereo', samples=samples)
        silence.sample_rate = self.sample_rate
        silence.time_base = self.time_base
        
        # Use monotonic PTS
        silence.pts = self.next_pts
        self.next_pts += samples
        
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
            # Ждем пока видео не начнет отправлять реальные кадры
            while self.video_track and self.video_track.running and not self.video_track.first_real_frame_sent:
                await asyncio.sleep(0.01)
            
            # Если трек остановился пока ждали
            if self.video_track and not self.video_track.running:
                return self._create_silence_frame(480)

            self.audio_started = True
            
            # Capture baseline video frames
            if self.video_track and hasattr(self.video_track, 'content_frames_sent'):
                 with self.video_track.content_frames_lock:
                     self.initial_video_frames = self.video_track.content_frames_sent
            
            # Artificial Delay to compensate for Video Encoding Latency
            # Audio is much faster to encode/send. Video needs a head start.
            logger.info(f"[DelayedMediaPlayerTrack] ✓ Audio started. Baseline video frames: {self.initial_video_frames}. Waiting 200ms for video sync...")
            await asyncio.sleep(0.2)
            
            self.silence_frames_sent = 0  
            logger.info(f"[DelayedMediaPlayerTrack] ✓ Starting audio stream now.")
        
        if self.media_player_ended:
            return self._create_silence_frame(480)
            
        # --- AV SYNC CHECK ---
        if self.video_track and hasattr(self.video_track, 'content_frames_sent'):
            # Считаем текущее время аудио (сколько воспроизвели)
            audio_time = self.audio_samples_read / self.sample_rate
            
            # Считаем текущее время видео (сколько контентных кадров показали ОТНОСИТЕЛЬНО НАЧАЛА АУДИО)
            # Используем lock для атомарности
            with self.video_track.content_frames_lock:
                current_video_frames = self.video_track.content_frames_sent
            
            # Relative frames (since speech started)
            video_frames = current_video_frames - self.initial_video_frames
            if video_frames < 0: video_frames = 0
            
            video_fps = self.video_track.fps
            video_time = video_frames / video_fps
            
            # Если аудио убежало вперед видео больше чем на порог
            if audio_time > video_time + self.sync_threshold_seconds:
                # logger.debug(f"[AV-SYNC] Audio paused: A={audio_time:.3f}s > V={video_time:.3f}s (+{self.sync_threshold_seconds}s)")
                # Возвращаем тишину, НЕ читая из плеера (ставим на паузу)
                # Важно: не увеличиваем audio_samples_read, так как это тишина "вставки"
                return self._create_silence_frame(480)
        # ---------------------
        
        if self.media_player and self.media_player.audio:
            try:
                # --- PACING START ---
                if self.start_time is None:
                    self.start_time = time.time()
                
                # Expected time based on samples sent (at 48000Hz)
                # We strictly enforce 48000Hz output now
                expected_time_offset = self.samples_sent / 48000.0
                expected_time = self.start_time + expected_time_offset
                
                # Wait if we are ahead
                wait_time = expected_time - time.time()
                if wait_time > 0:
                     # Cap max wait to avoid long stalls? 
                     if wait_time > 0.5: wait_time = 0.5
                     await asyncio.sleep(wait_time)
                # --- PACING END ---
                
                # 1. Check if we have frames in buffer
                if self.frame_buffer:
                     out_frame = self.frame_buffer.popleft()
                     # Process this frame below
                else:
                    # 2. Read new frame from player
                    frame = await self.media_player.audio.recv()
                    self.error_count = 0
                    
                    # Ensure resampler
                    if not hasattr(self, 'resampler') or self.resampler is None:
                        logger.info(f"[DelayedMediaPlayerTrack] INPUT FRAME: rate={frame.sample_rate}, layout={frame.layout.name}, format={frame.format.name}, samples={frame.samples}")
                        self.resampler = av.AudioResampler(format='s16', layout='stereo', rate=48000)
                        logger.info(f"[DelayedMediaPlayerTrack] Created AudioResampler: Target 48000Hz, s16, stereo")

                    # Resample
                    resampled_frames = self.resampler.resample(frame)
                    
                    if not resampled_frames:
                        # Buffering in resampler? Return silence for now but don't increment audio_samples_read of content?
                        # Wait, we need to return SOMETHING.
                        return self._create_silence_frame(480)
                        
                    # Add all to buffer
                    self.frame_buffer.extend(resampled_frames)
                    out_frame = self.frame_buffer.popleft()
                
                audio_data = out_frame.to_ndarray()
                actual_samples = out_frame.samples
                
                # Ensure our track properties stay at 48k
                self.sample_rate = 48000
                self.time_base = Fraction(1, 48000)
                
                audio_frame = AudioFrame(format='s16', layout='stereo', samples=actual_samples)
                audio_frame.sample_rate = 48000
                audio_frame.time_base = Fraction(1, 48000)
                
                audio_frame.planes[0].update(audio_data.tobytes())
                
                # Overwrite PTS with monotonic counter
                audio_frame.pts = self.next_pts
                self.next_pts += actual_samples
                
                self.audio_samples_read += actual_samples
                self.samples_sent += actual_samples
                
                return audio_frame
            except Exception as copy_error:
                # Log detailed error
                logger.error(f"[DelayedMediaPlayerTrack] Conversion error: {repr(copy_error)}", exc_info=True)
                # Fallback to silence to be safe
                return self._create_silence_frame(480)

            except av.EOFError:
                self.media_player_ended = True
                logger.info(f"[DelayedMediaPlayerTrack] MediaPlayer закончил воспроизведение (EOF)")
                return self._create_silence_frame(480)
            except Exception as e:
                self.error_count += 1
                error_str = str(e).lower()
                if 'end' in error_str or 'eof' in error_str or 'finished' in error_str or (not str(e) and self._recv_call_count > 50):
                    self.media_player_ended = True
                    if self.error_count == 1:
                        logger.info(f"[DelayedMediaPlayerTrack] MediaPlayer закончил воспроизведение (нормально)")
                    return self._create_silence_frame(480)
                
                current_time = time.time()
                if current_time - self.last_error_log_time >= 1.0:
                    logger.error(f"[DelayedMediaPlayerTrack] Ошибка при получении кадра из MediaPlayer (ошибок: {self.error_count}): {repr(e)}")
                    self.last_error_log_time = current_time
                
                if self.error_count >= 10:
                    self.media_player_ended = True
                    logger.warning(f"[DelayedMediaPlayerTrack] Слишком много ошибок ({self.error_count}), считаем MediaPlayer завершенным")
                
                return self._create_silence_frame(480)
        else:
            return self._create_silence_frame(480)
    
    def stop(self):
        """Останавливает трек"""
        _safe_close_media_player(self.media_player)
        logger.info(f"[DelayedMediaPlayerTrack] Аудио трек остановлен")


class VideoStreamGenerator(VideoStreamTrack):
    """Генератор видео потока из кадров"""
    
    def __init__(self, fps=25, loop=None, video_width=640, video_height=480, video_path=None, preloaded_frames=None):
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
        self.original_video_frames = preloaded_frames if preloaded_frames is not None else []
        self.original_frame_index = 0
        self.first_real_frame_sent = False
        self.first_real_frame_lock = threading.Lock()
        self.generation_completed = False
        self.generation_completed_lock = threading.Lock()
        
        self.content_frames_sent = 0 
        self.content_frames_lock = threading.Lock()

        
        self.time_base = Fraction(1, 90000)
        
        self.conversion_times = []
        self.queue_wait_times = []
        self.last_metrics_log = time.time()
        self.metrics_lock = threading.Lock()
        
        logger.info(f"[VideoStreamGenerator] Инициализирован: fps={fps}, loop={loop is not None}, размер={video_width}x{video_height}, video_path={video_path}")
        print(f"DEBUG: VideoStreamGenerator init. Video path: {video_path}, Exists: {os.path.exists(video_path) if video_path else 'None'}")
        
        if len(self.original_video_frames) > 0:
             logger.info(f"[VideoStreamGenerator] Using {len(self.original_video_frames)} preloaded frames")
             # Init last frame immediately
             with self.last_frame_lock:
                self.last_frame = self.original_video_frames[0].copy()
                
        elif video_path and os.path.exists(video_path):
            self.loading_future = None
            if loop:
                 logger.info(f"[VideoStreamGenerator] Scheduling async video loading: {video_path}")
                 self.loading_future = loop.run_in_executor(None, self._load_original_video_frames)
            else:
                 logger.warning("[VideoStreamGenerator] No loop provided, loading video synchronously (BLOCKING)")
                 self._load_original_video_frames()
        else:
            print(f"DEBUG: Video path not found or None: {video_path}")

        
        if loop:
            print("DEBUG: Loop provided, scheduling tasks...")
            async def send_test():
                await self._send_test_frame()
            asyncio.run_coroutine_threadsafe(send_test(), loop)
            
            def schedule_keepalive():
                print("DEBUG: Scheduling keepalive")
                self.keepalive_task = loop.create_task(self._start_keepalive())
            loop.call_soon_threadsafe(schedule_keepalive)
            
            def schedule_original_video():
                print("DEBUG: Scheduling original video playback")
                self.original_video_task = loop.create_task(self._play_original_video())
            loop.call_soon_threadsafe(schedule_original_video)
        else:
            print("DEBUG: NO LOOP PROVIDED to VideoStreamGenerator!")
    
    def _load_original_video_frames(self):
        """Загружает все кадры из оригинального видео для использования в тестовых кадрах (с зацикливанием)"""
        # Если кадры уже есть (переданы извне), не грузим
        if self.original_video_frames:
            return

        logger.info(f"[VideoStreamGenerator] Start loading frames from {self.video_path} ...")
        start_t = time.time()
        try:
            # Check cache first (double check if logic changed)
            frames = video_frame_cache.get_frames(self.video_path, self.video_width, self.video_height)
            if frames:
                self.original_video_frames = frames
                logger.info(f"[VideoStreamGenerator] ✓ Loaded {len(frames)} frames from CACHE")
                if len(self.original_video_frames) > 0:
                     with self.last_frame_lock:
                        self.last_frame = self.original_video_frames[0].copy()
                return

            video_cap = cv2.VideoCapture(self.video_path)
            if not video_cap.isOpened():
                logger.warning(f"[VideoStreamGenerator] ⚠️ Не удалось открыть видео для загрузки кадров: {self.video_path}")
                return
            
            total_frames = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            # video_fps = video_cap.get(cv2.CAP_PROP_FPS)
            
            local_frames = []
            
            while True:
                ret, frame = video_cap.read()
                if not ret:
                    break
                
                if frame.shape[1] != self.video_width or frame.shape[0] != self.video_height:
                    frame = cv2.resize(frame, (self.video_width, self.video_height))
                
                local_frames.append(frame.copy())
                
            video_cap.release()
            
            # Atomic update of the list (assignment is atomic)
            self.original_video_frames = local_frames
            
            # Save to cache
            if len(local_frames) > 0:
                video_frame_cache.set_frames(self.video_path, self.video_width, self.video_height, local_frames)
            
            if len(self.original_video_frames) > 0:
                logger.info(f"[VideoStreamGenerator] ✓ Загружено {len(self.original_video_frames)} кадров из оригинального видео за {time.time() - start_t:.2f}s")
                with self.last_frame_lock:
                    self.last_frame = self.original_video_frames[0].copy()
            else:
                 logger.warning(f"[VideoStreamGenerator] Video loaded but 0 frames found.")
                 
        except Exception as e:
            logger.error(f"[VideoStreamGenerator] Ошибка при загрузке кадров из оригинального видео: {e}", exc_info=True)
    
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
                logger.error(f"[VideoStreamGenerator] Ошибка при воспроизведении оригинального видео: {e}")
                await asyncio.sleep(1.0)
        
        logger.info(f"[VideoStreamGenerator] Задача воспроизведения оригинального видео остановлена")
    
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
                logger.error(f"[VideoStreamGenerator] Keepalive: неожиданная ошибка: {e}")
                await asyncio.sleep(1.0)
        
        logger.info(f"[VideoStreamGenerator] Keepalive задача остановлена")
    
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
            
            logger.info(f"[VideoStreamGenerator] 📊 МЕТРИКИ ПРОИЗВОДИТЕЛЬНОСТИ:")
            logger.info(f"  - Очередь: {queue_size}/{queue_capacity} ({queue_usage:.1f}%)")
            logger.info(f"  - Конвертация кадров: avg={avg_conv*1000:.2f}ms, min={min_conv*1000:.2f}ms, max={max_conv*1000:.2f}ms")
            logger.info(f"  - Ожидание в очереди: avg={avg_wait*1000:.2f}ms, max={max_wait*1000:.2f}ms")
            logger.info(f"  - Текущий FPS: {current_fps:.2f} (целевой: {self.fps})")
            logger.info(f"  - Всего кадров отправлено: {self.frame_count}")
    
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
                logger.warning(f"[VideoStreamGenerator] ⚠️ Пропущено {frames_skipped} кадров из-за переполнения очереди!")
        
        try:
            self.loop.call_soon_threadsafe(add_batch_non_blocking)
        except Exception as e:
            logger.error(f"[VideoStreamGenerator] Ошибка при добавлении батча кадров: {e}")
    
    def __repr__(self):
        return f"VideoStreamGenerator(fps={self.fps}, running={self.running}, queue_size={self.frame_queue.qsize()})"
        
    async def recv(self):
        """Получает следующий кадр для отправки"""
        self._recv_called = True
        
        if not self.running:
            raise Exception("Stream stopped")
        
        if self.start_time is None:
            self.start_time = time.time()
            print("DEBUG: VideoStreamGenerator.recv called for the first time!")
        
        # print("DEBUG: recv called") # Uncomment for spam
        
        try:
            # --- PACING LOGIC ---
            # Строгое соблюдение тайминга 25 FPS
            # Если мы бежим вперед, ждем своего времени
            expected_time = self.start_time + (self.frame_count * self.frame_time)
            wait_time = expected_time - time.time()
            if wait_time > 0:
                await asyncio.sleep(wait_time)
            # --------------------

            queue_wait_start = time.time()
            frame_array = await asyncio.wait_for(self.frame_queue.get(), timeout=0.5)
            queue_wait_time = time.time() - queue_wait_start

            # Это контентный кадр, увеличиваем счетчик
            with self.content_frames_lock:
                self.content_frames_sent += 1
            
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
            logger.error(f"[VideoStreamGenerator] recv: КРИТИЧЕСКАЯ ОШИБКА: {e}", exc_info=True)
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
                    logger.error(f"[VideoStreamGenerator] Ошибка при добавлении кадра в очередь: {e}")

            self.loop.call_soon_threadsafe(put_frame_non_blocking)
            
        except Exception as e:
            logger.error(f"[VideoStreamGenerator] Ошибка при обработке кадра: {e}", exc_info=True)
    
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
        
        logger.info(f"[VideoStreamGenerator] Трек остановлен, очередь очищена, keepalive остановлен")


    async def play_video_file(self, video_path: str, fps: int = 25):
        """
        Воспроизводит видео из файла в существующий трек.
        Синхронно читает кадры, но отправляет их с учетом FPS.
        """
        if not os.path.exists(video_path):
            logger.error(f"[StreamService] play_video_file: File not found: {video_path}")
            return

        logger.info(f"[StreamService] Starting playback from file: {video_path}")
        cap = cv2.VideoCapture(video_path)
        
        frame_interval = 1.0 / fps
        next_frame_time = time.time()
        
        try:
            while cap.isOpened() and self.running:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Check cancellation?
                
                # Send frame
                self.add_frame(frame)
                
                # Pacing
                next_frame_time += frame_interval
                wait = next_frame_time - time.time()
                if wait > 0:
                    await asyncio.sleep(wait)
                else:
                    # We are lagging, yield control at least
                    await asyncio.sleep(0.001)
                    
        except asyncio.CancelledError:
            logger.info("[StreamService] Playback cancelled")
        except Exception as e:
            logger.error(f"[StreamService] Error during file playback: {e}")
        finally:
            cap.release()
            # Mark completion
            with self.generation_completed_lock:
                self.generation_completed = True
            
            logger.info(f"[StreamService] Playback finished for {video_path}")

class StreamService:
    """Сервис для управления медиа-потоками"""
    
    def __init__(self):
        pass
        
    def create_video_track(self, fps=25, loop=None, video_width=640, video_height=480, video_path=None):
        # Auto-detect resolution if video_path exists
        if video_path and os.path.exists(video_path):
            try:
                cap = cv2.VideoCapture(video_path)
                if cap.isOpened():
                    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    
                    if orig_width > 0 and orig_height > 0:
                        video_width = orig_width
                        video_height = orig_height
                        logger.info(f"[StreamService] Auto-detected video resolution: {video_width}x{video_height} from {video_path}")
                    cap.release()
            except Exception as e:
                logger.warning(f"[StreamService] Failed to auto-detect video resolution: {e}. Using default {video_width}x{video_height}")
                
        # Try to get frames from cache immediately to avoid async complexity inside generator if possible
        # However, if not cached, we let generator do it in background because reading file is slow.
        # But wait, if we read here, we block. If we let generator do it, it does it in executor.
        # We can enable generator to use cache as well.
        # Let's pass preloaded frames if available in cache.
        
        preloaded_frames = video_frame_cache.get_frames(video_path, video_width, video_height)
        
        return VideoStreamGenerator(fps, loop, video_width, video_height, video_path, preloaded_frames=preloaded_frames)
    
    def create_audio_track(self, media_player, video_track):
        return DelayedMediaPlayerTrack(media_player, video_track)
    
    def reset_audio_track(self, audio_track, new_audio_path=None):
        """Сбрасывает состояние аудио трека для перезапуска"""
        if not audio_track:
            return

        logger.info(f"Сброс состояния аудио трека...")
        
        if hasattr(audio_track, 'audio_started'):
            audio_track.audio_started = False
            logger.info(f"Флаг audio_started сброшен")

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
        if hasattr(audio_track, 'audio_samples_read'):
            audio_track.audio_samples_read = 0
            
        if hasattr(audio_track, 'frame_buffer'):
            audio_track.frame_buffer.clear()
            
        if hasattr(audio_track, 'samples_sent'):
            audio_track.samples_sent = 0
        if hasattr(audio_track, 'start_time'):
            audio_track.start_time = None
        if hasattr(audio_track, 'resampler'):
            audio_track.resampler = None
        if hasattr(audio_track, 'initial_video_frames'):
            audio_track.initial_video_frames = 0
            
        # IMPORTANT: Do NOT reset next_pts. We want strictly monotonic timestamps 
        # to ensure the browser sees a continuous stream of audio.
        # if hasattr(audio_track, 'next_pts'): pass

        
        # Перезапуск MediaPlayer для DelayedMediaPlayerTrack
        # Перезапуск MediaPlayer для DelayedMediaPlayerTrack
        # Если есть новый путь, или если уже был плеер - пытаемся создать/обновить
        old_media_player = getattr(audio_track, 'media_player', None)
        path_to_use = new_audio_path
        
        # Если нового пути нет, но был старый плеер, берем путь оттуда
        if path_to_use is None and old_media_player and hasattr(old_media_player, '_path'):
            path_to_use = old_media_player._path
            
        if path_to_use:
             if old_media_player:
                _safe_close_media_player(old_media_player)
            
             try:
                new_media_player = MediaPlayer(path_to_use)
                if new_media_player and new_media_player.audio:
                    audio_track.media_player = new_media_player
                    # Сохраняем путь к файлу в объекте плеера для будущего использования (если библиотека не сохраняет)
                    if not hasattr(new_media_player, '_path'):
                        new_media_player._path = path_to_use
                    logger.info(f"✓ MediaPlayer перезапущен с аудио: {path_to_use}")
                else:
                    logger.warning(f"⚠️ Не удалось создать новый MediaPlayer для {path_to_use}")
             except Exception as e:
                logger.error(f"Ошибка при перезапуске MediaPlayer: {e}", exc_info=True)
        else:
            logger.info("MediaPlayer не создан: нет аудио файла")
                
    def safe_close_media_player(self, media_player):
        _safe_close_media_player(media_player)
