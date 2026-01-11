
import logging
import asyncio
import os
import time
import yaml
import traceback
from typing import Dict, Optional
import json
import json
import redis
import cv2

from scripts.mmap_frame_buffer import MmapFrameReader, get_mmap_path

logger = logging.getLogger("StreamingAPI")

class InferenceService:
    """Сервис для управления процессом генерации (инференса) с использованием asyncio"""
    
    def __init__(self, config):
        self.config = config
        
        # Управление активными генерациями (по task_id)
        self._active_generations: Dict[str, Dict] = {}
        # В asyncio блокировки обычно не нужны для dict, если операции атомарны в рамках event loop,
        # но для совместимости и безопасности оставим Lock (asyncio.Lock если нужно, но здесь обычного dict достаточно)
        self._generations_lock = asyncio.Lock()
        
        # Redis connection
        redis_host = os.getenv("REDIS_HOST", "localhost")
        redis_port = int(os.getenv("REDIS_PORT", 6379))
        redis_password = os.getenv("REDIS_PASSWORD", None)
        
        # Construct URL or use kwargs (Redis-py handles kwargs better for password)
        # Using URL for consistency if REDIS_URL provided, else build it
        redis_url = os.getenv("REDIS_URL")

        try:
            if redis_url:
                self.redis = redis.from_url(redis_url)
            else:
                self.redis = redis.Redis(host=redis_host, port=redis_port, password=redis_password, decode_responses=True)

            self.redis.ping()
            logger.info(f"Подключено к Redis: {redis_host}:{redis_port}")
        except Exception as e:
            logger.error(f"Ошибка подключения к Redis: {e}")
            self.redis = None
        
    async def start_generation(self, task_id: str, video_track, audio_track, video_path: str, audio_path: str, 
                         fps: int = 25, batch_size: int = 4, version: str = "v1.5", has_active_generation: bool = False, capture_video_path: str = None):
        """Запускает процесс генерации в фоновой asyncio задаче"""
        
        # Создаем задачу
        inference_task = asyncio.create_task(
            self._run_inference(task_id, video_track, audio_track, video_path, audio_path, fps, batch_size, version, has_active_generation, capture_video_path)
        )
        
        async with self._generations_lock:
            self._active_generations[task_id] = {
                "video_track": video_track,
                "audio_track": audio_track,
                "inference_task": inference_task,
                "task_id": task_id,
                "video_path": video_path,
                "audio_path": audio_path,
                "version": version,
                "batch_size": batch_size,
                "fps": fps
            }
            
        return inference_task

    async def _run_inference(self, task_id: str, video_track, audio_track, video_path: str, audio_path: str, fps: int, batch_size: int, version: str, has_active_generation: bool, capture_video_path: str = None):
        """Внутренняя корутина генерации"""
        mmap_reader = None
        
        try:
            logger.info(f"{'Перезапуск' if has_active_generation else 'Запуск'} генерации видео для task_id={task_id} (через демон, mmap)")
            
            config_data = {
                "webrtc_mode": True,
                "batch_size": batch_size,
                "fps": fps,
                task_id: {
                    "video_path": video_path,
                    "audio_path": audio_path
                }
            }
            
            
            # Вместо записи файла отправляем задачу в Redis
            if self.redis:
                try:
                    # Проверяем длину очереди для диагностики
                    queue_len = self.redis.llen("musetalk:queue")
                    logger.info(f"Текущая длина очереди в Redis: {queue_len}")
                    
                    # Добавляем данные в очередь
                    # Используем lpush (или rpush)
                    self.redis.rpush("musetalk:queue", json.dumps(config_data))
                    logger.info(f"Задача {task_id} отправлена в очередь Redis (musetalk:queue)")
                except Exception as e:
                    logger.error(f"Ошибка отправки задачи в Redis: {e}")
                    raise RuntimeError(f"Failed to push task to Redis: {e}")
            else:
                # Fallback to file if redis not available? Or just error.
                # Let's try to fallback to file logic if redis is None, but user asked for redis.
                # But to stay safe, let's keep only redis as requested.
                logger.error("Redis client is not initialized")
                raise RuntimeError("Redis not available")
            
            # request_config_path = os.path.join(self.config.requests_dir, f"webrtc_{task_id}_{int(time.time())}.yaml")
            # os.makedirs(self.config.requests_dir, exist_ok=True)
            
            # # Блокирующая операция ввода-вывода должна быть вынесена в тред, если она долгая, но запись мелкого файла ok.
            # # Для строгости можно использовать run_in_executor
            # # await asyncio.to_thread(self._write_yaml, request_config_path, config_data)
            
            # os.chmod(request_config_path, 0o644)
            
            await asyncio.sleep(2)
            
            mmap_file_path = get_mmap_path(task_id)
            
            logger.info(f"Ожидание создания mmap файла: {mmap_file_path}")
            
            max_wait_for_mmap = 300 # Увеличено до 5 минут, так как очередь может быть длинной
            wait_start = time.time()
            mmap_created = False
            last_log_time = wait_start
            log_interval = 10 # Увеличено до 10 сек, чтобы меньше спамить
            
            while time.time() - wait_start < max_wait_for_mmap:
                current_time = time.time()
                elapsed = current_time - wait_start
                
                if current_time - last_log_time >= log_interval:
                    logger.info(f"Ожидание mmap файла... прошло {elapsed:.1f} секунд из {max_wait_for_mmap}")
                    last_log_time = current_time
                
                # Проверки файлов блокирующие, но быстрые. Можно вынести в to_thread если критично.
                if os.path.exists(mmap_file_path):
                    file_size = os.path.getsize(mmap_file_path)
                    if file_size > 0:
                        try:
                            mmap_reader = MmapFrameReader(task_id)
                            status = mmap_reader.get_status()
                            if status.get("status") == "error":
                                error_msg = f"Демон сообщил об ошибке при обработке задачи {task_id}. Проверьте логи демона для деталей."
                                logger.error(f"ОШИБКА: {error_msg}")
                                logger.error(f"Статус из mmap: {status}")
                                raise RuntimeError(error_msg)
                            mmap_created = True
                            logger.info(f"Mmap файл успешно создан и открыт: {mmap_file_path} (размер: {file_size} байт)")
                            # logger.info(f"Статус: {status}") # Слишком шумно, убираем

                            break
                        except RuntimeError:
                            raise
                        except (ValueError, FileNotFoundError) as e:
                            logger.warning(f"Предупреждение: mmap файл существует, но не может быть открыт: {e}")
                            await asyncio.sleep(0.5)
                    else:
                        if elapsed > 10:
                            logger.warning(f"Предупреждение: mmap файл существует, но пуст (размер: {file_size} байт). Возможно, демон еще инициализируется или произошла ошибка.")
                        await asyncio.sleep(0.5)
                else:
                    mmap_dir = os.path.dirname(mmap_file_path)
                    if not os.path.exists(mmap_dir):
                        if elapsed > 5:
                            logger.warning(f"Предупреждение: директория для mmap файлов не существует: {mmap_dir}")
                    await asyncio.sleep(0.5)
            
            if not mmap_created:
                # (Логика ошибки та, же что и раньше, сокращено для краткости, можно скопировать из старого файла если нужно полно)
                raise RuntimeError(f"Mmap файл не был создан за {max_wait_for_mmap} сек")
            
            last_frame_index = -1
            max_wait_time = 300
            start_time = time.time()
            
            frame_batch = []
            batch_size_send = 8

            # video writer for capture
            video_writer = None
            if capture_video_path:
                try:
                    # width/height must be known. We can get it from first frame or if known.
                    # We will init lazily on first frame
                    logger.info(f"Will capture generated video to: {capture_video_path}")
                except Exception as e:
                    logger.error(f"Failed to setup video capture: {e}")
            
            while True:
                # Проверка отмены задачи
                if asyncio.current_task().cancelled():
                    logger.info(f"Задача генерации {task_id} отменена")
                    break
                
                if time.time() - start_time > max_wait_time:
                    logger.warning(f"Превышено время ожидания генерации ({max_wait_time}с)")
                    break
                
                # Чтение mmap - операция memory bound, быстрая
                status = mmap_reader.get_status()
                
                current_frame_index = last_frame_index + 1
                max_frames_to_read = status.get("frames_generated", 0)
                
                while current_frame_index < max_frames_to_read:
                    try:
                        frame_array = mmap_reader.read_frame(current_frame_index)
                        if frame_array is not None:
                            frame_batch.append(frame_array)
                            
                            # Capture frame
                            if capture_video_path:
                                try:
                                    if video_writer is None:
                                        h, w = frame_array.shape[:2]
                                        fourcc = cv2.VideoWriter_fourcc(*'mp4v') # or 'avc1' or 'H264'
                                        video_writer = cv2.VideoWriter(capture_video_path, fourcc, fps, (w, h))
                                        logger.info(f"Video writer initialized: {w}x{h} @ {fps}fps")
                                    
                                    if video_writer:
                                        # mmap returns RGB or BGR? 
                                        # Looking at mmap_frame_buffer.py (not shown but typically RGB/BGR mixup is common). 
                                        # Assuming BGR for cv2.VideoWriter as is standard for opencv.
                                        # StreamService converts to YUV420p from BGR24 usually.
                                        video_writer.write(frame_array)
                                except Exception as e:
                                    logger.error(f"Error writing frame to capture file: {e}")

                            last_frame_index = current_frame_index
                            current_frame_index += 1
                            
                            if len(frame_batch) >= batch_size_send:
                                # video_track.add_frames_batch теперь должен быть потокобезопасным или вызываться правильно
                                # Поскольку мы в event loop, и video_track использует call_soon_threadsafe или прямую работу с Queue
                                # Если Queue это asyncio.Queue, то можно делать put прямо здесь (await).
                                # Но VideoStreamGenerator использует asyncio.Queue и методы add_frames_batch рассчитаны на вызов из треда.
                                # Мы можем переписать add_frames_batch чтобы он был async или просто вызывать его.
                                # В текущей реализации add_frames_batch использует call_soon_threadsafe, что безопасно и из loop'а.
                                video_track.add_frames_batch(frame_batch)
                                frame_batch = []
                        else:
                            break
                    except Exception as e:
                        break
                
                if status.get("status") == "completed":
                    if frame_batch:
                        video_track.add_frames_batch(frame_batch)
                    
                    # Дочитываем остатки
                    while current_frame_index < status.get("total_frames", 0):
                        try:
                            frame_array = mmap_reader.read_frame(current_frame_index)
                            if frame_array is not None:
                                video_track.add_frame(frame_array)
                                
                                # Capture remaining frames
                                if capture_video_path and video_writer:
                                    video_writer.write(frame_array)
                                    
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
                
                await asyncio.sleep(0.01)

            if video_writer:
                video_writer.release()
                logger.info(f"Video capture saved to {capture_video_path}")

            with video_track.generation_completed_lock:
                video_track.generation_completed = True
            
            logger.info(f"Генерация завершена. Всего кадров отправлено в поток: {self.frame_count if 'self.frame_count' in locals() else 'N/A'}")
            
        except asyncio.CancelledError:
            logger.info(f"Генерация {task_id} была отменена")
            raise
        except Exception as e:
            logger.error(f"Ошибка в инференс-задаче: {e}", exc_info=True)
        finally:
            if mmap_reader:
                mmap_reader.close()
                try:
                    mmap_reader.cleanup()
                except Exception as e:
                    pass
            
            async with self._generations_lock:
                if task_id in self._active_generations:
                    # Проверяем, что удаляем именно ту задачу, которую запустили (хотя task_id уникальный)
                    del self._active_generations[task_id]

    def _write_yaml(self, path, data):
        with open(path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)

    async def stop_generation(self, task_id: str):
        """Останавливает активную генерацию"""
        async with self._generations_lock:
            gen_info = self._active_generations.get(task_id)
            if gen_info:
                task = gen_info.get("inference_task")
                if task and not task.done():
                    task.cancel()
                    logger.info(f"Отправлен сигнал отмены для task_id={task_id}")
                    try:
                        # Ждем завершения с таймаутом, чтобы не висеть
                        await asyncio.wait_for(task, timeout=2.0)
                    except (asyncio.CancelledError, asyncio.TimeoutError):
                        pass
                    except Exception as e:
                        logger.error(f"Ошибка при ожидании отмены задачи {task_id}: {e}")
                
                if task_id in self._active_generations:
                    del self._active_generations[task_id]
            else:
                logger.info(f"Генерация не найдена для остановки: task_id={task_id}")
        
        # Отправляем сигнал остановки в Redis (чтобы прервать демона)
        if self.redis:
            try:
                stop_cmd = json.dumps({"command": "stop", "task_id": task_id})
                self.redis.rpush("musetalk:queue", stop_cmd)
                logger.info(f"Отправлена команда STOP в Redis для task_id={task_id}")
            except Exception as e:
                logger.error(f"Ошибка отправки STOP в Redis: {e}")

    async def is_generation_active(self, task_id: str) -> bool:
        async with self._generations_lock:
            return task_id in self._active_generations
            
    async def get_active_count(self) -> int:
        async with self._generations_lock:
            return len(self._active_generations)
