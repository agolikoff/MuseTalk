#!/usr/bin/env python3
"""
Демон-сервис для предзагрузки моделей MuseTalk.
Модели загружаются один раз при старте и остаются в памяти.
Сервис следит за директорией запросов и обрабатывает их по мере поступления.
"""
import signal
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
import os
import json
import time
import argparse
import subprocess
import threading
import queue
import shutil
import shlex
import redis
import hashlib
from pathlib import Path
from omegaconf import OmegaConf
import torch
from transformers import WhisperModel
import torchvision.transforms as transforms
# Используем простой polling вместо watchdog для совместимости

from musetalk.utils.blending import get_image_blending, get_image
from musetalk.utils.face_parsing import FaceParsing
from musetalk.utils.audio_processor import AudioProcessor
from musetalk.utils.utils import get_file_type, get_video_fps, datagen, load_all_model
from musetalk.utils.preprocessing import get_landmark_and_bbox, read_imgs, coord_placeholder
import cv2
import copy
import glob
import pickle
import numpy as np
from tqdm import tqdm
import gc

from musetalk.utils.blending import get_image_prepare_material
import shutil
import glob
import pickle
import copy
import threading

class AvatarDaemon:
    """
    Класс, управляющий данными конкретного аватара.
    Реализует логику "Smart Caching": проверяет параметры и пересоздает материалы только при необходимости.
    """
    def __init__(self, task_id, video_path, result_dir, version, 
                 bbox_shift, left_cheek_width, right_cheek_width, parsing_mode, 
                 extra_margin, daemon_service):
        self.task_id = task_id
        self.video_path = video_path
        self.daemon = daemon_service # Доступ к моделям (vae, unet, pe, и т.д.)
        
        self.params = {
            "bbox_shift": bbox_shift,
            "left_cheek_width": left_cheek_width,
            "right_cheek_width": right_cheek_width,
            "version": version,
            "parsing_mode": parsing_mode,
            "extra_margin": extra_margin 
        }

        # Paths
        if version == "v15":
            self.base_path = f"{result_dir}/{version}/avatars/{task_id}"
        else:
            self.base_path = f"{result_dir}/avatars/{task_id}"
            
        self.full_imgs_path = f"{self.base_path}/full_imgs"
        self.coords_path = f"{self.base_path}/coords.pkl"
        self.latents_out_path = f"{self.base_path}/latents.pt"
        self.mask_out_path = f"{self.base_path}/mask"
        self.mask_coords_path = f"{self.base_path}/mask_coords.pkl"
        self.avatar_info_path = f"{self.base_path}/avatar_info.json" # NOTE: Using avatar_info.json
        
        # In-memory materials
        self.materials = None

    def prepare(self):
        """
        Проверяет актуальность кеша и загружает материалы.
        Поддерживает частичную регенерацию (только маски) если изменились только параметры маскирования.
        """
        action = "none" # none, partial, full
        
        if not os.path.exists(self.avatar_info_path):
            print(f"[Avatar {self.task_id}] {self.avatar_info_path} - Инфо не найдено. Полная генерация.")
            action = "full"
        else:
            try:
                with open(self.avatar_info_path, "r") as f:
                    saved_info = json.load(f)
                
                # Compare critical geometry parameters (requires full regen)
                geo_changed = False
                for key in ["bbox_shift", "extra_margin", "version"]:
                    if saved_info.get(key) != self.params.get(key):
                        print(f"[Avatar {self.task_id}] Геометрия изменилась: {key} ({saved_info.get(key)} -> {self.params.get(key)}). Полная перегенерация.")
                        geo_changed = True
                        break
                
                if geo_changed:
                    action = "full"
                else:
                    # Compare mask parameters (requires partial regen)
                    mask_changed = False
                    for key in ["left_cheek_width", "right_cheek_width", "parsing_mode"]:
                        if saved_info.get(key) != self.params.get(key):
                            print(f"[Avatar {self.task_id}] Маска изменилась: {key} ({saved_info.get(key)} -> {self.params.get(key)}). Обновление масок.")
                            mask_changed = True
                            break
                    
                    if mask_changed:
                        action = "partial"
                    else:
                        # Check critical files existence
                        required_files = [self.coords_path, self.latents_out_path, self.mask_coords_path]
                        if not all(os.path.exists(p) for p in required_files):
                            print(f"[Avatar {self.task_id}] Отсутствуют файлы кеша (хотя конфиг совпадает). Полная перегенерация.")
                            action = "full"

            except Exception as e:
                print(f"[Avatar {self.task_id}] Ошибка чтения инфо: {e}. Полная перегенерация.")
                action = "full"

        if action == "full":
            self._generate_materials(full=True)
        elif action == "partial":
            self._generate_materials(full=False)
        else:
            print(f"[Avatar {self.task_id}] Кеш актуален. Используем {self.base_path}")

        # Load into memory if not loaded
        if self.materials is None:
            self._load_materials()

    def _generate_materials(self, full=True):
        """
        Генерирует кадры, координаты, латенты и маски.
        full=True -> удалить всё и создать заново.
        full=False -> пересоздать только маски (используя существующие кадры/координаты).
        """
        print(f"[Avatar {self.task_id}] {'Полная' if full else 'Частичная'} генерация материалов...", flush=True)
        start_time = time.time()
        
        # Update FaceParsing mask parameters dynamically
        if self.daemon.fp:
            try:
                new_mask = self.daemon.fp._create_cheek_mask(
                    left_cheek_width=self.params["left_cheek_width"], 
                    right_cheek_width=self.params["right_cheek_width"]
                )
                self.daemon.fp.cheek_mask = new_mask
            except Exception as e:
                print(f"[Avatar {self.task_id}] Ошибка обновления маски FaceParsing: {e}")

        if full:
            # Clean directory
            if os.path.exists(self.base_path):
                try:
                     shutil.rmtree(self.base_path)
                except Exception as e:
                     print(f"Warning cleaning cached dir: {e}")
            
            os.makedirs(self.base_path, exist_ok=True)
            os.makedirs(self.full_imgs_path, exist_ok=True)
            os.makedirs(self.mask_out_path, exist_ok=True)
            
            # --- BASE GENERATION (Expensive) ---
            frames = self._ensure_base_materials()
        else:
            # Ensure mask dir exists (clearing it might be safer for partial too?)
            if os.path.exists(self.mask_out_path):
                 shutil.rmtree(self.mask_out_path)
            os.makedirs(self.mask_out_path, exist_ok=True)
            
            # Load frames/coords for mask generation
            # We need raw frames to generate masks
            img_list = sorted(glob.glob(os.path.join(self.full_imgs_path, '*.[jpJP][pnPN]*[gG]')))
            frames = read_imgs(img_list)
            if not frames:
                print("[Avatar] Ошибка: Не найдены кадры для частичного обновления. Откат к полной генерации.")
                self._generate_materials(full=True)
                return

        # --- MASK GENERATION (Cheap) ---
        self._ensure_masks(frames)
        
        # Save updated info
        with open(self.avatar_info_path, "w") as f:
            json.dump(self.params, f)

        print(f"[Avatar {self.task_id}] Генерация завершена. Время: {time.time() - start_time:.2f}с")

    def _ensure_base_materials(self):
        """Генерирует кадры, bbox'ы и латенты (то, что не зависит от cheek_mask)"""
        # 1. Extract Frames
        frames = []
        if os.path.isfile(self.video_path):
            cap = cv2.VideoCapture(self.video_path)
            count = 0
            while True:
                ret, frame = cap.read()
                if not ret: break
                cv2.imwrite(f"{self.full_imgs_path}/{count:08d}.png", frame)
                frames.append(frame)
                count += 1
            cap.release()
        elif os.path.isdir(self.video_path):
             img_exts = {'.png', '.jpg', '.jpeg'}
             src_files = sorted([f for f in os.listdir(self.video_path) if os.path.splitext(f)[1].lower() in img_exts])
             for i, fname in enumerate(src_files):
                 frame = cv2.imread(os.path.join(self.video_path, fname))
                 cv2.imwrite(f"{self.full_imgs_path}/{i:08d}.png", frame)
                 frames.append(frame)
        
        if not frames:
            raise ValueError(f"Не найдены кадры для {self.task_id}")

        # 2. Landmarks & BBox
        coord_list, _ = get_landmark_and_bbox(frames, upperbondrange=self.params["bbox_shift"])
        
        # 3. Latents & Adjusted BBox
        input_latent_list = []
        
        # Adjust bbox and generate latents
        for i, (bbox, frame) in enumerate(zip(coord_list, frames)):
            if bbox == coord_placeholder:
                pass 
            
            x1, y1, x2, y2 = bbox
            
            # V15 extra margin logic
            if self.params["version"] == "v15":
                y2 = y2 + self.params["extra_margin"]
                y2 = min(y2, frame.shape[0])
                coord_list[i] = [x1, y1, x2, y2]
            
            # Latent generation
            crop_frame = frame[y1:y2, x1:x2]
            resized = cv2.resize(crop_frame, (256, 256), interpolation=cv2.INTER_LANCZOS4)
            latents = self.daemon.vae.get_latents_for_unet(resized)
            input_latent_list.append(latents)

        # Save Latents & Coords
        with open(self.coords_path, 'wb') as f:
            pickle.dump(coord_list + coord_list[::-1], f)
            
        torch.save(input_latent_list + input_latent_list[::-1], self.latents_out_path)
        
        return frames

    def _ensure_masks(self, frames):
        """Генерирует только маски используя готовые frames и coords"""
        # Load coords (they might be newly generated or existing)
        with open(self.coords_path, 'rb') as f:
            # Load raw list (without cycle duplication since we enumerate frames)
            # Wait, pickle saved list+list[::-1]. We need original length.
            # Helper: just reload and take half? Or pass coord_list from _ensure_base_materials?
            # Ideally _ensure_base_materials returns coord_list.
            # But if partial, we load from file.
            coord_list_cycle = pickle.load(f)
            
        # Recover single loop coord_list
        coord_list = coord_list_cycle[:len(frames)]
        
        mask_list = []
        mask_coords_list = []
        
        mode = self.params["parsing_mode"] if self.params["version"] == "v15" else "raw"

        for i, (bbox, frame) in enumerate(zip(coord_list, frames)):
            if bbox == coord_placeholder:
                # Dummy mask/box?
                mask = np.zeros((256, 256, 1), dtype=np.uint8) # Placeholder?
                crop_box = [0,0,0,0]
            else:
                # get_image_prepare_material uses global bbox (x1,y1,x2,y2) 
                # and generates mask relative to it? 
                # Nope, it crops frame using bbox, then masks.
                mask, crop_box = get_image_prepare_material(frame, bbox, fp=self.daemon.fp, mode=mode)
            
            cv2.imwrite(f"{self.mask_out_path}/{i:08d}.png", mask)
            mask_list.append(mask)
            mask_coords_list.append(crop_box)
            
        with open(self.mask_coords_path, 'wb') as f:
            pickle.dump(mask_coords_list + mask_coords_list[::-1], f)

    def _load_materials(self):
        """Загружает данные в память."""
        # Load from disk
        with open(self.coords_path, 'rb') as f:
            coord_list_cycle = pickle.load(f)
            
        img_list = sorted(glob.glob(os.path.join(self.full_imgs_path, '*.[jpJP][pnPN]*[gG]')))
        frame_list_cycle = read_imgs(img_list)
        frame_list_cycle = frame_list_cycle + frame_list_cycle[::-1]
        
        input_latent_list_cycle = torch.load(self.latents_out_path)
        
        with open(self.mask_coords_path, 'rb') as f:
            mask_coords_list_cycle = pickle.load(f)
            
        mask_files = sorted(glob.glob(os.path.join(self.mask_out_path, '*.[jpJP][pnPN]*[gG]')))
        mask_list_cycle = read_imgs(mask_files)
        mask_list_cycle = mask_list_cycle + mask_list_cycle[::-1]
        
        self.materials = {
            'coord_list_cycle': coord_list_cycle,
            'frame_list_cycle': frame_list_cycle,
            'input_latent_list_cycle': input_latent_list_cycle,
            'mask_list_cycle': mask_list_cycle,
            'mask_coords_list_cycle': mask_coords_list_cycle
        }

    def inference(self, audio_path, batch_size, fps, device, process_callback=None):
        """
        Запускает инференс для заданного аудио.
        Returns: generator yielding frames (or handled by callback).
        """
        # Audio feature
        whisper_input_features, librosa_length = self.daemon.audio_processor.get_audio_feature(audio_path, weight_dtype=self.daemon.weight_dtype)
        whisper_chunks = self.daemon.audio_processor.get_whisper_chunk(
            whisper_input_features, device, self.daemon.weight_dtype, self.daemon.whisper, librosa_length, fps=fps
        )
        
        video_num = len(whisper_chunks)
        # Prepare cycles
        latent_cycle = self.materials['input_latent_list_cycle']
        
        # Batch generation
        gen = datagen(
            whisper_chunks=whisper_chunks,
            vae_encode_latents=latent_cycle,
            batch_size=batch_size,
            delay_frame=0,
            device=device
        )
        
        return gen, video_num

from scripts.mmap_frame_buffer import MmapFrameWriter

def fast_check_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except:
        return False


def format_time(seconds):
    """Форматирует время в читаемый вид"""
    if seconds < 60:
        return f"{seconds:.2f}с"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}м {secs:.2f}с"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}ч {minutes}м {secs:.2f}с"


# Простой polling механизм без внешних зависимостей


class ModelDaemonService:
    """Демон-сервис с предзагруженными моделями"""
    
    def __init__(self, version="v15", gpu_id=0, use_float16=True, 
                 whisper_dir="./models/whisper", vae_type="sd-vae",
                 ffmpeg_path="./ffmpeg-4.4-amd64-static/",
                 left_cheek_width=90, right_cheek_width=90,
                 request_dir="./requests", result_dir="./results",
                 skip_model_loading=False, redis_host="localhost", redis_port=6379, redis_password=None):
        self.version = version
        self.gpu_id = gpu_id
        self.use_float16 = use_float16
        self.whisper_dir = whisper_dir
        self.vae_type = vae_type
        self.ffmpeg_path = ffmpeg_path
        self.left_cheek_width = left_cheek_width
        self.right_cheek_width = right_cheek_width
        self.request_dir = request_dir
        self.result_dir = result_dir
        self.skip_model_loading = skip_model_loading
        
        # Создаем директории
        os.makedirs(request_dir, exist_ok=True)
        os.makedirs(result_dir, exist_ok=True)
        
        # Определяем пути к моделям
        if version == "v1.0" or version == "v1":
            self.model_dir = "./models/musetalk"
            self.unet_model_path = f"{self.model_dir}/pytorch_model.bin"
            self.unet_config = f"{self.model_dir}/musetalk.json"
            self.version_arg = "v1"
        elif version == "v1.5" or version == "v15":
            self.model_dir = "./models/musetalkV15"
            self.unet_model_path = f"{self.model_dir}/unet.pth"
            self.unet_config = f"{self.model_dir}/musetalk.json"
            self.version_arg = "v15"
        else:
            raise ValueError(f"Неверная версия: {version}. Используйте v1.0 или v1.5")
        
        # Инициализируем модели
        self.device = None
        self.vae = None
        self.unet = None
        self.pe = None
        self.whisper = None
        self.audio_processor = None
        self.fp = None
        self.timesteps = None
        self.weight_dtype = None
        
        # Блокировка для потокобезопасности
        self.lock = threading.RLock()
        
        # Отслеживание обработанных файлов
        # Отслеживание обработанных файлов
        self.processed_files = set()
        
        # Кеш для материалов аватара
        self.avatar_cache = {}

        # Redis initialization
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.redis_password = redis_password
        self.redis_client = None
        
        # Управление активными задачами для прерывания
        self.active_tasks = {} # task_id -> threading.Event
        self.active_tasks_lock = threading.Lock()

        # Initial Redis connection attempt
        self.redis_client = self._connect_to_redis()
        
    def _connect_to_redis(self):
        """Пытается подключиться к Redis и возвращает клиент или None"""
        try:
            client = None
            # Check for REDIS_URL first
            redis_url = os.getenv("REDIS_URL")
            if redis_url:
                 client = redis.from_url(redis_url)
                 print(f"[Daemon] Подключено к Redis через URL")
            else:
                print(f"[Daemon] Подключение к Redis {self.redis_host}:{self.redis_port}...")
                client = redis.Redis(host=self.redis_host, port=self.redis_port, password=self.redis_password, decode_responses=True)
            
            client.ping()
            print(f"[Daemon] Успешное подключение к Redis!")
            return client
        except Exception as e:
            print(f"[Daemon] Ошибка подключения к Redis: {e}. Будет использоваться только файловый режим (до восстановления связи).")
            return None
        
        # Кеш для материалов аватаров (загруженных в память)
        # Ключ: task_id, значение: словарь с материалами
        self.avatar_cache = {}
        
        print(f"[Daemon] Инициализация сервиса для версии {version}...", flush=True)
        if skip_model_loading:
            print("[Daemon] Пропуск загрузки моделей (skip_model_loading=True). Модели должны быть загружены в отдельном демоне.", flush=True)
        else:
            self._load_models()
        print("[Daemon] Модели перезагружены", flush=True)

    def _process_frames_pipeline(self, res_frame_queue, video_len, coord_list_cycle, 
                                  frame_list_cycle, mask_list_cycle, mask_coords_list_cycle,
                                  coord_placeholder, result_img_save_path):
        """
        Обрабатывает кадры из очереди параллельно с генерацией.
        Вызывается в отдельном потоке для пайплайнинга.
        """
        try:
            frames_processed = 0
            while frames_processed < video_len:
                try:
                    res_frame = res_frame_queue.get(timeout=30) # 30 sec timeout
                except queue.Empty:
                    print("[Daemon] Timeout waiting for frames in pipeline")
                    break
                
                # Blending logic (same as single thread)
                idx = frames_processed
                bbox = coord_list_cycle[idx % len(coord_list_cycle)]
                ori_frame = frame_list_cycle[idx % len(frame_list_cycle)].copy()
                mask = mask_list_cycle[idx % len(mask_list_cycle)]
                mask_crop_box = mask_coords_list_cycle[idx % len(mask_coords_list_cycle)]
                
                x1, y1, x2, y2 = bbox
                try:
                    res_frame = cv2.resize(res_frame.astype(np.uint8), (x2-x1, y2-y1))
                    combine_frame = get_image_blending(ori_frame, res_frame, bbox, mask, mask_crop_box)
                    cv2.imwrite(f"{result_img_save_path}/{str(frames_processed).zfill(8)}.png", combine_frame)
                    frames_processed += 1
                except Exception as e:
                    print(f"Error blending frame {idx}: {e}")
                    frames_processed += 1 # Skip but count to avoid stall
                    
        except Exception as e:
            print(f"[Daemon] Error in pipeline thread: {e}")

        print(f"[Daemon] Сервис готов к обработке запросов из: {request_dir}", flush=True)
    
    def _load_models(self):
        """Загружает все модели в память"""
        start_time = time.perf_counter()
        
        # Настройка ffmpeg
        ffmpeg_start = time.perf_counter()
        if not fast_check_ffmpeg():
            print("Добавление ffmpeg в PATH")
            path_separator = ';' if sys.platform == 'win32' else ':'
            os.environ["PATH"] = f"{self.ffmpeg_path}{path_separator}{os.environ['PATH']}"
            if not fast_check_ffmpeg():
                print("Предупреждение: Не удалось найти ffmpeg")
        ffmpeg_time = time.perf_counter() - ffmpeg_start
        print(f"[Daemon] Настройка ffmpeg: {format_time(ffmpeg_time)}")
        
        # Устройство
        device_start = time.perf_counter()
        self.device = torch.device(f"cuda:{self.gpu_id}" if torch.cuda.is_available() else "cpu")
        
        # Настройка CUDA allocator для уменьшения фрагментации памяти
        if torch.cuda.is_available():
            # Устанавливаем max_split_size_mb через переменную окружения, если не установлена
            if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
                os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
                print(f"[Daemon] Установлена настройка PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 для уменьшения фрагментации памяти")
            else:
                print(f"[Daemon] Используется существующая настройка PYTORCH_CUDA_ALLOC_CONF={os.environ['PYTORCH_CUDA_ALLOC_CONF']}")
        
        device_time = time.perf_counter() - device_start
        print(f"[Daemon] Используется устройство: {self.device} (время: {format_time(device_time)})")
        
        # Загрузка основных моделей
        print("[Daemon] Загрузка VAE, UNet и PositionalEncoding...")
        models_start = time.perf_counter()
        self.vae, self.unet, self.pe = load_all_model(
            unet_model_path=self.unet_model_path,
            vae_type=self.vae_type,
            unet_config=self.unet_config,
            device=self.device
        )
        models_time = time.perf_counter() - models_start
        print(f"[Daemon] Загрузка VAE, UNet и PositionalEncoding: {format_time(models_time)}", flush=True)
        
        # Установка типа данных
        dtype_start = time.perf_counter()
        if self.use_float16:
            print("[Daemon] Конвертация в float16...", flush=True)
            self.pe = self.pe.half()
            self.vae.vae = self.vae.vae.half()
            self.unet.model = self.unet.model.half()
            self.weight_dtype = torch.float16
        else:
            self.weight_dtype = torch.float32
        dtype_time = time.perf_counter() - dtype_start
        print(f"[Daemon] Конвертация типа данных: {format_time(dtype_time)}", flush=True)
        
        # Перемещение на устройство
        move_start = time.perf_counter()
        self.pe = self.pe.to(self.device)
        self.vae.vae = self.vae.vae.to(self.device)
        self.unet.model = self.unet.model.to(self.device)
        move_time = time.perf_counter() - move_start
        print(f"[Daemon] Перемещение моделей на устройство: {format_time(move_time)}", flush=True)
        
        # Загрузка Whisper
        print("[Daemon] Загрузка Whisper модели...", flush=True)
        whisper_start = time.perf_counter()
        self.audio_processor = AudioProcessor(feature_extractor_path=self.whisper_dir)
        self.whisper = WhisperModel.from_pretrained(self.whisper_dir)
        self.whisper = self.whisper.to(device=self.device, dtype=self.weight_dtype).eval()
        self.whisper.requires_grad_(False)
        whisper_time = time.perf_counter() - whisper_start
        print(f"[Daemon] Загрузка Whisper модели: {format_time(whisper_time)}", flush=True)
        
        # Загрузка FaceParser
        print("[Daemon] Загрузка FaceParser...", flush=True)
        fp_start = time.perf_counter()
        if self.version_arg == "v15":
            self.fp = FaceParsing(
                left_cheek_width=self.left_cheek_width,
                right_cheek_width=self.right_cheek_width
            )
        else:
            self.fp = FaceParsing()
        fp_time = time.perf_counter() - fp_start
        print(f"[Daemon] Загрузка FaceParser: {format_time(fp_time)}")
        
        # Timesteps
        self.timesteps = torch.tensor([0], device=self.device)
        
        total_time = time.perf_counter() - start_time
        print(f"[Daemon] Все модели успешно загружены! Общее время загрузки: {format_time(total_time)}", flush=True)
    
    def _preload_avatars(self):
        """Предзагружает все готовые аватары из results/{version}/avatars в память."""
        if self.version_arg == "v15":
            avatars_dir = os.path.join(self.result_dir, self.version_arg, "avatars")
        else:
            avatars_dir = os.path.join(self.result_dir, "avatars")
        
        if not os.path.exists(avatars_dir):
            print(f"[Daemon] Папка аватаров не найдена: {avatars_dir}", flush=True)
            return
        
        # Собираем все подпапки — это готовые аватары (пропускаем хэшированные версии типа couch_1_a3f8b2d1)
        import re
        hash_suffix_pattern = re.compile(r'_[0-9a-f]{8}$')
        avatar_dirs = [d for d in os.listdir(avatars_dir) 
                       if os.path.isdir(os.path.join(avatars_dir, d)) 
                       and not hash_suffix_pattern.search(d)]
        
        if not avatar_dirs:
            print(f"[Daemon] Нет готовых аватаров в {avatars_dir}", flush=True)
            return
        
        print(f"[Daemon] Предзагрузка {len(avatar_dirs)} аватаров из {avatars_dir}...", flush=True)
        preload_start = time.perf_counter()
        loaded_count = 0
        
        for avatar_name in avatar_dirs:
            try:
                avatar_start = time.perf_counter()
                
                # Ищем соответствующий видеофайл в data/video/
                video_path = None
                # Имя аватара может содержать хэш параметров (name_hash), берём базовое имя
                base_name = avatar_name.split('_')[0] if '_' in avatar_name else avatar_name
                # Но может быть и составное имя типа couch_1, поэтому пробуем точное совпадение сначала
                for ext in ('.mp4', '.avi', '.mov'):
                    candidate = os.path.join("data/video", f"{avatar_name}{ext}")
                    if os.path.exists(candidate):
                        video_path = candidate
                        break
                
                # Если не нашли по полному имени, ищем по базовому (без хэша параметров)
                if video_path is None and '_' in avatar_name:
                    # Пробуем убрать последний сегмент (хэш), если он выглядит как хэш (8 символов hex)
                    parts = avatar_name.rsplit('_', 1)
                    if len(parts) == 2 and len(parts[1]) == 8:
                        potential_base = parts[0]
                    else:
                        potential_base = avatar_name
                    
                    for ext in ('.mp4', '.avi', '.mov'):
                        candidate = os.path.join("data/video", f"{potential_base}{ext}")
                        if os.path.exists(candidate):
                            video_path = candidate
                            break
                
                if video_path is None:
                    print(f"[Daemon] ⚠️ Видеофайл для аватара '{avatar_name}' не найден, пропускаем", flush=True)
                    continue
                
                # Создаём AvatarDaemon с дефолтными параметрами
                avatar = AvatarDaemon(
                    task_id=avatar_name,
                    video_path=video_path,
                    result_dir=self.result_dir,
                    version=self.version_arg,
                    bbox_shift=0,
                    left_cheek_width=self.left_cheek_width,
                    right_cheek_width=self.right_cheek_width,
                    parsing_mode="jaw",
                    extra_margin=10,
                    daemon_service=self
                )
                
                # prepare() проверяет кэш на диске и вызывает _load_materials()
                avatar.prepare()
                
                if avatar.materials is not None:
                    self.avatar_cache[avatar_name] = avatar
                    loaded_count += 1
                    avatar_time = time.perf_counter() - avatar_start
                    print(f"[Daemon] ✓ Аватар '{avatar_name}' загружен в память ({format_time(avatar_time)})", flush=True)
                else:
                    print(f"[Daemon] ⚠️ Не удалось загрузить материалы для '{avatar_name}'", flush=True)
                    
            except Exception as e:
                print(f"[Daemon] ⚠️ Ошибка при загрузке аватара '{avatar_name}': {e}", flush=True)
                import traceback
                traceback.print_exc()
        
        preload_time = time.perf_counter() - preload_start
        print(f"[Daemon] Предзагрузка завершена: {loaded_count}/{len(avatar_dirs)} аватаров за {format_time(preload_time)}", flush=True)

    
    def _process_frames_pipeline(self, res_frame_queue, video_len, coord_list_cycle, 
                                  frame_list_cycle, mask_list_cycle, mask_coords_list_cycle,
                                  coord_placeholder, result_img_save_path):
        """
        Обрабатывает кадры из очереди параллельно с генерацией.
        Вызывается в отдельном потоке для пайплайнинга.
        """
        idx = 0
        empty_count = 0
        max_empty_retries = 10  # Максимальное количество пустых попыток перед завершением
        
        while idx < video_len:
            try:
                res_frame = res_frame_queue.get(block=True, timeout=1)
                empty_count = 0  # Сброс счетчика при успешном получении кадра
            except queue.Empty:
                empty_count += 1
                # Если очередь пуста слишком долго и мы обработали все ожидаемые кадры, завершаем
                if empty_count >= max_empty_retries:
                    break
                continue
            
            bbox = coord_list_cycle[idx % (len(coord_list_cycle))]
            if bbox == coord_placeholder:
                idx += 1
                continue
            
            # Используем .copy() вместо deepcopy для numpy массивов (быстрее)
            ori_frame = frame_list_cycle[idx % (len(frame_list_cycle))].copy()
            mask = mask_list_cycle[idx % (len(mask_list_cycle))]
            mask_crop_box = mask_coords_list_cycle[idx % (len(mask_coords_list_cycle))]
            
            x1, y1, x2, y2 = bbox
            try:
                res_frame = cv2.resize(res_frame.astype(np.uint8), (x2-x1, y2-y1))
            except Exception as e:
                print(f"[Daemon] Ошибка при изменении размера кадра {idx}: {e}")
                idx += 1
                continue
            
            try:
                combine_frame = get_image_blending(ori_frame, res_frame, bbox, mask, mask_crop_box)
                if combine_frame is None:
                    print(f"[Daemon] Предупреждение: combine_frame равен None для кадра {idx}")
                    idx += 1
                    continue
                if not isinstance(combine_frame, np.ndarray):
                    print(f"[Daemon] Предупреждение: combine_frame имеет неправильный тип {type(combine_frame)} для кадра {idx}")
                    idx += 1
                    continue
                cv2.imwrite(f"{result_img_save_path}/{str(idx).zfill(8)}.png", combine_frame)
            except Exception as e:
                print(f"[Daemon] Ошибка при сохранении кадра {idx}: {e}")
                import traceback
                traceback.print_exc()
            idx += 1
    

    
    def _process_frames_pipeline_stream(self, res_frame_queue, video_len, coord_list_cycle, 
                                        frame_list_cycle, mask_list_cycle, mask_coords_list_cycle,
                                        coord_placeholder, segment_callback, segment_duration_seconds=2.0, fps=25):
        """
        Обрабатывает кадры из очереди и создает видео сегменты для стриминга.
        Вызывается в отдельном потоке для пайплайнинга.
        
        Args:
            segment_callback: Функция callback(segment_data, segment_index, start_frame, end_frame) 
                             для отправки готовых сегментов
            segment_duration_seconds: Длительность одного сегмента в секундах
            fps: FPS видео
        """
        idx = 0
        empty_count = 0
        max_empty_retries = 10
        frames_per_segment = int(segment_duration_seconds * fps)
        current_segment_frames = []
        current_segment_index = 0
        segment_start_frame = 0
        
        while idx < video_len:
            try:
                res_frame = res_frame_queue.get(block=True, timeout=1)
                empty_count = 0
            except queue.Empty:
                empty_count += 1
                if empty_count >= max_empty_retries:
                    break
                continue
            
            bbox = coord_list_cycle[idx % (len(coord_list_cycle))]
            if bbox == coord_placeholder:
                idx += 1
                continue
            
            ori_frame = frame_list_cycle[idx % (len(frame_list_cycle))].copy()
            mask = mask_list_cycle[idx % (len(mask_list_cycle))]
            mask_crop_box = mask_coords_list_cycle[idx % (len(mask_coords_list_cycle))]
            
            x1, y1, x2, y2 = bbox
            try:
                res_frame = cv2.resize(res_frame.astype(np.uint8), (x2-x1, y2-y1))
            except:
                idx += 1
                continue
            
            combine_frame = get_image_blending(ori_frame, res_frame, bbox, mask, mask_crop_box)
            current_segment_frames.append(combine_frame)
            
            # Если накопили достаточно кадров для сегмента, создаем сегмент
            if len(current_segment_frames) >= frames_per_segment:
                segment_end_frame = idx
                try:
                    segment_callback(current_segment_frames, current_segment_index, 
                                   segment_start_frame, segment_end_frame)
                except Exception as e:
                    print(f"[Daemon] Ошибка при отправке сегмента {current_segment_index}: {e}")
                
                current_segment_frames = []
                current_segment_index += 1
                segment_start_frame = idx + 1
            
            idx += 1
        
        # Отправляем оставшиеся кадры как последний сегмент
        if len(current_segment_frames) > 0:
            try:
                segment_callback(current_segment_frames, current_segment_index, 
                               segment_start_frame, idx - 1)
            except Exception as e:
                print(f"[Daemon] Ошибка при отправке последнего сегмента: {e}")
    def _process_request_raw(self, task_id, video_path, audio_paths, output_vid_name, 
                           result_dir, fps=25, batch_size=8, 
                           audio_padding_length_left=2, audio_padding_length_right=2,
                           bbox_shift=0, extra_margin=10, parsing_mode="jaw",
                           left_cheek_width=90, right_cheek_width=90):
        """
        Обработка запроса "с нуля" (без использования кеша), позволяет менять параметры bbox и cheek_width.
        """
        print(f"[Daemon] Запуск RAW обработки для {task_id} с параметрами: bbox_shift={bbox_shift}, cheeks={left_cheek_width}/{right_cheek_width}", flush=True)
        
        # 1. Update FaceParsing mask parameters dynamically
        # We access the internal method of FaceParsing instance to update its mask
        if self.fp:
            try:
                # Re-create cheek mask with new parameters
                new_mask = self.fp._create_cheek_mask(left_cheek_width=left_cheek_width, right_cheek_width=right_cheek_width)
                self.fp.cheek_mask = new_mask
                print(f"[Daemon] Обновлена маска FaceParsing для {task_id}")
            except Exception as e:
                print(f"[Daemon] Ошибка обновления параметров FaceParsing: {e}")
        
        input_basename = os.path.basename(video_path).split('.')[0]
        

    def process_config(self, inference_config, source_id, is_file=False, restart_count=0, max_restarts=1, stop_event=None):
        """
        Обрабатывает конфигурацию инференса (словарь).
        Refactored to use AvatarDaemon for Smart Caching.
        """
        # Register stop event
        registered_tasks = []
        if stop_event:
            for task_id in inference_config:
                 if task_id not in ["audio_padding_length_left", "audio_padding_length_right", 
                                  "batch_size", "fps", "extra_margin", "parsing_mode",
                                  "use_saved_coord", "saved_coord", "result_dir", "webrtc_mode",
                                  "bbox_shift", "left_cheek_width", "right_cheek_width", "use_preprocessed"]:
                    with self.active_tasks_lock:
                        self.active_tasks[task_id] = stop_event
                        registered_tasks.append(task_id)

        with self.lock:
            try:
                request_start = time.perf_counter()
                print(f"[Daemon] Обработка конфигурации: {source_id}", flush=True)
                
                if stop_event and stop_event.is_set():
                        print(f"[Daemon] Заранее получен сигнал остановки для {source_id}, пропускаем", flush=True)
                        return

                # Global params
                audio_padding_length_left = inference_config.get("audio_padding_length_left", 2)
                audio_padding_length_right = inference_config.get("audio_padding_length_right", 2)
                batch_size = inference_config.get("batch_size", 8)
                fps = inference_config.get("fps", 25)
                extra_margin = inference_config.get("extra_margin", 10)
                parsing_mode = inference_config.get("parsing_mode", "jaw")
                result_dir = inference_config.get("result_dir", self.result_dir)
                webrtc_mode = inference_config.get("webrtc_mode", False)
                
                # Force bbox_shift to 0 for v15 to match realtime_inference.py and prepare_avatar.py
                if self.version_arg == "v15":
                     global_bbox_shift = 0
                else:
                     global_bbox_shift = inference_config.get("bbox_shift", 0)

                global_left_cheek = inference_config.get("left_cheek_width", self.left_cheek_width)
                global_right_cheek = inference_config.get("right_cheek_width", self.right_cheek_width)
                
                if webrtc_mode:
                    print(f"[Daemon] Обнаружен WebRTC режим: webrtc_mode={webrtc_mode} (mmap)", flush=True)
                
                for task_id in inference_config:
                    if task_id in ["audio_padding_length_left", "audio_padding_length_right", 
                                  "batch_size", "fps", "extra_margin", "parsing_mode",
                                  "use_saved_coord", "saved_coord", "result_dir", "webrtc_mode",
                                  "bbox_shift", "left_cheek_width", "right_cheek_width", "use_preprocessed"]:
                        continue
                    
                    try:
                        task_start = time.perf_counter()
                        print(f"[Daemon] Обработка задачи: {task_id}", flush=True)
                        
                        task_conf = inference_config[task_id]
                        video_path = task_conf["video_path"]
                        
                        # Determine efficient processing params
                        current_bbox_shift = task_conf.get("bbox_shift", global_bbox_shift)
                        current_left_cheek = task_conf.get("left_cheek_width", global_left_cheek)
                        current_right_cheek = task_conf.get("right_cheek_width", global_right_cheek)
                        current_parsing_mode = task_conf.get("parsing_mode", parsing_mode)
                        current_extra_margin = task_conf.get("extra_margin", extra_margin)
                        

                        # Use video basename as avatar_id for caching
                        base_avatar_id = os.path.splitext(os.path.basename(video_path))[0]
                        
                        # --- Avatar Versioning Logic ---
                        # Create hash of parameters
                        params_str = f"{current_bbox_shift}_{current_left_cheek}_{current_right_cheek}_{current_parsing_mode}_{current_extra_margin}_{self.version_arg}"
                        params_hash = hashlib.md5(params_str.encode('utf-8')).hexdigest()[:8]
                        
                        avatar_id = f"{base_avatar_id}_{params_hash}"
                        
                        # Copy existing cache if needed
                        if self.version_arg == "v15":
                             avatars_dir = f"{result_dir}/{self.version_arg}/avatars"
                        else:
                             avatars_dir = f"{result_dir}/avatars"

                        hashed_avatar_path = os.path.join(avatars_dir, avatar_id)
                        base_avatar_path = os.path.join(avatars_dir, base_avatar_id)
                        
                        # If hashed version doesn't exist but base exists (common case for first run with new logic or default params)
                        if not os.path.exists(hashed_avatar_path):
                            if os.path.exists(base_avatar_path):
                                print(f"[Daemon] Копирование базового аватара в хешированную версию: {base_avatar_id} -> {avatar_id}")
                                try:
                                    # Use copytree
                                    shutil.copytree(base_avatar_path, hashed_avatar_path)
                                except Exception as e:
                                     print(f"[Daemon] Ошибка копирования папки аватара: {e}")

                        
                        # Create AvatarDaemon instance
                        avatar = AvatarDaemon(
                            task_id=avatar_id,
                            video_path=video_path,
                            result_dir=result_dir,
                            version=self.version_arg,
                            bbox_shift=current_bbox_shift,
                            left_cheek_width=current_left_cheek,
                            right_cheek_width=current_right_cheek,
                            parsing_mode=current_parsing_mode,
                            extra_margin=current_extra_margin,
                            daemon_service=self
                        )
                        
                        # Smart Caching: Reuse memory if available
                        cache_hit = False
                        if task_id in self.avatar_cache:
                            old_avatar = self.avatar_cache[task_id]
                            # Check if old avatar was an AvatarDaemon instance (could be dict from legacy)
                            if isinstance(old_avatar, AvatarDaemon):
                                if old_avatar.params == avatar.params and old_avatar.materials is not None:
                                    print(f"[Daemon] Использование прогретого кеша памяти для {task_id}")
                                    avatar.materials = old_avatar.materials
                                    cache_hit = True
                        
                        # Update cache reference
                        self.avatar_cache[task_id] = avatar
                        
                        # Prepare (checks disk cache, regenerates if needed, loads to memory)
                        # Optimization: Skip disk check if we already have valid materials in memory
                        if cache_hit:
                             print(f"[Daemon] Пропуск проверки диска (memory cache hit) для {task_id}")
                        else:
                             avatar.prepare()
                        
                        # Prepare Audio Paths
                        if "audio_path" in task_conf:
                            audio_paths = [task_conf["audio_path"]]
                        elif "audio_clips" in task_conf:
                            audio_paths = list(task_conf["audio_clips"].values())
                        else:
                            audio_paths = [] # Should probably raise error if generating?
                            
                        # WebRTC File Mode logic
                        if webrtc_mode:
                             if not audio_paths:
                                 raise ValueError(f"Не найден audio_path в конфиге для {task_id} (WebRTC режим)")
                             # WebRTC only supports single audio path usually?
                             audio_path = audio_paths[0]
                             
                             print(f"[Daemon] WebRTC запуск генератора для {task_id}...")
                             # Get generator
                             gen, video_num = avatar.inference(audio_path, batch_size, fps, self.device)
                             
                             # We need mmap writer. And we need to consume the generator.
                             # Reusing logic part: writing to mmap.
                             # I'll implement the loop here.
                             
                             # Determine frame shape from materials
                             sample_frame = avatar.materials['frame_list_cycle'][0]
                             frame_shape = sample_frame.shape
                             
                             mmap_writer = None
                             try:
                                 mmap_writer = MmapFrameWriter(task_id, total_frames=video_num, frame_shape=frame_shape, frame_dtype=np.uint8)
                                 print(f"[Daemon] Mmap writer init: {frame_shape}, {video_num} frames")
                                 
                                 frames_generated = 0
                                 
                                 for i, (whisper_batch, latent_batch) in enumerate(tqdm(gen, total=int(np.ceil(float(video_num) / batch_size)))):
                                     # Check stop
                                     if stop_event and stop_event.is_set():
                                         mmap_writer.set_stopped(frames_generated)
                                         break

                                     audio_feature_batch = self.pe(whisper_batch.to(self.device))
                                     latent_batch = latent_batch.to(device=self.device, dtype=self.unet.model.dtype)
                                     pred_latents = self.unet.model(latent_batch, self.timesteps, encoder_hidden_states=audio_feature_batch).sample
                                     pred_latents = pred_latents.to(device=self.device, dtype=self.vae.vae.dtype)
                                     recon = self.vae.decode_latents(pred_latents)
                                     
                                     for res_frame in recon:
                                         # Need blending! 
                                         # AvatarDaemon stores cycle lists. We need to match index.
                                         idx = frames_generated
                                         bbox = avatar.materials['coord_list_cycle'][idx % len(avatar.materials['coord_list_cycle'])]
                                         ori_frame = avatar.materials['frame_list_cycle'][idx % len(avatar.materials['frame_list_cycle'])].copy()
                                         mask = avatar.materials['mask_list_cycle'][idx % len(avatar.materials['mask_list_cycle'])]
                                         mask_crop_box = avatar.materials['mask_coords_list_cycle'][idx % len(avatar.materials['mask_coords_list_cycle'])]
                                         
                                         x1, y1, x2, y2 = bbox
                                         try:
                                            res_frame = cv2.resize(res_frame.astype(np.uint8), (x2-x1, y2-y1))
                                            combine_frame = get_image_blending(ori_frame, res_frame, bbox, mask, mask_crop_box)
                                            mmap_writer.write_frame(idx, combine_frame)
                                            frames_generated += 1
                                         except Exception as e:
                                            print(f"Error blending frame {idx}: {e}")
                                            continue
                                     
                                     if i % 10 == 0: gc.collect()

                                 if frames_generated == video_num:
                                     mmap_writer.set_completed()
                                     print(f"[Daemon] WebRTC завершено: {frames_generated} кадров")
                                 else:
                                     if not (stop_event and stop_event.is_set()):
                                         mmap_writer.set_stopped(frames_generated)

                             except Exception as e:
                                 print(f"[Daemon] WebRTC Error: {e}")
                                 if mmap_writer: mmap_writer.set_error(str(e))
                                 else: create_error_mmap_file(task_id, str(e))
                                 import traceback
                                 traceback.print_exc()
                             finally:
                                 if mmap_writer: mmap_writer.close()
                             
                             continue # End WebRTC task

                        # Standard File Generation Mode
                        output_vid_name = task_conf.get("result_name", None)
                        input_basename = os.path.basename(video_path).split('.')[0]
                        
                        for audio_idx, audio_path in enumerate(audio_paths):
                            audio_basename = os.path.basename(audio_path).split('.')[0]
                            output_basename = f"{input_basename}_{audio_basename}"
                            
                            temp_dir = os.path.join(result_dir, f"{self.version_arg}")
                            os.makedirs(temp_dir, exist_ok=True)
                            
                            result_img_save_path = os.path.join(temp_dir, output_basename)
                            os.makedirs(result_img_save_path, exist_ok=True)
                            
                            if output_vid_name is None:
                                current_output_vid_name = os.path.join(temp_dir, output_basename + ".mp4")
                            else:
                                if len(audio_paths) > 1:
                                    base_name = os.path.splitext(output_vid_name)[0]
                                    ext = os.path.splitext(output_vid_name)[1]
                                    current_output_vid_name = os.path.join(temp_dir, f"{base_name}_{audio_idx}{ext}")
                                else:
                                    current_output_vid_name = os.path.join(temp_dir, output_vid_name)
                                    
                            print(f"[Daemon] Запуск генерации для {audio_basename}...")
                            
                            gen, video_num = avatar.inference(audio_path, batch_size, fps, self.device)
                            frames_generated = 0
                            
                            for i, (whisper_batch, latent_batch) in enumerate(tqdm(gen, total=int(np.ceil(float(video_num) / batch_size)))):
                                if stop_event and stop_event.is_set(): break
                                
                                audio_feature_batch = self.pe(whisper_batch.to(self.device))
                                latent_batch = latent_batch.to(device=self.device, dtype=self.unet.model.dtype)
                                pred_latents = self.unet.model(latent_batch, self.timesteps, encoder_hidden_states=audio_feature_batch).sample
                                pred_latents = pred_latents.to(device=self.device, dtype=self.vae.vae.dtype)
                                recon = self.vae.decode_latents(pred_latents)
                                
                                for res_frame in recon:
                                    idx = frames_generated
                                    bbox = avatar.materials['coord_list_cycle'][idx % len(avatar.materials['coord_list_cycle'])]
                                    ori_frame = avatar.materials['frame_list_cycle'][idx % len(avatar.materials['frame_list_cycle'])].copy()
                                    mask = avatar.materials['mask_list_cycle'][idx % len(avatar.materials['mask_list_cycle'])]
                                    mask_crop_box = avatar.materials['mask_coords_list_cycle'][idx % len(avatar.materials['mask_coords_list_cycle'])]
                                    
                                    x1, y1, x2, y2 = bbox
                                    try:
                                        res_frame = cv2.resize(res_frame.astype(np.uint8), (x2-x1, y2-y1))
                                        combine_frame = get_image_blending(ori_frame, res_frame, bbox, mask, mask_crop_box)
                                        cv2.imwrite(f"{result_img_save_path}/{str(frames_generated).zfill(8)}.png", combine_frame)
                                        frames_generated += 1
                                    except Exception as e:
                                        print(f"Error blending frame {idx}: {e}")

                            # FFMPEG
                            if not (stop_event and stop_event.is_set()):
                                temp_vid_path = f"{temp_dir}/temp_{input_basename}_{audio_basename}.mp4"
                                subprocess.run(shlex.split(f"ffmpeg -y -v warning -r {fps} -f image2 -i {result_img_save_path}/%08d.png -vcodec libx264 -vf format=yuv420p -crf 18 {temp_vid_path}"), check=True)
                                subprocess.run(shlex.split(f"ffmpeg -y -v warning -i {audio_path} -i {temp_vid_path} -c:v copy -c:a aac -shortest {current_output_vid_name}"), check=True)
                                
                                shutil.rmtree(result_img_save_path)
                                os.remove(temp_vid_path)
                                print(f"[Daemon] Результат сохранен: {current_output_vid_name}")
                        
                        task_time = time.perf_counter() - task_start
                        print(f"[Daemon] Задача {task_id} завершена: {format_time(task_time)}")

                    except Exception as e:
                        print(f"[Daemon] Ошибка при обработке задачи {task_id}: {e}")
                        import traceback
                        traceback.print_exc()

            except Exception as e:
                print(f"[Daemon] Критическая ошибка конфигурации: {e}")
                import traceback
                traceback.print_exc()
            finally:
                # Cleanup registered tasks
                for task_id in registered_tasks:
                     with self.active_tasks_lock:
                         if task_id in self.active_tasks:
                             del self.active_tasks[task_id]
                
                if is_file:
                    self._move_config_to_processed(source_id)
                
                request_time = time.perf_counter() - request_start
                print(f"[Daemon] Обработка конфигурации завершена: {format_time(request_time)}")
                

        # Очистка зарегистрированных задач
        if stop_event and registered_tasks:
                with self.active_tasks_lock:
                    for t_id in registered_tasks:
                        if t_id in self.active_tasks and self.active_tasks[t_id] == stop_event:
                             del self.active_tasks[t_id]

    def process_request(self, config_path, restart_count=0, max_restarts=1):
        """
        Обрабатывает запрос на инференс из файла (Legacy wrapper)
        """
        with self.lock:
            try:
                if not os.path.exists(config_path) or os.path.getsize(config_path) == 0:
                    print(f"[Daemon] ОШИБКА: Конфиг не существует или пустой: {config_path}")
                    return
                
                print(f"[Daemon] Загрузка конфига из файла: {config_path}")
                try:
                    inference_config = OmegaConf.load(config_path)
                    # Convert to dict for safety
                    inference_config = OmegaConf.to_container(inference_config, resolve=True)
                except Exception as e:
                    print(f"[Daemon] Ошибка при загрузке конфига: {e}")
                    import traceback
                    traceback.print_exc()
                    return
                
                self.process_config(inference_config, config_path, is_file=True, restart_count=restart_count, max_restarts=max_restarts)
            except Exception as e:
                print(f"[Daemon] Ошибка process_request: {e}")    
                import traceback
                traceback.print_exc()
    
    def process_request_stream(self, task_id, video_path, audio_path, 
                              segment_callback, version="v15",
                              audio_padding_length_left=2, audio_padding_length_right=2,
                              batch_size=8, fps=25, extra_margin=10, parsing_mode="jaw",
                              segment_duration_seconds=2.0):
        """
        Обрабатывает запрос на стриминг инференса.
        
        Args:
            task_id: ID задачи (для загрузки материалов аватара)
            video_path: Путь к видео файлу аватара
            audio_path: Путь к аудио файлу
            segment_callback: Функция callback(segment_data_base64, segment_index, start_time, end_time)
                            для отправки готовых сегментов. segment_data_base64 - base64 строка видео сегмента
            version: Версия модели ("v15" или "v1")
            segment_duration_seconds: Длительность одного сегмента в секундах
            Остальные параметры: как в process_request
        
        Returns:
            None (работает асинхронно через callback)
        """
        with self.lock:
            try:
                request_start = time.perf_counter()
                print(f"[Daemon] Начало стриминга для задачи: {task_id}")
                
                # Определение пути к подготовленному аватару
                if self.version_arg == "v15":
                    avatar_base_path = f"./results/{self.version_arg}/avatars/{task_id}"
                else:  # v1
                    avatar_base_path = f"./results/avatars/{task_id}"
                
                # Загрузка материалов аватара
                avatar_materials = self._load_avatar_materials(task_id, avatar_base_path)
                coord_list_cycle = avatar_materials['coord_list_cycle']
                frame_list_cycle = avatar_materials['frame_list_cycle']
                input_latent_list_cycle = avatar_materials['input_latent_list_cycle']
                mask_list_cycle = avatar_materials['mask_list_cycle']
                mask_coords_list_cycle = avatar_materials['mask_coords_list_cycle']
                
                # Определяем FPS для генерации видео
                # Используем переданный параметр fps (если указан), иначе берем из видео файла
                if fps and fps > 0:
                    video_fps = fps
                    print(f"[Daemon] Используется переданный FPS для генерации: {video_fps}")
                elif get_file_type(video_path) == "video":
                    video_fps = get_video_fps(video_path)
                    print(f"[Daemon] Используется FPS из видео файла: {video_fps}")
                else:
                    video_fps = 25  # Значение по умолчанию
                    print(f"[Daemon] Используется FPS по умолчанию: {video_fps}")
                
                # Извлечение аудио фич
                print(f"[Daemon] Извлечение аудио фич для {audio_path}...")
                whisper_input_features, librosa_length = self.audio_processor.get_audio_feature(
                    audio_path, weight_dtype=self.weight_dtype)
                whisper_chunks = self.audio_processor.get_whisper_chunk(
                    whisper_input_features,
                    self.device,
                    self.weight_dtype,
                    self.whisper,
                    librosa_length,
                    fps=video_fps,
                    audio_padding_length_left=audio_padding_length_left,
                    audio_padding_length_right=audio_padding_length_right,
                )
                
                video_num = len(whisper_chunks)
                print(f"[Daemon] Будет сгенерировано {video_num} кадров")
                
                # Создаем очередь для пайплайнинга
                res_frame_queue = queue.Queue()
                
                # Переменная для хранения init segment
                init_segment_sent = [False]
                
                # Функция для создания init segment из первого сегмента
                def create_init_segment_from_first_segment(first_segment_path, audio_start=0.0, audio_duration=0.1):
                    """Создает init segment из первого сегмента"""
                    if init_segment_sent[0]:
                        return None
                    
                    temp_init_dir = os.path.join(self.result_dir, "temp_segments", "init")
                    os.makedirs(temp_init_dir, exist_ok=True)
                    
                    # Создаем init segment - очень короткий фрагмент с полной структурой
                    temp_init_path = f"{temp_init_dir}/init.mp4"
                    cmd_init = (
                        f"ffmpeg -y -v warning -i {audio_path} -i {first_segment_path} "
                        f"-ss {audio_start:.3f} -t {audio_duration:.3f} "
                        f"-c:v libx264 -c:a aac -b:v 1M -b:a 128k "
                        f"-movflags frag_keyframe+empty_moov+default_base_moof "
                        f"-f mp4 -shortest {temp_init_path}"
                    )
                    
                    try:
                        result = subprocess.run(
                            shlex.split(cmd_init),
                            capture_output=True,
                            text=True,
                            check=True,
                            timeout=30
                        )
                        
                        if os.path.exists(temp_init_path):
                            with open(temp_init_path, 'rb') as f:
                                init_data = f.read()
                            
                            shutil.rmtree(temp_init_dir, ignore_errors=True)
                            return init_data
                    except subprocess.CalledProcessError as e:
                        print(f"[Daemon] Ошибка при создании init segment: {e}")
                        if hasattr(e, 'stderr') and e.stderr:
                            print(f"[Daemon] stderr: {e.stderr[:500]}")
                    except subprocess.TimeoutExpired as e:
                        print(f"[Daemon] Таймаут при создании init segment: {e}")
                    except Exception as e:
                        print(f"[Daemon] Неожиданная ошибка при создании init segment: {e}")
                        import traceback
                        traceback.print_exc()
                    
                    shutil.rmtree(temp_init_dir, ignore_errors=True)
                    return None
                
                # Функция для создания и отправки сегмента
                def create_and_send_segment(frames_list, segment_index, start_frame, end_frame):
                    """Создает видео сегмент с аудио и отправляет через callback"""
                    if len(frames_list) == 0:
                        return
                    
                    # Создаем временную директорию для сегмента
                    temp_segment_dir = os.path.join(self.result_dir, "temp_segments", f"segment_{segment_index}")
                    os.makedirs(temp_segment_dir, exist_ok=True)
                    
                    # Сохраняем кадры
                    for i, frame in enumerate(frames_list):
                        if frame is None:
                            print(f"[Daemon] Предупреждение: кадр {i} равен None, пропускаем")
                            continue
                        if not isinstance(frame, np.ndarray):
                            print(f"[Daemon] Предупреждение: кадр {i} имеет неправильный тип {type(frame)}, пропускаем")
                            continue
                        try:
                            cv2.imwrite(f"{temp_segment_dir}/{str(i).zfill(8)}.png", frame)
                        except Exception as e:
                            print(f"[Daemon] Ошибка при сохранении кадра {i}: {e}")
                            import traceback
                            traceback.print_exc()
                            continue
                    
                    # Создаем видео без аудио
                    temp_video_path = f"{temp_segment_dir}/video.mp4"
                    cmd_img2video = (
                        f"ffmpeg -y -v warning -r {video_fps} -f image2 "
                        f"-i {temp_segment_dir}/%08d.png -vcodec libx264 "
                        f"-vf format=yuv420p -crf 18 -pix_fmt yuv420p {temp_video_path}"
                    )
                    
                    try:
                        result = subprocess.run(
                            shlex.split(cmd_img2video),
                            capture_output=True,
                            text=True,
                            check=True,
                            timeout=60
                        )
                    except subprocess.CalledProcessError as e:
                        print(f"[Daemon] Ошибка при создании видео сегмента: {e}")
                        if hasattr(e, 'stderr') and e.stderr:
                            print(f"[Daemon] stderr: {e.stderr[:500]}")
                        shutil.rmtree(temp_segment_dir, ignore_errors=True)
                        return
                    except subprocess.TimeoutExpired as e:
                        print(f"[Daemon] Таймаут при создании видео сегмента: {e}")
                        shutil.rmtree(temp_segment_dir, ignore_errors=True)
                        return
                    except Exception as e:
                        print(f"[Daemon] Неожиданная ошибка при создании видео сегмента: {e}")
                        import traceback
                        traceback.print_exc()
                        shutil.rmtree(temp_segment_dir, ignore_errors=True)
                        return
                    
                    # Вычисляем временные метки для аудио
                    start_time = start_frame / video_fps
                    end_time = (end_frame + 1) / video_fps
                    duration = end_time - start_time
                    
                    # Для первого сегмента создаем init segment
                    if segment_index == 0 and not init_segment_sent[0]:
                        init_data = create_init_segment_from_first_segment(temp_video_path, start_time, min(0.1, duration))
                        if init_data:
                            import base64
                            init_base64 = base64.b64encode(init_data).decode('utf-8')
                            segment_callback(init_base64, -1, 0, 0, is_init=True)
                            init_segment_sent[0] = True
                            print(f"[Daemon] Init segment отправлен")
                    
                    # Создаем медиа сегмент (только moof+mdat, без moov)
                    temp_segment_path = f"{temp_segment_dir}/segment.mp4"
                    cmd_combine_audio = (
                        f"ffmpeg -y -v warning -i {audio_path} -i {temp_video_path} "
                        f"-ss {start_time:.3f} -t {duration:.3f} "
                        f"-c:v libx264 -c:a aac -b:v 1M -b:a 128k "
                        f"-movflags frag_keyframe+empty_moov+default_base_moof "
                        f"-f mp4 -shortest {temp_segment_path}"
                    )
                    
                    try:
                        result = subprocess.run(
                            shlex.split(cmd_combine_audio),
                            capture_output=True,
                            text=True,
                            check=True,
                            timeout=60
                        )
                        
                        # Читаем сегмент и отправляем через callback
                        if os.path.exists(temp_segment_path):
                            with open(temp_segment_path, 'rb') as f:
                                segment_data = f.read()
                            
                            import base64
                            segment_base64 = base64.b64encode(segment_data).decode('utf-8')
                            
                            # Отправляем медиа сегмент (не init)
                            segment_callback(segment_base64, segment_index, start_time, end_time, is_init=False)
                            
                            print(f"[Daemon] Сегмент {segment_index} отправлен ({start_time:.2f}s - {end_time:.2f}s)")
                        else:
                            print(f"[Daemon] Предупреждение: файл сегмента не найден: {temp_segment_path}")
                        
                    except subprocess.CalledProcessError as e:
                        print(f"[Daemon] Ошибка при создании сегмента {segment_index}: {e}")
                        if hasattr(e, 'stderr') and e.stderr:
                            print(f"[Daemon] stderr: {e.stderr[:500]}")
                    except subprocess.TimeoutExpired as e:
                        print(f"[Daemon] Таймаут при создании сегмента {segment_index}: {e}")
                    except Exception as e:
                        print(f"[Daemon] Неожиданная ошибка при создании сегмента {segment_index}: {e}")
                        import traceback
                        traceback.print_exc()
                    
                    # Очистка временных файлов
                    shutil.rmtree(temp_segment_dir, ignore_errors=True)
                
                # Запускаем поток обработки кадров для стриминга
                process_thread = threading.Thread(
                    target=self._process_frames_pipeline_stream,
                    args=(
                        res_frame_queue, video_num, coord_list_cycle,
                        frame_list_cycle, mask_list_cycle, mask_coords_list_cycle,
                        coord_placeholder, create_and_send_segment, 
                        segment_duration_seconds, video_fps
                    )
                )
                process_thread.start()
                
                # Генерация кадров и отправка в очередь
                gen = datagen(
                    whisper_chunks=whisper_chunks,
                    vae_encode_latents=input_latent_list_cycle,
                    batch_size=batch_size,
                    delay_frame=0,
                    device=self.device,
                )
                
                total = int(np.ceil(float(video_num) / batch_size))
                frames_generated = 0
                
                for i, (whisper_batch, latent_batch) in enumerate(tqdm(gen, total=total)):
                    audio_feature_batch = self.pe(whisper_batch.to(self.device))
                    latent_batch = latent_batch.to(device=self.device, dtype=self.unet.model.dtype)
                    
                    pred_latents = self.unet.model(latent_batch, self.timesteps, encoder_hidden_states=audio_feature_batch).sample
                    pred_latents = pred_latents.to(device=self.device, dtype=self.vae.vae.dtype)
                    recon = self.vae.decode_latents(pred_latents)
                    
                    for res_frame in recon:
                        res_frame_queue.put(res_frame)
                        frames_generated += 1
                
                print(f"[Daemon] Инференс завершен: сгенерировано {frames_generated} кадров")
                
                # Ждем завершения обработки всех кадров
                process_thread.join(timeout=600)
                if process_thread.is_alive():
                    print("[Daemon] Предупреждение: поток обработки кадров не завершился в течение таймаута")
                
                request_time = time.perf_counter() - request_start
                print(f"[Daemon] Стриминг завершен. Общее время: {format_time(request_time)}")
                
            except Exception as e:
                print(f"[Daemon] Ошибка при обработке стриминга: {e}")
                import traceback
                traceback.print_exc()
                raise
    
    def process_request_webrtc(self, task_id, video_path, audio_path, 
                               frame_callback, version="v15",
                               audio_padding_length_left=2, audio_padding_length_right=2,
                               batch_size=8, fps=25, extra_margin=10, parsing_mode="jaw",
                               video_track=None, stop_flag=None):
        """
        Обрабатывает запрос на WebRTC стриминг инференса.
        Отправляет кадры по одному через callback.
        
        Args:
            task_id: ID задачи
            video_path: Путь к видео файлу аватара
            audio_path: Путь к аудио файлу
            frame_callback: Функция callback(frame_array, frame_index) для отправки каждого кадра
            Остальные параметры: как в process_request_stream
        """
        # Проверяем, что модели загружены
        if self.skip_model_loading or self.whisper is None or self.unet is None or self.vae is None:
            raise RuntimeError(
                "Модели не загружены. process_request_webrtc требует прямого доступа к моделям. "
                "Если демон уже запущен отдельно, не используйте skip_model_loading=True для WebRTC API, "
                "или запустите API без отдельного демона."
            )
        
        with self.lock:
            try:
                request_start = time.perf_counter()
                print(f"[Daemon] Начало WebRTC стриминга для задачи: {task_id}")
                
                # Определение пути к подготовленному аватару
                if self.version_arg == "v15":
                    avatar_base_path = f"./results/{self.version_arg}/avatars/{task_id}"
                else:  # v1
                    avatar_base_path = f"./results/avatars/{task_id}"
                
                # Загрузка материалов аватара
                avatar_materials = self._load_avatar_materials(task_id, avatar_base_path)
                coord_list_cycle = avatar_materials['coord_list_cycle']
                frame_list_cycle = avatar_materials['frame_list_cycle']
                input_latent_list_cycle = avatar_materials['input_latent_list_cycle']
                mask_list_cycle = avatar_materials['mask_list_cycle']
                mask_coords_list_cycle = avatar_materials['mask_coords_list_cycle']
                
                # Определяем FPS для генерации видео
                # Используем переданный параметр fps (если указан), иначе берем из видео файла
                if fps and fps > 0:
                    video_fps = fps
                    print(f"[Daemon] Используется переданный FPS для генерации: {video_fps}")
                elif get_file_type(video_path) == "video":
                    video_fps = get_video_fps(video_path)
                    print(f"[Daemon] Используется FPS из видео файла: {video_fps}")
                else:
                    video_fps = 25  # Значение по умолчанию
                    print(f"[Daemon] Используется FPS по умолчанию: {video_fps}")
                
                # Извлечение аудио фич
                print(f"[Daemon] Извлечение аудио фич для {audio_path}...")
                whisper_input_features, librosa_length = self.audio_processor.get_audio_feature(
                    audio_path, weight_dtype=self.weight_dtype)
                whisper_chunks = self.audio_processor.get_whisper_chunk(
                    whisper_input_features,
                    self.device,
                    self.weight_dtype,
                    self.whisper,
                    librosa_length,
                    fps=video_fps,
                    audio_padding_length_left=audio_padding_length_left,
                    audio_padding_length_right=audio_padding_length_right,
                )
                
                video_num = len(whisper_chunks)
                print(f"[Daemon] Будет сгенерировано {video_num} кадров для WebRTC")
                
                # Создаем очередь для пайплайнинга
                res_frame_queue = queue.Queue()
                
                # Запускаем поток обработки кадров для WebRTC
                process_thread = threading.Thread(
                    target=self._process_frames_pipeline_webrtc,
                    args=(
                        res_frame_queue, video_num, coord_list_cycle,
                        frame_list_cycle, mask_list_cycle, mask_coords_list_cycle,
                        coord_placeholder, frame_callback, video_track
                    )
                )
                process_thread.start()
                
                # Генерация кадров и отправка в очередь
                gen = datagen(
                    whisper_chunks=whisper_chunks,
                    vae_encode_latents=input_latent_list_cycle,
                    batch_size=batch_size,
                    delay_frame=0,
                    device=self.device,
                )
                
                total = int(np.ceil(float(video_num) / batch_size))
                frames_generated = 0
                
                # Проверяем состояние перед началом генерации
                if video_track:
                    print(f"[Daemon] Состояние video_track перед генерацией: running={video_track.running}, queue_size={video_track.frame_queue.qsize()}")
                    if not video_track.running:
                        print(f"[Daemon] ВНИМАНИЕ: video_track.running = False перед началом генерации! Устанавливаем в True")
                        video_track.running = True
                
                for i, (whisper_batch, latent_batch) in enumerate(tqdm(gen, total=total)):
                    # Проверяем, не остановлен ли поток или не установлен ли флаг остановки
                    if stop_flag and stop_flag.is_set():
                        print(f"[Daemon] Получен сигнал остановки генерации на батче {i}/{total}")
                        break
                    if video_track and not video_track.running:
                        print(f"[Daemon] Поток остановлен, прерываем генерацию на батче {i}/{total}")
                        print(f"[Daemon] Состояние video_track: running={video_track.running}, queue_size={video_track.frame_queue.qsize()}")
                        break
                    try:
                        # Проверяем валидность входных данных
                        if whisper_batch is None or latent_batch is None:
                            print(f"[Daemon] Предупреждение: пропускаем батч {i} из-за None значений")
                            continue
                        
                        audio_feature_batch = self.pe(whisper_batch.to(self.device))
                        latent_batch = latent_batch.to(device=self.device, dtype=self.unet.model.dtype)
                        
                        pred_latents = self.unet.model(latent_batch, self.timesteps, encoder_hidden_states=audio_feature_batch).sample
                        pred_latents = pred_latents.to(device=self.device, dtype=self.vae.vae.dtype)
                        
                        # Освобождаем промежуточные тензоры
                        del audio_feature_batch, latent_batch
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        
                        # Декодируем латенты через VAE (используем меньший размер батча для декодирования, если батч большой)
                        vae_decode_batch_size = min(4, batch_size) if batch_size > 4 else None
                        recon = self.vae.decode_latents(pred_latents, max_batch_size=vae_decode_batch_size)
                        
                        # Освобождаем pred_latents после декодирования
                        del pred_latents
                        
                        # Проверяем результат декодирования
                        if recon is None:
                            print(f"[Daemon] Предупреждение: результат декодирования равен None для батча {i}")
                            continue
                        
                        for res_frame in recon:
                            # Проверяем флаг остановки и состояние потока перед добавлением кадра
                            if stop_flag and stop_flag.is_set():
                                print(f"[Daemon] Получен сигнал остановки генерации")
                                break
                            if video_track and not video_track.running:
                                print(f"[Daemon] Поток остановлен, прерываем генерацию")
                                break
                            res_frame_queue.put(res_frame)
                            frames_generated += 1
                        
                        # Проверяем флаг остановки и состояние потока после обработки батча
                        if stop_flag and stop_flag.is_set():
                            print(f"[Daemon] Получен сигнал остановки генерации")
                            break
                        if video_track and not video_track.running:
                            print(f"[Daemon] Поток остановлен, прерываем генерацию")
                            break
                        
                        # Освобождаем recon после обработки
                        del recon
                        
                        # Периодическая очистка CUDA кэша (каждый 5-й батч или последний)
                        if (i + 1) % 5 == 0 or i == total - 1:
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            gc.collect()
                            
                    except torch.cuda.OutOfMemoryError as e:
                        # Очищаем память и пробуем обработать меньшим подбатчем
                        print(f"[Daemon] CUDA OOM на батче {i}, попытка обработки меньшими подбатчами...")
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        gc.collect()
                        
                        # Разбиваем батч на меньшие части
                        sub_batch_size = max(1, batch_size // 2)
                        whisper_sub_batches = torch.split(whisper_batch, sub_batch_size, dim=0)
                        latent_sub_batches = torch.split(latent_batch, sub_batch_size, dim=0)
                        
                        for sub_whisper, sub_latent in zip(whisper_sub_batches, latent_sub_batches):
                            audio_feature_batch = self.pe(sub_whisper.to(self.device))
                            sub_latent = sub_latent.to(device=self.device, dtype=self.unet.model.dtype)
                            
                            pred_latents = self.unet.model(sub_latent, self.timesteps, encoder_hidden_states=audio_feature_batch).sample
                            pred_latents = pred_latents.to(device=self.device, dtype=self.vae.vae.dtype)
                            
                            del audio_feature_batch, sub_latent
                            if torch.cuda.is_available():
                                torch.cuda.synchronize()
                            
                            # Используем меньший размер батча для декодирования в подбатчах
                            recon = self.vae.decode_latents(pred_latents, max_batch_size=2)
                            del pred_latents
                            
                            for res_frame in recon:
                                # Проверяем состояние потока перед добавлением кадра
                                if video_track and not video_track.running:
                                    print(f"[Daemon] Поток остановлен, прерываем генерацию в подбатче")
                                    break
                                res_frame_queue.put(res_frame)
                                frames_generated += 1
                            
                            # Проверяем состояние потока после обработки подбатча
                            if video_track and not video_track.running:
                                print(f"[Daemon] Поток остановлен, прерываем генерацию")
                                break
                            
                            del recon
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            gc.collect()
                        
                        # Проверяем состояние потока после обработки всех подбатчей
                        if video_track and not video_track.running:
                            print(f"[Daemon] Поток остановлен, прерываем генерацию")
                            break
                        
                        del whisper_batch, latent_batch
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except RuntimeError as e:
                        # Обработка других RuntimeError (может включать CUDA ошибки)
                        error_msg = str(e)
                        print(f"[Daemon] RuntimeError на батче {i}: {error_msg}")
                        if "CUDA" in error_msg or "cuda" in error_msg:
                            print(f"[Daemon] Очистка CUDA кэша после ошибки...")
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                        gc.collect()
                        # Пропускаем проблемный батч и продолжаем
                        continue
                    except Exception as e:
                        # Обработка всех остальных исключений для предотвращения segfault
                        print(f"[Daemon] Неожиданная ошибка на батче {i}: {e}")
                        import traceback
                        traceback.print_exc()
                        # Очищаем память
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        gc.collect()
                        # Пропускаем проблемный батч и продолжаем
                        continue
                
                # Проверяем, была ли генерация прервана
                was_interrupted = video_track and not video_track.running
                if was_interrupted:
                    print(f"[Daemon] Инференс прерван (поток остановлен): сгенерировано {frames_generated} из {video_num} кадров")
                else:
                    print(f"[Daemon] Инференс завершен: сгенерировано {frames_generated} кадров")
                
                # Финальная очистка памяти после генерации всех кадров
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                
                # Ждем завершения обработки всех кадров
                process_thread.join(timeout=600)
                if process_thread.is_alive():
                    print("[Daemon] Предупреждение: поток обработки кадров не завершился в течение таймаута")
                
                request_time = time.perf_counter() - request_start
                print(f"[Daemon] WebRTC стриминг завершен. Общее время: {format_time(request_time)}")
                
            except Exception as e:
                print(f"[Daemon] Ошибка при обработке WebRTC стриминга: {e}")
                import traceback
                traceback.print_exc()
                # Очищаем память при ошибке
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                raise
    
    def process_request_webrtc_file(self, task_id, video_path, audio_path, 
                                    frames_dir, version="v15",
                                    audio_padding_length_left=2, audio_padding_length_right=2,
                                    batch_size=8, fps=25, extra_margin=10, parsing_mode="jaw",
                                    stop_flag=None):
        """
        Обрабатывает запрос на WebRTC стриминг инференса и сохраняет кадры в директорию.
        Используется когда API работает без загруженных моделей.
        
        Args:
            task_id: ID задачи
            video_path: Путь к видео файлу аватара
            audio_path: Путь к аудио файлу
            frames_dir: Директория для сохранения кадров
            Остальные параметры: как в process_request_webrtc
        """
        # Проверяем, что модели загружены
        # Проверяем, что модели загружены
        if self.skip_model_loading or self.whisper is None or self.unet is None or self.vae is None:
            debug_info = f"skip_model_loading={self.skip_model_loading}, whisper={type(self.whisper)}, unet={type(self.unet)}, vae={type(self.vae)}"
            print(f"[Daemon ERROR] Модели не загружены при попытке process_request_webrtc_file. {debug_info}", flush=True)
            raise RuntimeError(
                f"Модели не загружены. process_request_webrtc_file требует загруженных моделей в демоне. State: {debug_info}"
            )
        
        # НЕ используем self.lock здесь, так как метод вызывается из process_request, который уже держит lock
        try:
            request_start = time.perf_counter()
            print(f"[Daemon] Начало WebRTC стриминга (файловый режим) для задачи: {task_id}", flush=True)
            
            # Определение пути к подготовленному аватару
            if self.version_arg == "v15":
                avatar_base_path = f"./results/{self.version_arg}/avatars/{task_id}"
            else:  # v1
                avatar_base_path = f"./results/avatars/{task_id}"
            
            # Загрузка материалов аватара
            avatar_materials = self._load_avatar_materials(task_id, avatar_base_path)
            coord_list_cycle = avatar_materials['coord_list_cycle']
            frame_list_cycle = avatar_materials['frame_list_cycle']
            input_latent_list_cycle = avatar_materials['input_latent_list_cycle']
            mask_list_cycle = avatar_materials['mask_list_cycle']
            mask_coords_list_cycle = avatar_materials['mask_coords_list_cycle']
            
            # Определяем FPS для генерации видео
            if fps and fps > 0:
                video_fps = fps
                print(f"[Daemon] Используется переданный FPS для генерации: {video_fps}", flush=True)
            elif get_file_type(video_path) == "video":
                video_fps = get_video_fps(video_path)
                print(f"[Daemon] Используется FPS из видео файла: {video_fps}", flush=True)
            else:
                video_fps = 25
                print(f"[Daemon] Используется FPS по умолчанию: {video_fps}", flush=True)
            
            # Извлечение аудио фич
            print(f"[Daemon] Извлечение аудио фич для {audio_path}...", flush=True)
            whisper_input_features, librosa_length = self.audio_processor.get_audio_feature(
                audio_path, weight_dtype=self.weight_dtype)
            whisper_chunks = self.audio_processor.get_whisper_chunk(
                whisper_input_features,
                self.device,
                self.weight_dtype,
                self.whisper,
                librosa_length,
                fps=video_fps,
                audio_padding_length_left=audio_padding_length_left,
                audio_padding_length_right=audio_padding_length_right,
            )
            
            video_num = len(whisper_chunks)
            print(f"[Daemon] Будет сгенерировано {video_num} кадров для WebRTC (файловый режим)", flush=True)
            
            # Проверяем, что frame_list_cycle не пуст перед запуском потока
            if len(frame_list_cycle) == 0:
                error_msg = f"[Daemon] КРИТИЧЕСКАЯ ОШИБКА: frame_list_cycle пуст для task_id={task_id}. Аватар не был подготовлен или загружен неправильно. Путь к аватару: {avatar_base_path}"
                print(error_msg, flush=True)
                # Создаем mmap файл с ошибкой, чтобы API мог обнаружить проблему
                try:
                    create_error_mmap_file(task_id, error_msg)
                except Exception as e:
                    print(f"[Daemon] Не удалось создать mmap файл для индикации ошибки: {e}", flush=True)
                raise RuntimeError(error_msg)
            
            # Проверяем, что video_num > 0
            if video_num == 0:
                error_msg = f"[Daemon] КРИТИЧЕСКАЯ ОШИБКА: video_num равен 0 для task_id={task_id}. Нет кадров для генерации (whisper_chunks пуст)."
                print(error_msg, flush=True)
                # Создаем mmap файл с ошибкой, чтобы API мог обнаружить проблему
                try:
                    create_error_mmap_file(task_id, error_msg)
                except Exception as e:
                    print(f"[Daemon] Не удалось создать mmap файл для индикации ошибки: {e}", flush=True)
                raise RuntimeError(error_msg)
            
            print(f"[Daemon] Загружено {len(frame_list_cycle)} кадров аватара для task_id={task_id}", flush=True)
            
            # Создаем очередь для пайплайнинга
            res_frame_queue = queue.Queue()
            
            # Запускаем поток обработки кадров для WebRTC (mmap режим)
            from musetalk.utils.preprocessing import coord_placeholder
            from scripts.mmap_frame_buffer import get_mmap_path
            mmap_file_path = get_mmap_path(task_id)
            
            print(f"[Daemon] Запуск потока обработки кадров для task_id={task_id}", flush=True)
            print(f"[Daemon] Ожидаемый путь к mmap файлу: {mmap_file_path}", flush=True)
            print(f"[Daemon] Параметры: video_num={video_num}, frame_list_cycle_len={len(frame_list_cycle)}, coord_list_cycle_len={len(coord_list_cycle)}", flush=True)
            
            process_thread = threading.Thread(
                target=self._process_frames_pipeline_webrtc_file,
                args=(
                    res_frame_queue, video_num, coord_list_cycle,
                    frame_list_cycle, mask_list_cycle, mask_coords_list_cycle,
                    coord_placeholder, task_id, stop_flag
                ),
                name=f"WebRTCFrameProcessor-{task_id}"
            )
            process_thread.start()
            print(f"[Daemon] Поток обработки кадров запущен для task_id={task_id}", flush=True)
            
            # Даем потоку время на инициализацию mmap файла
            mmap_initialized = False
            for i in range(10):  # Ждем до 5 секунд (10 * 0.5)
                time.sleep(0.5)
                if os.path.exists(mmap_file_path):
                    file_size = os.path.getsize(mmap_file_path)
                    if file_size > 0:
                        print(f"[Daemon] Mmap файл успешно создан потоком: {mmap_file_path} (размер: {file_size} байт)", flush=True)
                        mmap_initialized = True
                        break
                if not process_thread.is_alive():
                    print(f"[Daemon] ПРЕДУПРЕЖДЕНИЕ: Поток обработки кадров завершился до создания mmap файла!", flush=True)
                    # Создаем mmap файл с ошибкой, чтобы API мог обнаружить проблему
                    try:
                        create_error_mmap_file(task_id, "Поток обработки кадров завершился до создания mmap файла")
                    except Exception as e:
                        print(f"[Daemon] Не удалось создать mmap файл для индикации ошибки: {e}", flush=True)
                    break
            else:
                if not os.path.exists(mmap_file_path):
                    print(f"[Daemon] ПРЕДУПРЕЖДЕНИЕ: Mmap файл не был создан в течение 5 секунд после запуска потока", flush=True)
                    print(f"[Daemon] Поток все еще работает: {process_thread.is_alive()}", flush=True)
                    # Если поток еще работает, но файл не создан, возможно, произошла ошибка
                    # Создаем mmap файл с ошибкой на всякий случай
                    if not process_thread.is_alive():
                        try:
                            create_error_mmap_file(task_id, "Поток обработки кадров завершился до создания mmap файла")
                        except Exception as e:
                            print(f"[Daemon] Не удалось создать mmap файл для индикации ошибки: {e}", flush=True)
            
            # Генерация кадров и отправка в очередь
            gen = datagen(
                whisper_chunks=whisper_chunks,
                vae_encode_latents=input_latent_list_cycle,
                batch_size=batch_size,
                delay_frame=0,
                device=self.device,
            )
            
            total = int(np.ceil(float(video_num) / batch_size))
            frames_generated = 0
            
            for i, (whisper_batch, latent_batch) in enumerate(tqdm(gen, total=total)):
                # Проверяем флаг остановки
                if stop_flag and stop_flag.is_set():
                    print(f"[Daemon] Получен сигнал остановки генерации на батче {i}/{total}", flush=True)
                    break
                
                try:
                    if whisper_batch is None or latent_batch is None:
                        print(f"[Daemon] Предупреждение: пропускаем батч {i} из-за None значений", flush=True)
                        continue
                    
                    audio_feature_batch = self.pe(whisper_batch.to(self.device))
                    latent_batch = latent_batch.to(device=self.device, dtype=self.unet.model.dtype)
                    
                    pred_latents = self.unet.model(latent_batch, self.timesteps, encoder_hidden_states=audio_feature_batch).sample
                    pred_latents = pred_latents.to(device=self.device, dtype=self.vae.vae.dtype)
                    
                    del audio_feature_batch, latent_batch
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    
                    vae_decode_batch_size = min(4, batch_size) if batch_size > 4 else None
                    recon = self.vae.decode_latents(pred_latents, max_batch_size=vae_decode_batch_size)
                    
                    del pred_latents
                    
                    if recon is None:
                        print(f"[Daemon] Предупреждение: результат декодирования равен None для батча {i}", flush=True)
                        continue
                    
                    for res_frame in recon:
                        if stop_flag and stop_flag.is_set():
                            break
                        res_frame_queue.put(res_frame)
                        frames_generated += 1
                    
                    if stop_flag and stop_flag.is_set():
                        break
                    
                    del recon
                    
                    if (i + 1) % 5 == 0 or i == total - 1:
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        gc.collect()
                        
                except Exception as e:
                    print(f"[Daemon] Ошибка на батче {i}: {e}", flush=True)
                    import traceback
                    traceback.print_exc()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()
                    continue
            
            print(f"[Daemon] Инференс завершен: сгенерировано {frames_generated} кадров", flush=True)
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
            # Ждем завершения обработки всех кадров
            process_thread.join(timeout=600)
            if process_thread.is_alive():
                print("[Daemon] Предупреждение: поток обработки кадров не завершился в течение таймаута", flush=True)
            
            request_time = time.perf_counter() - request_start
            print(f"[Daemon] WebRTC стриминг (файловый режим) завершен. Общее время: {format_time(request_time)}", flush=True)
            
        except Exception as e:
            print(f"[Daemon] Ошибка при обработке WebRTC стриминга (файловый режим): {e}", flush=True)
            import traceback
            traceback.print_exc()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            raise
    
    def _move_config_to_processed(self, config_path):
        """Перемещает конфиг в processed директорию"""
        try:
            processed_dir = os.path.join(self.request_dir, "processed")
            os.makedirs(processed_dir, exist_ok=True)
            processed_path = os.path.join(processed_dir, os.path.basename(config_path))
            if os.path.exists(config_path):
                shutil.move(config_path, processed_path)
                print(f"[Daemon] Конфиг перемещен в processed: {processed_path}", flush=True)
            # Удаляем из обработанных
            if config_path in self.processed_files:
                self.processed_files.remove(config_path)
        except Exception as e:
            print(f"[Daemon] ОШИБКА при перемещении конфига в processed: {e}", flush=True)
            import traceback
            traceback.print_exc()
    
    def _check_for_new_configs(self):
        """Проверяет наличие новых конфигов в директории запросов"""
        processed_dir = os.path.join(self.request_dir, "processed")
        
        # Проверяем .yaml файлы (только в корне request_dir, не в поддиректориях)
        config_files = list(Path(self.request_dir).glob("*.yaml"))
        if config_files:
            print(f"[Daemon] Найдено {len(config_files)} .yaml файлов в {self.request_dir}")
        
        for config_file in config_files:
            config_file_str = str(config_file)
            # Пропускаем файлы из processed директории
            if "processed" in config_file_str:
                print(f"[Daemon] Пропущен конфиг из processed директории: {config_file_str}")
                continue
            
            if config_file_str not in self.processed_files:
                print(f"[Daemon] Проверка нового конфига: {config_file_str} (еще не в processed_files)")
                # Проверяем, что файл полностью записан (размер не меняется)
                time.sleep(0.5)
                if config_file_str not in self.processed_files and os.path.exists(config_file):
                    file_size = os.path.getsize(config_file)
                    print(f"[Daemon] Размер файла {config_file_str}: {file_size} байт")
                    if file_size > 0:  # Проверяем, что файл не пустой
                        self.processed_files.add(config_file_str)
                        print(f"[Daemon] Обнаружен новый конфиг: {config_file_str} (размер: {file_size} байт)", flush=True)
                        threading.Thread(target=self.process_request, args=(config_file_str,), daemon=True).start()
                    else:
                        print(f"[Daemon] Пропущен пустой конфиг: {config_file_str} (размер: {file_size} байт)", flush=True)
                else:
                    if config_file_str in self.processed_files:
                        print(f"[Daemon] Конфиг уже в processed_files: {config_file_str}")
                    elif not os.path.exists(config_file):
                        print(f"[Daemon] Конфиг не существует: {config_file_str}")
            else:
                print(f"[Daemon] Конфиг уже обработан: {config_file_str} (в processed_files)")
        
        # Проверяем .yml файлы
        for config_file in Path(self.request_dir).glob("*.yml"):
            # Пропускаем файлы из processed директории
            if "processed" in str(config_file):
                continue
            if str(config_file) not in self.processed_files:
                time.sleep(0.5)
                if str(config_file) not in self.processed_files and os.path.exists(config_file):
                    self.processed_files.add(str(config_file))
                    print(f"[Daemon] Обнаружен новый конфиг: {config_file}")
                    threading.Thread(target=self.process_request, args=(str(config_file),), daemon=True).start()
    
    def run(self):
        """Запускает демон-сервис"""
        # Load models if not skipped
        if not self.skip_model_loading:
            self._load_models()
        else:
            print("[Daemon] Пропуск загрузки моделей (skip_model_loading=True)", flush=True)

        # Предзагрузка всех готовых аватаров в память (опционально, управляется через .env)
        if os.getenv("PRELOAD_AVATARS", "true").lower() == "true":
            self._preload_avatars()
        else:
            print("[Daemon] Предзагрузка аватаров отключена (PRELOAD_AVATARS=false)", flush=True)

        print(f"[Daemon] Сервис запущен.", flush=True)
        if self.redis_client:
            print(f"[Daemon] Режим работы: Redis очередь (musetalk:queue) + Файлы", flush=True)
        else:
             print(f"[Daemon] Режим работы: Только Файлы ({self.request_dir})", flush=True)

        print("[Daemon] Для остановки нажмите Ctrl+C", flush=True)
        
        # Обрабатываем существующие файлы
        self._check_for_new_configs()
        
        last_redis_reconnect_attempt = 0
        redis_reconnect_interval = 5  # секунд

        try:
            while True:
                # 0. Попытка переподключения к Redis, если не подключено
                if self.redis_client is None:
                    current_time = time.time()
                    if current_time - last_redis_reconnect_attempt > redis_reconnect_interval:
                        last_redis_reconnect_attempt = current_time
                        self.redis_client = self._connect_to_redis()

                # 1. Проверяем Redis (приоритет)
                if self.redis_client:
                    try:
                        # Используем таймаут 1 секунду, чтобы успевать проверять файлы (если нужно)
                        task = self.redis_client.blpop("musetalk:queue", timeout=1)
                        if task:
                            # task is tuple (queue_name, data)
                            queue_name, data_str = task
                            print(f"[Daemon] Получена задача из Redis!", flush=True)
                            try:
                                config_data = json.loads(data_str)
                                if "command" in config_data and config_data["command"] == "stop":
                                    stop_task_id = config_data.get("task_id")
                                    if stop_task_id:
                                        print(f"[Daemon] Получена команда STOP для {stop_task_id}", flush=True)
                                        with self.active_tasks_lock:
                                            if stop_task_id in self.active_tasks:
                                                self.active_tasks[stop_task_id].set()
                                                print(f"[Daemon] Сигнал остановки отправлен задаче {stop_task_id}", flush=True)
                                            else:
                                                print(f"[Daemon] Задача {stop_task_id} не найдена в активных", flush=True)
                                    continue
                                
                                # Генерируем ID для логов
                                task_id = "unknown"
                                # Пытаемся найти task_id в ключах (обычно там один ключ с task_id)
                                for k in config_data.keys():
                                    if k not in ["webrtc_mode", "batch_size", "fps"]:
                                        task_id = k
                                        break
                                
                                # Создаем event для остановки
                                stop_event = threading.Event()
                                
                                # Запускаем обработку в отдельном потоке (как и было для файлов)
                                # Но process_config блокирует через lock.
                                # В оригинале process_request запускался в Thread.
                                threading.Thread(target=self.process_config, args=(config_data, f"redis_{task_id}", False, 0, 1, stop_event), daemon=True).start()
                                
                            except json.JSONDecodeError:
                                print(f"[Daemon] Ошибка декодирования JSON из Redis: {data_str}")
                            except Exception as e:
                                print(f"[Daemon] Ошибка обработки задачи из Redis: {e}")
                    except redis.RedisError as e:
                        print(f"[Daemon] Ошибка Redis: {e}. Переключение в режим переподключения.")
                        self.redis_client = None
                        last_redis_reconnect_attempt = time.time() # Сразу не долбить
                
                # 2. Проверяем файлы (как резерв)
                self._check_for_new_configs()
                
                if not self.redis_client:
                    time.sleep(1) # Если без редиса, спим
                    
        except KeyboardInterrupt:
            print("\n[Daemon] Остановка сервиса...")
        print("[Daemon] Сервис остановлен")


def main():
    parser = argparse.ArgumentParser(description="Демон-сервис предзагрузки моделей MuseTalk")
    parser.add_argument("--version", type=str, default="v1.5", choices=["v1.0", "v1.5", "v1", "v15"],
                       help="Версия модели")
    parser.add_argument("--gpu_id", type=int, default=0, help="ID GPU")
    parser.add_argument("--use_float16", action="store_true", help="Использовать float16")
    parser.add_argument("--whisper_dir", type=str, default="./models/whisper", help="Директория Whisper")
    parser.add_argument("--vae_type", type=str, default="sd-vae", help="Тип VAE")
    parser.add_argument("--ffmpeg_path", type=str, default="./ffmpeg-4.4-amd64-static/", help="Путь к ffmpeg")
    parser.add_argument("--left_cheek_width", type=int, default=90, help="Ширина левой щеки")
    parser.add_argument("--right_cheek_width", type=int, default=90, help="Ширина правой щеки")
    parser.add_argument("--request_dir", type=str, default="./requests", help="Директория для запросов")
    parser.add_argument("--result_dir", type=str, default="./results", help="Директория для результатов")
    parser.add_argument("--redis_host", type=str, default=os.getenv("REDIS_HOST", "localhost"), help="Redis host")
    parser.add_argument("--redis_port", type=int, default=int(os.getenv("REDIS_PORT", 6379)), help="Redis port")
    parser.add_argument("--redis_password", type=str, default=os.getenv("REDIS_PASSWORD", None), help="Redis password")
    
    args = parser.parse_args()
    
    try:
        print(f"[Daemon] Запуск демон-сервиса (версия: {args.version}, GPU: {args.gpu_id})", flush=True)
        
        # Создание сервиса
        service = ModelDaemonService(
            version=args.version,
            gpu_id=args.gpu_id,
            use_float16=args.use_float16,
            whisper_dir=args.whisper_dir,
            vae_type=args.vae_type,
            ffmpeg_path=args.ffmpeg_path,
            left_cheek_width=args.left_cheek_width,
            right_cheek_width=args.right_cheek_width,
            request_dir=args.request_dir,
            result_dir=args.result_dir,
            redis_host=args.redis_host,
            redis_port=args.redis_port,
            redis_password=args.redis_password
        )
        
        # Запуск сервиса
        print("[Daemon] Запуск основного цикла сервиса...", flush=True)
        service.run()
    except Exception as e:
        print(f"[Daemon] КРИТИЧЕСКАЯ ОШИБКА при запуске сервиса: {e}", flush=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

