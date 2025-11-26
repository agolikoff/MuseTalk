#!/usr/bin/env python3
"""
Демон-сервис для предзагрузки моделей MuseTalk.
Модели загружаются один раз при старте и остаются в памяти.
Сервис следит за директорией запросов и обрабатывает их по мере поступления.
"""
import os
import sys
import json
import time
import argparse
import subprocess
import threading
import queue
import shutil
import shlex
from pathlib import Path
from omegaconf import OmegaConf
import torch
from transformers import WhisperModel
# Используем простой polling вместо watchdog для совместимости

from musetalk.utils.blending import get_image_blending
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
                 request_dir="./requests", result_dir="./results"):
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
        self.lock = threading.Lock()
        
        # Отслеживание обработанных файлов
        self.processed_files = set()
        
        # Кеш для материалов аватаров (загруженных в память)
        # Ключ: task_id, значение: словарь с материалами
        self.avatar_cache = {}
        
        print(f"[Daemon] Инициализация сервиса для версии {version}...")
        self._load_models()
        print("[Daemon] Все модели загружены в память!")
        print(f"[Daemon] Сервис готов к обработке запросов из: {request_dir}")
    
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
        print(f"[Daemon] Загрузка VAE, UNet и PositionalEncoding: {format_time(models_time)}")
        
        # Установка типа данных
        dtype_start = time.perf_counter()
        if self.use_float16:
            print("[Daemon] Конвертация в float16...")
            self.pe = self.pe.half()
            self.vae.vae = self.vae.vae.half()
            self.unet.model = self.unet.model.half()
            self.weight_dtype = torch.float16
        else:
            self.weight_dtype = torch.float32
        dtype_time = time.perf_counter() - dtype_start
        print(f"[Daemon] Конвертация типа данных: {format_time(dtype_time)}")
        
        # Перемещение на устройство
        move_start = time.perf_counter()
        self.pe = self.pe.to(self.device)
        self.vae.vae = self.vae.vae.to(self.device)
        self.unet.model = self.unet.model.to(self.device)
        move_time = time.perf_counter() - move_start
        print(f"[Daemon] Перемещение моделей на устройство: {format_time(move_time)}")
        
        # Загрузка Whisper
        print("[Daemon] Загрузка Whisper модели...")
        whisper_start = time.perf_counter()
        self.audio_processor = AudioProcessor(feature_extractor_path=self.whisper_dir)
        self.whisper = WhisperModel.from_pretrained(self.whisper_dir)
        self.whisper = self.whisper.to(device=self.device, dtype=self.weight_dtype).eval()
        self.whisper.requires_grad_(False)
        whisper_time = time.perf_counter() - whisper_start
        print(f"[Daemon] Загрузка Whisper модели: {format_time(whisper_time)}")
        
        # Загрузка FaceParser
        print("[Daemon] Загрузка FaceParser...")
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
        print(f"[Daemon] Все модели успешно загружены! Общее время загрузки: {format_time(total_time)}")
    
    def _load_avatar_materials(self, task_id, avatar_base_path):
        """
        Загружает материалы аватара из файлов или возвращает из кеша.
        Материалы загружаются один раз и остаются в памяти для последующих использований.
        
        Args:
            task_id: ID задачи (используется как ключ кеша)
            avatar_base_path: Базовый путь к директории аватара
            
        Returns:
            Словарь с материалами аватара:
            - coord_list_cycle: координаты
            - frame_list_cycle: кадры
            - input_latent_list_cycle: latents
            - mask_list_cycle: маски
            - mask_coords_list_cycle: координаты масок
        """
        # Проверяем кеш
        if task_id in self.avatar_cache:
            print(f"[Daemon] Использование материалов аватара {task_id} из кеша памяти")
            return self.avatar_cache[task_id]
        
        # Загружаем материалы в первый раз
        print(f"[Daemon] Загрузка материалов аватара {task_id} в память...")
        load_start = time.perf_counter()
        
        full_imgs_path = f"{avatar_base_path}/full_imgs"
        coords_path = f"{avatar_base_path}/coords.pkl"
        latents_out_path = f"{avatar_base_path}/latents.pt"
        mask_out_path = f"{avatar_base_path}/mask"
        mask_coords_path = f"{avatar_base_path}/mask_coords.pkl"
        
        # Проверка существования файлов
        if not os.path.exists(mask_out_path) or not os.path.exists(mask_coords_path):
            raise FileNotFoundError(f"Предварительно созданные маски не найдены для {task_id}. "
                                   f"Используйте prepare_avatar.py для подготовки аватара.")
        if not os.path.exists(coords_path):
            raise FileNotFoundError(f"Координаты не найдены для {task_id}. "
                                   f"Используйте prepare_avatar.py для подготовки аватара.")
        if not os.path.exists(latents_out_path):
            raise FileNotFoundError(f"Latents не найдены для {task_id}. "
                                   f"Используйте prepare_avatar.py для подготовки аватара.")
        
        # Загрузка координат
        coords_load_start = time.perf_counter()
        with open(coords_path, 'rb') as f:
            coord_list_cycle = pickle.load(f)
        coords_load_time = time.perf_counter() - coords_load_start
        print(f"[Daemon] Загрузка координат: {format_time(coords_load_time)}")
        
        # Загрузка кадров
        frames_load_start = time.perf_counter()
        input_img_list = glob.glob(os.path.join(full_imgs_path, '*.[jpJP][pnPN]*[gG]'))
        input_img_list = sorted(input_img_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
        frame_list_cycle = read_imgs(input_img_list)
        frames_load_time = time.perf_counter() - frames_load_start
        print(f"[Daemon] Загрузка кадров: {format_time(frames_load_time)} (кадров: {len(frame_list_cycle)})")
        
        # Загрузка latents
        latents_load_start = time.perf_counter()
        input_latent_list_cycle = torch.load(latents_out_path)
        latents_load_time = time.perf_counter() - latents_load_start
        print(f"[Daemon] Загрузка latents: {format_time(latents_load_time)}")
        
        # Загрузка масок
        mask_load_start = time.perf_counter()
        with open(mask_coords_path, 'rb') as f:
            mask_coords_list_cycle = pickle.load(f)
        input_mask_list = glob.glob(os.path.join(mask_out_path, '*.[jpJP][pnPN]*[gG]'))
        input_mask_list = sorted(input_mask_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
        mask_list_cycle = read_imgs(input_mask_list)
        mask_load_time = time.perf_counter() - mask_load_start
        print(f"[Daemon] Загрузка масок: {format_time(mask_load_time)} (масок: {len(mask_list_cycle)})")
        
        # Валидация
        if len(coord_list_cycle) == 0:
            raise ValueError(f"Координаты аватара пусты для {task_id}")
        if len(frame_list_cycle) == 0:
            raise ValueError(f"Кадры аватара пусты для {task_id}")
        if len(input_latent_list_cycle) == 0:
            raise ValueError(f"Latents аватара пусты для {task_id}")
        
        # Сохраняем в кеш
        materials = {
            'coord_list_cycle': coord_list_cycle,
            'frame_list_cycle': frame_list_cycle,
            'input_latent_list_cycle': input_latent_list_cycle,
            'mask_list_cycle': mask_list_cycle,
            'mask_coords_list_cycle': mask_coords_list_cycle
        }
        
        self.avatar_cache[task_id] = materials
        
        total_load_time = time.perf_counter() - load_start
        print(f"[Daemon] Материалы аватара {task_id} загружены в память: {format_time(total_load_time)}")
        print(f"[Daemon] Материалы будут использоваться из кеша при последующих запросах")
        
        return materials
    
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
            except:
                idx += 1
                continue
            
            combine_frame = get_image_blending(ori_frame, res_frame, bbox, mask, mask_crop_box)
            cv2.imwrite(f"{result_img_save_path}/{str(idx).zfill(8)}.png", combine_frame)
            idx += 1
    
    def process_request(self, config_path):
        """Обрабатывает запрос на инференс"""
        with self.lock:
            try:
                request_start = time.perf_counter()
                print(f"[Daemon] Начало обработки: {config_path}")
                
                # Загрузка конфига
                config_start = time.perf_counter()
                inference_config = OmegaConf.load(config_path)
                config_time = time.perf_counter() - config_start
                print(f"[Daemon] Загрузка конфига: {format_time(config_time)}")
                
                # Параметры из конфига или по умолчанию
                audio_padding_length_left = inference_config.get("audio_padding_length_left", 2)
                audio_padding_length_right = inference_config.get("audio_padding_length_right", 2)
                batch_size = inference_config.get("batch_size", 8)
                fps = inference_config.get("fps", 25)
                extra_margin = inference_config.get("extra_margin", 10)
                parsing_mode = inference_config.get("parsing_mode", "jaw")
                # Параметры use_saved_coord и saved_coord больше не используются
                # (координаты всегда берутся из подготовленного аватара),
                # но оставлены для обратной совместимости с конфигами
                use_saved_coord = inference_config.get("use_saved_coord", True)
                saved_coord = inference_config.get("saved_coord", False)
                
                # Определение result_dir
                result_dir = inference_config.get("result_dir", self.result_dir)
                
                # Обработка каждой задачи
                for task_id in inference_config:
                    if task_id in ["audio_padding_length_left", "audio_padding_length_right", 
                                  "batch_size", "fps", "extra_margin", "parsing_mode",
                                  "use_saved_coord", "saved_coord", "result_dir"]:
                        continue
                    
                    try:
                        task_start = time.perf_counter()
                        print(f"[Daemon] Обработка задачи: {task_id}")
                        
                        # Получение конфигурации задачи
                        video_path = inference_config[task_id]["video_path"]
                        
                        # Определение пути к подготовленному аватару
                        if self.version_arg == "v15":
                            avatar_base_path = f"./results/{self.version_arg}/avatars/{task_id}"
                        else:  # v1
                            avatar_base_path = f"./results/avatars/{task_id}"
                        
                        # Поддержка обоих форматов
                        if "audio_path" in inference_config[task_id]:
                            audio_paths = [inference_config[task_id]["audio_path"]]
                        elif "audio_clips" in inference_config[task_id]:
                            audio_paths = list(inference_config[task_id]["audio_clips"].values())
                        else:
                            raise ValueError(f"Не найден audio_path или audio_clips в конфиге для {task_id}")
                        
                        if "result_name" in inference_config[task_id]:
                            output_vid_name = inference_config[task_id]["result_name"]
                        else:
                            output_vid_name = None
                        
                        # bbox_shift
                        if self.version_arg == "v15":
                            bbox_shift = 0
                        else:
                            bbox_shift = inference_config[task_id].get("bbox_shift", 0)
                        
                        # Обработка каждого аудио файла
                        input_basename = os.path.basename(video_path).split('.')[0]
                        
                        # Определяем FPS из video_path (если это видео) для использования в аудио обработке
                        if get_file_type(video_path) == "video":
                            video_fps = get_video_fps(video_path)
                        else:
                            video_fps = fps
                        
                        # Загрузка материалов аватара (с кешированием в памяти)
                        avatar_materials = self._load_avatar_materials(task_id, avatar_base_path)
                        
                        # Извлекаем материалы из кеша
                        coord_list_cycle = avatar_materials['coord_list_cycle']
                        frame_list_cycle = avatar_materials['frame_list_cycle']
                        input_latent_list_cycle = avatar_materials['input_latent_list_cycle']
                        mask_list_cycle = avatar_materials['mask_list_cycle']
                        mask_coords_list_cycle = avatar_materials['mask_coords_list_cycle']
                        
                        # Обработка каждого аудио файла
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
                            
                            # Извлечение аудио фич
                            print(f"[Daemon] Извлечение аудио фич для {audio_path}...")
                            audio_features_start = time.perf_counter()
                            whisper_input_features, librosa_length = self.audio_processor.get_audio_feature(audio_path, weight_dtype=self.weight_dtype)
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
                            audio_features_time = time.perf_counter() - audio_features_start
                            print(f"[Daemon] Извлечение аудио фич: {format_time(audio_features_time)}")
                            
                            # Инференс с пайплайнингом
                            print("[Daemon] Запуск инференса с пайплайнингом...")
                            inference_start = time.perf_counter()
                            video_num = len(whisper_chunks)
                            
                            # Создаем очередь для пайплайнинга
                            res_frame_queue = queue.Queue()
                            
                            # Запускаем поток обработки кадров параллельно с инференсом
                            process_thread = threading.Thread(
                                target=self._process_frames_pipeline,
                                args=(
                                    res_frame_queue, video_num, coord_list_cycle,
                                    frame_list_cycle, mask_list_cycle, mask_coords_list_cycle,
                                    coord_placeholder, result_img_save_path
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
                                
                                # Отправляем кадры в очередь для параллельной обработки
                                for res_frame in recon:
                                    res_frame_queue.put(res_frame)
                                    frames_generated += 1
                            
                            inference_time = time.perf_counter() - inference_start
                            print(f"[Daemon] Инференс завершен: {format_time(inference_time)} (сгенерировано кадров: {frames_generated})")
                            
                            # Обновляем video_num на реальное количество сгенерированных кадров
                            # (на случай, если оно отличается от ожидаемого)
                            if frames_generated != video_num:
                                print(f"[Daemon] Предупреждение: ожидалось {video_num} кадров, сгенерировано {frames_generated}")
                            
                            # Ждем завершения обработки всех кадров
                            print(f"[Daemon] Ожидание завершения объединения кадров...", flush=True)
                            combine_start = time.perf_counter()
                            process_thread.join(timeout=300)  # Таймаут 5 минут на случай зависания
                            if process_thread.is_alive():
                                print("[Daemon] Предупреждение: поток обработки кадров не завершился в течение таймаута", flush=True)
                            combine_time = time.perf_counter() - combine_start
                            print(f"[Daemon] Объединение с оригинальными кадрами завершено: {format_time(combine_time)}", flush=True)
                            print(f"[Daemon] Сохранено кадров: {frames_generated}", flush=True)
                            
                            total_pipeline_time = time.perf_counter() - inference_start
                            print(f"[Daemon] Общее время пайплайна (инференс + объединение): {format_time(total_pipeline_time)}", flush=True)
                            
                            # Сохранение видео
                            temp_vid_path = f"{temp_dir}/temp_{input_basename}_{audio_basename}.mp4"
                            # Проверяем наличие кадров
                            saved_frames = glob.glob(os.path.join(result_img_save_path, '*.png'))
                            print(f"[Daemon] Проверка: найдено {len(saved_frames)} кадров для генерации видео", flush=True)
                            if len(saved_frames) == 0:
                                raise ValueError(f"Не найдено кадров в {result_img_save_path}")
                            
                            cmd_img2video = f"ffmpeg -y -v warning -r {video_fps} -f image2 -i {result_img_save_path}/%08d.png -vcodec libx264 -vf format=yuv420p -crf 18 {temp_vid_path}"
                            print(f"[Daemon] Генерация видео...", flush=True)
                            print(f"[Daemon] Команда: {cmd_img2video}", flush=True)
                            video_gen_start = time.perf_counter()
                            try:
                                result = subprocess.run(
                                    shlex.split(cmd_img2video),
                                    capture_output=True,
                                    text=True,
                                    check=True,
                                    timeout=600  # 10 минут максимум
                                )
                                video_gen_time = time.perf_counter() - video_gen_start
                                print(f"[Daemon] Генерация видео завершена: {format_time(video_gen_time)}", flush=True)
                                if result.stderr:
                                    print(f"[Daemon] FFmpeg stderr: {result.stderr[:500]}", flush=True)
                            except subprocess.TimeoutExpired:
                                print(f"[Daemon] ОШИБКА: Генерация видео превысила лимит времени (10 минут)", flush=True)
                                raise
                            except subprocess.CalledProcessError as e:
                                print(f"[Daemon] ОШИБКА при генерации видео: {e}", flush=True)
                                print(f"[Daemon] stderr: {e.stderr[:500] if e.stderr else 'нет'}", flush=True)
                                raise
                            
                            # Проверяем существование временного видео
                            if not os.path.exists(temp_vid_path):
                                raise FileNotFoundError(f"Временное видео не найдено: {temp_vid_path}")
                            print(f"[Daemon] Временное видео найдено: {temp_vid_path} ({os.path.getsize(temp_vid_path)} байт)", flush=True)
                            
                            # Проверяем существование аудио файла
                            if not os.path.exists(audio_path):
                                raise FileNotFoundError(f"Аудио файл не найден: {audio_path}")
                            print(f"[Daemon] Аудио файл найден: {audio_path} ({os.path.getsize(audio_path)} байт)", flush=True)
                            
                            cmd_combine_audio = f"ffmpeg -y -v warning -i {audio_path} -i {temp_vid_path} -c:v copy -c:a aac -shortest {current_output_vid_name}"
                            print(f"[Daemon] Объединение с аудио...", flush=True)
                            print(f"[Daemon] Команда: {cmd_combine_audio}", flush=True)
                            audio_combine_start = time.perf_counter()
                            try:
                                result = subprocess.run(
                                    shlex.split(cmd_combine_audio),
                                    capture_output=True,
                                    text=True,
                                    check=True,
                                    timeout=600  # 10 минут максимум
                                )
                                audio_combine_time = time.perf_counter() - audio_combine_start
                                print(f"[Daemon] Объединение с аудио завершено: {format_time(audio_combine_time)}", flush=True)
                                if result.stderr:
                                    print(f"[Daemon] FFmpeg stderr: {result.stderr[:500]}", flush=True)
                                
                                # Проверяем, что итоговый файл создан
                                if os.path.exists(current_output_vid_name):
                                    file_size = os.path.getsize(current_output_vid_name)
                                    print(f"[Daemon] Итоговый файл создан: {current_output_vid_name} ({file_size} байт)", flush=True)
                                else:
                                    raise FileNotFoundError(f"Итоговый файл не создан: {current_output_vid_name}")
                            except subprocess.TimeoutExpired:
                                print(f"[Daemon] ОШИБКА: Объединение с аудио превысило лимит времени (10 минут)", flush=True)
                                raise
                            except subprocess.CalledProcessError as e:
                                print(f"[Daemon] ОШИБКА при объединении с аудио: {e}", flush=True)
                                print(f"[Daemon] stderr: {e.stderr[:500] if e.stderr else 'нет'}", flush=True)
                                raise
                            
                            # Очистка
                            cleanup_temp_start = time.perf_counter()
                            print(f"[Daemon] Очистка временных файлов...", flush=True)
                            if os.path.exists(result_img_save_path):
                                shutil.rmtree(result_img_save_path)
                            if os.path.exists(temp_vid_path):
                                os.remove(temp_vid_path)
                            cleanup_temp_time = time.perf_counter() - cleanup_temp_start
                            
                            print(f"[Daemon] Очистка временных файлов: {format_time(cleanup_temp_time)}", flush=True)
                            print(f"[Daemon] Результат сохранен: {current_output_vid_name}", flush=True)
                        
                        # Финальная очистка
                        task_time = time.perf_counter() - task_start
                        print(f"[Daemon] Задача {task_id} завершена успешно! Общее время задачи: {format_time(task_time)}", flush=True)
                        
                    except Exception as e:
                        print(f"[Daemon] Ошибка при обработке задачи {task_id}: {e}")
                        import traceback
                        traceback.print_exc()
                
                # Перемещаем обработанный конфиг
                move_start = time.perf_counter()
                processed_dir = os.path.join(self.request_dir, "processed")
                os.makedirs(processed_dir, exist_ok=True)
                processed_path = os.path.join(processed_dir, os.path.basename(config_path))
                if os.path.exists(config_path):
                    shutil.move(config_path, processed_path)
                    print(f"[Daemon] Конфиг перемещен в: {processed_path}")
                # Удаляем из обработанных
                if config_path in self.processed_files:
                    self.processed_files.remove(config_path)
                move_time = time.perf_counter() - move_start
                
                request_time = time.perf_counter() - request_start
                print(f"[Daemon] Перемещение конфига: {format_time(move_time)}", flush=True)
                print(f"[Daemon] Обработка запроса завершена. Общее время: {format_time(request_time)}", flush=True)
                
            except Exception as e:
                print(f"[Daemon] Ошибка при обработке запроса: {e}")
                import traceback
                traceback.print_exc()
    
    def _check_for_new_configs(self):
        """Проверяет наличие новых конфигов в директории запросов"""
        processed_dir = os.path.join(self.request_dir, "processed")
        
        # Проверяем .yaml файлы (только в корне request_dir, не в поддиректориях)
        for config_file in Path(self.request_dir).glob("*.yaml"):
            # Пропускаем файлы из processed директории
            if "processed" in str(config_file):
                continue
            if str(config_file) not in self.processed_files:
                # Проверяем, что файл полностью записан (размер не меняется)
                time.sleep(0.5)
                if str(config_file) not in self.processed_files and os.path.exists(config_file):
                    self.processed_files.add(str(config_file))
                    print(f"[Daemon] Обнаружен новый конфиг: {config_file}")
                    threading.Thread(target=self.process_request, args=(str(config_file),), daemon=True).start()
        
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
        # Обрабатываем существующие файлы
        self._check_for_new_configs()
        
        print(f"[Daemon] Сервис запущен и следит за директорией: {self.request_dir}")
        print("[Daemon] Для остановки нажмите Ctrl+C")
        
        try:
            while True:
                self._check_for_new_configs()
                time.sleep(2)  # Проверяем каждые 2 секунды
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
    
    args = parser.parse_args()
    
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
        result_dir=args.result_dir
    )
    
    # Запуск сервиса
    service.run()


if __name__ == "__main__":
    main()

