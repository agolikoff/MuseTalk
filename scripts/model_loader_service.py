#!/usr/bin/env python3
"""
Сервис для предзагрузки моделей MuseTalk в память.
Модели загружаются один раз при старте и остаются в памяти для быстрого инференса.
"""
import os
import sys
import json
import time
import argparse
import subprocess
import threading
import queue
from pathlib import Path
from omegaconf import OmegaConf
import torch
from transformers import WhisperModel

from musetalk.utils.blending import get_image
from musetalk.utils.face_parsing import FaceParsing
from musetalk.utils.audio_processor import AudioProcessor
from musetalk.utils.utils import get_file_type, get_video_fps, datagen, load_all_model
from musetalk.utils.preprocessing import get_landmark_and_bbox, read_imgs, coord_placeholder
import cv2
import copy
import glob
import pickle
import shutil
import numpy as np
from tqdm import tqdm


def fast_check_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except:
        return False


class ModelService:
    """Сервис с предзагруженными моделями для быстрого инференса"""
    
    def __init__(self, version="v15", gpu_id=0, use_float16=True, 
                 whisper_dir="./models/whisper", vae_type="sd-vae",
                 ffmpeg_path="./ffmpeg-4.4-amd64-static/",
                 left_cheek_width=90, right_cheek_width=90):
        self.version = version
        self.gpu_id = gpu_id
        self.use_float16 = use_float16
        self.whisper_dir = whisper_dir
        self.vae_type = vae_type
        self.ffmpeg_path = ffmpeg_path
        self.left_cheek_width = left_cheek_width
        self.right_cheek_width = right_cheek_width
        
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
        
        print(f"[ModelService] Инициализация сервиса для версии {version}...")
        self._load_models()
        print("[ModelService] Все модели загружены в память!")
    
    def _load_models(self):
        """Загружает все модели в память"""
        # Настройка ffmpeg
        if not fast_check_ffmpeg():
            print("Добавление ffmpeg в PATH")
            path_separator = ';' if sys.platform == 'win32' else ':'
            os.environ["PATH"] = f"{self.ffmpeg_path}{path_separator}{os.environ['PATH']}"
            if not fast_check_ffmpeg():
                print("Предупреждение: Не удалось найти ffmpeg")
        
        # Устройство
        self.device = torch.device(f"cuda:{self.gpu_id}" if torch.cuda.is_available() else "cpu")
        print(f"[ModelService] Используется устройство: {self.device}")
        
        # Загрузка основных моделей
        print("[ModelService] Загрузка VAE, UNet и PositionalEncoding...")
        self.vae, self.unet, self.pe = load_all_model(
            unet_model_path=self.unet_model_path,
            vae_type=self.vae_type,
            unet_config=self.unet_config,
            device=self.device
        )
        
        # Установка типа данных
        if self.use_float16:
            print("[ModelService] Конвертация в float16...")
            self.pe = self.pe.half()
            self.vae.vae = self.vae.vae.half()
            self.unet.model = self.unet.model.half()
            self.weight_dtype = torch.float16
        else:
            self.weight_dtype = torch.float32
        
        # Перемещение на устройство
        self.pe = self.pe.to(self.device)
        self.vae.vae = self.vae.vae.to(self.device)
        self.unet.model = self.unet.model.to(self.device)
        
        # Загрузка Whisper
        print("[ModelService] Загрузка Whisper модели...")
        self.audio_processor = AudioProcessor(feature_extractor_path=self.whisper_dir)
        self.whisper = WhisperModel.from_pretrained(self.whisper_dir)
        self.whisper = self.whisper.to(device=self.device, dtype=self.weight_dtype).eval()
        self.whisper.requires_grad_(False)
        
        # Загрузка FaceParser
        print("[ModelService] Загрузка FaceParser...")
        if self.version_arg == "v15":
            self.fp = FaceParsing(
                left_cheek_width=self.left_cheek_width,
                right_cheek_width=self.right_cheek_width
            )
        else:
            self.fp = FaceParsing()
        
        # Timesteps
        self.timesteps = torch.tensor([0], device=self.device)
        
        print("[ModelService] Все модели успешно загружены!")
    
    @torch.no_grad()
    def process_inference(self, config_path, result_dir=None, 
                         audio_padding_length_left=2, audio_padding_length_right=2,
                         batch_size=8, fps=25, extra_margin=10, parsing_mode="jaw",
                         use_saved_coord=True, saved_coord=False, output_vid_name=None):
        """
        Быстрый инференс с использованием предзагруженных моделей
        
        Args:
            config_path: путь к конфигу inference
            result_dir: директория для результатов
            остальные параметры: параметры инференса
        """
        # Загрузка конфига
        inference_config = OmegaConf.load(config_path)
        print(f"[ModelService] Загружен конфиг: {config_path}")
        
        # Определение result_dir
        if result_dir is None:
            if "normal" in config_path.lower() or "test" in config_path.lower():
                result_dir = "./results/test"
            else:
                result_dir = "./results/realtime"
        
        # Обработка каждой задачи
        for task_id in inference_config:
            try:
                print(f"[ModelService] Обработка задачи: {task_id}")
                
                # Получение конфигурации задачи
                video_path = inference_config[task_id]["video_path"]
                
                # Поддержка обоих форматов: audio_path (normal) и audio_clips (realtime)
                if "audio_path" in inference_config[task_id]:
                    # Normal формат - один аудио файл
                    audio_paths = [inference_config[task_id]["audio_path"]]
                elif "audio_clips" in inference_config[task_id]:
                    # Realtime формат - несколько аудио файлов
                    audio_paths = list(inference_config[task_id]["audio_clips"].values())
                else:
                    raise ValueError(f"Не найден audio_path или audio_clips в конфиге для {task_id}")
                
                if "result_name" in inference_config[task_id]:
                    output_vid_name = inference_config[task_id]["result_name"]
                
                # bbox_shift
                if self.version_arg == "v15":
                    bbox_shift = 0
                else:
                    bbox_shift = inference_config[task_id].get("bbox_shift", 0)
                
                # Обработка каждого аудио файла
                input_basename = os.path.basename(video_path).split('.')[0]
                
                # Извлечение кадров (делаем один раз для всех аудио)
                if get_file_type(video_path) == "video":
                    temp_dir = os.path.join(result_dir, f"{self.version_arg}")
                    os.makedirs(temp_dir, exist_ok=True)
                    save_dir_full = os.path.join(temp_dir, input_basename)
                    os.makedirs(save_dir_full, exist_ok=True)
                    cmd = f"ffmpeg -v fatal -i {video_path} -start_number 0 {save_dir_full}/%08d.png"
                    os.system(cmd)
                    input_img_list = sorted(glob.glob(os.path.join(save_dir_full, '*.[jpJP][pnPN]*[gG]')))
                    video_fps = get_video_fps(video_path)
                elif get_file_type(video_path) == "image":
                    input_img_list = [video_path]
                    video_fps = fps
                elif os.path.isdir(video_path):
                    input_img_list = glob.glob(os.path.join(video_path, '*.[jpJP][pnPN]*[gG]'))
                    input_img_list = sorted(input_img_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
                    video_fps = fps
                else:
                    raise ValueError(f"{video_path} должен быть видео, изображением или директорией")
                
                crop_coord_save_path = os.path.join(result_dir, "../", input_basename+".pkl")
                
                # Предобработка изображений (делаем один раз для всех аудио)
                if os.path.exists(crop_coord_save_path) and use_saved_coord:
                    print("[ModelService] Использование сохраненных координат")
                    with open(crop_coord_save_path, 'rb') as f:
                        coord_list = pickle.load(f)
                    frame_list = read_imgs(input_img_list)
                else:
                    print("[ModelService] Извлечение landmarks...")
                    coord_list, frame_list = get_landmark_and_bbox(input_img_list, bbox_shift)
                    if saved_coord:
                        with open(crop_coord_save_path, 'wb') as f:
                            pickle.dump(coord_list, f)
                
                print(f"[ModelService] Количество кадров: {len(frame_list)}")
                
                # Обработка кадров (делаем один раз для всех аудио)
                input_latent_list = []
                for bbox, frame in zip(coord_list, frame_list):
                    if bbox == coord_placeholder:
                        continue
                    x1, y1, x2, y2 = bbox
                    if self.version_arg == "v15":
                        y2 = y2 + extra_margin
                        y2 = min(y2, frame.shape[0])
                    crop_frame = frame[y1:y2, x1:x2]
                    crop_frame = cv2.resize(crop_frame, (256, 256), interpolation=cv2.INTER_LANCZOS4)
                    latents = self.vae.get_latents_for_unet(crop_frame)
                    input_latent_list.append(latents)
                
                # Сглаживание
                frame_list_cycle = frame_list + frame_list[::-1]
                coord_list_cycle = coord_list + coord_list[::-1]
                input_latent_list_cycle = input_latent_list + input_latent_list[::-1]
                
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
                        # Если указано имя, добавляем индекс для множественных аудио
                        if len(audio_paths) > 1:
                            base_name = os.path.splitext(output_vid_name)[0]
                            ext = os.path.splitext(output_vid_name)[1]
                            current_output_vid_name = os.path.join(temp_dir, f"{base_name}_{audio_idx}{ext}")
                        else:
                            current_output_vid_name = os.path.join(temp_dir, output_vid_name)
                    
                    # Извлечение аудио фич
                    print(f"[ModelService] Извлечение аудио фич для {audio_path}...")
                    whisper_input_features, librosa_length = self.audio_processor.get_audio_feature(audio_path)
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
                    
                    # Инференс
                    print("[ModelService] Запуск инференса...")
                    video_num = len(whisper_chunks)
                    gen = datagen(
                        whisper_chunks=whisper_chunks,
                        vae_encode_latents=input_latent_list_cycle,
                        batch_size=batch_size,
                        delay_frame=0,
                        device=self.device,
                    )
                    
                    res_frame_list = []
                    total = int(np.ceil(float(video_num) / batch_size))
                    
                    for i, (whisper_batch, latent_batch) in enumerate(tqdm(gen, total=total)):
                        audio_feature_batch = self.pe(whisper_batch)
                        latent_batch = latent_batch.to(dtype=self.unet.model.dtype)
                        
                        pred_latents = self.unet.model(latent_batch, self.timesteps, encoder_hidden_states=audio_feature_batch).sample
                        recon = self.vae.decode_latents(pred_latents)
                        for res_frame in recon:
                            res_frame_list.append(res_frame)
                    
                    # Объединение с оригинальными кадрами
                    print("[ModelService] Объединение с оригинальными кадрами...")
                    for i, res_frame in enumerate(tqdm(res_frame_list)):
                        bbox = coord_list_cycle[i % (len(coord_list_cycle))]
                        ori_frame = copy.deepcopy(frame_list_cycle[i % (len(frame_list_cycle))])
                        x1, y1, x2, y2 = bbox
                        if self.version_arg == "v15":
                            y2 = y2 + extra_margin
                            y2 = min(y2, ori_frame.shape[0])
                        try:
                            res_frame = cv2.resize(res_frame.astype(np.uint8), (x2-x1, y2-y1))
                        except:
                            continue
                        
                        if self.version_arg == "v15":
                            combine_frame = get_image(ori_frame, res_frame, [x1, y1, x2, y2], mode=parsing_mode, fp=self.fp)
                        else:
                            combine_frame = get_image(ori_frame, res_frame, [x1, y1, x2, y2], fp=self.fp)
                        cv2.imwrite(f"{result_img_save_path}/{str(i).zfill(8)}.png", combine_frame)
                    
                    # Сохранение видео
                    temp_vid_path = f"{temp_dir}/temp_{input_basename}_{audio_basename}.mp4"
                    cmd_img2video = f"ffmpeg -y -v warning -r {video_fps} -f image2 -i {result_img_save_path}/%08d.png -vcodec libx264 -vf format=yuv420p -crf 18 {temp_vid_path}"
                    print(f"[ModelService] Генерация видео: {cmd_img2video}")
                    os.system(cmd_img2video)
                    
                    cmd_combine_audio = f"ffmpeg -y -v warning -i {audio_path} -i {temp_vid_path} {current_output_vid_name}"
                    print(f"[ModelService] Объединение с аудио: {cmd_combine_audio}")
                    os.system(cmd_combine_audio)
                    
                    # Очистка
                    shutil.rmtree(result_img_save_path)
                    os.remove(temp_vid_path)
                    
                    print(f"[ModelService] Результат сохранен: {current_output_vid_name}")
                
                # Финальная очистка
                if os.path.exists(save_dir_full):
                    shutil.rmtree(save_dir_full)
                if not saved_coord and os.path.exists(crop_coord_save_path):
                    os.remove(crop_coord_save_path)
                
            except Exception as e:
                print(f"[ModelService] Ошибка при обработке задачи {task_id}: {e}")
                import traceback
                traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description="Сервис предзагрузки моделей MuseTalk")
    parser.add_argument("--version", type=str, default="v1.5", choices=["v1.0", "v1.5", "v1", "v15"],
                       help="Версия модели")
    parser.add_argument("--gpu_id", type=int, default=0, help="ID GPU")
    parser.add_argument("--use_float16", action="store_true", help="Использовать float16")
    parser.add_argument("--whisper_dir", type=str, default="./models/whisper", help="Директория Whisper")
    parser.add_argument("--vae_type", type=str, default="sd-vae", help="Тип VAE")
    parser.add_argument("--ffmpeg_path", type=str, default="./ffmpeg-4.4-amd64-static/", help="Путь к ffmpeg")
    parser.add_argument("--left_cheek_width", type=int, default=90, help="Ширина левой щеки")
    parser.add_argument("--right_cheek_width", type=int, default=90, help="Ширина правой щеки")
    
    # Параметры для инференса
    parser.add_argument("--inference_config", type=str, required=True, help="Путь к конфигу инференса")
    parser.add_argument("--result_dir", type=str, default=None, help="Директория для результатов")
    parser.add_argument("--audio_padding_length_left", type=int, default=2, help="Левое аудио паддинг")
    parser.add_argument("--audio_padding_length_right", type=int, default=2, help="Правое аудио паддинг")
    parser.add_argument("--batch_size", type=int, default=8, help="Размер батча")
    parser.add_argument("--fps", type=int, default=25, help="FPS")
    parser.add_argument("--extra_margin", type=int, default=10, help="Дополнительный отступ")
    parser.add_argument("--parsing_mode", type=str, default="jaw", help="Режим парсинга")
    parser.add_argument("--use_saved_coord", action="store_true", help="Использовать сохраненные координаты")
    parser.add_argument("--saved_coord", action="store_true", help="Сохранить координаты")
    parser.add_argument("--output_vid_name", type=str, default=None, help="Имя выходного видео")
    
    args = parser.parse_args()
    
    # Создание сервиса
    service = ModelService(
        version=args.version,
        gpu_id=args.gpu_id,
        use_float16=args.use_float16,
        whisper_dir=args.whisper_dir,
        vae_type=args.vae_type,
        ffmpeg_path=args.ffmpeg_path,
        left_cheek_width=args.left_cheek_width,
        right_cheek_width=args.right_cheek_width
    )
    
    # Запуск инференса
    service.process_inference(
        config_path=args.inference_config,
        result_dir=args.result_dir,
        audio_padding_length_left=args.audio_padding_length_left,
        audio_padding_length_right=args.audio_padding_length_right,
        batch_size=args.batch_size,
        fps=args.fps,
        extra_margin=args.extra_margin,
        parsing_mode=args.parsing_mode,
        use_saved_coord=args.use_saved_coord,
        saved_coord=args.saved_coord,
        output_vid_name=args.output_vid_name
    )


if __name__ == "__main__":
    main()

