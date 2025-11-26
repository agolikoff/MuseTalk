import argparse
import os
import sys
import numpy as np
import cv2
import torch
import glob
import pickle
import json
import shutil
from tqdm import tqdm

from musetalk.utils.face_parsing import FaceParsing
from musetalk.utils.preprocessing import get_landmark_and_bbox, read_imgs
from musetalk.utils.blending import get_image_prepare_material
from musetalk.models.vae import VAE
from omegaconf import OmegaConf


def fast_check_ffmpeg():
    try:
        import subprocess
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except:
        return False


def video2imgs(vid_path, save_path, ext='.png', cut_frame=10000000):
    cap = cv2.VideoCapture(vid_path)
    count = 0
    while True:
        if count > cut_frame:
            break
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(f"{save_path}/{count:08d}.png", frame)
            count += 1
        else:
            break


def osmakedirs(path_list):
    for path in path_list:
        os.makedirs(path) if not os.path.exists(path) else None


@torch.no_grad()
def prepare_avatar(avatar_id, video_path, bbox_shift, version, vae, fp, extra_margin, parsing_mode):
    """
    Подготовка аватара: извлечение кадров, ландмарков, латентов и масок.
    """
    # Определение базового пути в зависимости от версии
    if version == "v15":
        base_path = f"./results/{version}/avatars/{avatar_id}"
    else:  # v1
        base_path = f"./results/avatars/{avatar_id}"
    
    avatar_path = base_path
    full_imgs_path = f"{avatar_path}/full_imgs"
    coords_path = f"{avatar_path}/coords.pkl"
    latents_out_path = f"{avatar_path}/latents.pt"
    mask_out_path = f"{avatar_path}/mask"
    mask_coords_path = f"{avatar_path}/mask_coords.pkl"
    avatar_info_path = f"{avatar_path}/avator_info.json"
    
    avatar_info = {
        "avatar_id": avatar_id,
        "video_path": video_path,
        "bbox_shift": bbox_shift,
        "version": version
    }
    
    # Проверка существования аватара
    if os.path.exists(avatar_path):
        response = input(f"{avatar_id} уже существует. Пересоздать? (y/n): ")
        if response.lower() == "y":
            shutil.rmtree(avatar_path)
            print("*********************************")
            print(f"  Создание аватара: {avatar_id}")
            print("*********************************")
        else:
            print(f"Аватар {avatar_id} уже подготовлен. Используйте существующие данные.")
            return
    
    # Создание директорий
    print("*********************************")
    print(f"  Создание аватара: {avatar_id}")
    print("*********************************")
    osmakedirs([avatar_path, full_imgs_path, mask_out_path])
    
    # Сохранение информации об аватаре
    print("Подготовка материалов...")
    with open(avatar_info_path, "w") as f:
        json.dump(avatar_info, f, indent=2)
    
    # Извлечение кадров из видео
    if os.path.isfile(video_path):
        print(f"Извлечение кадров из видео: {video_path}")
        video2imgs(video_path, full_imgs_path, ext='png')
    else:
        print(f"Копирование файлов из директории: {video_path}")
        files = os.listdir(video_path)
        files.sort()
        files = [file for file in files if file.split(".")[-1].lower() in ["png", "jpg", "jpeg"]]
        for filename in files:
            shutil.copyfile(f"{video_path}/{filename}", f"{full_imgs_path}/{filename}")
    
    input_img_list = sorted(glob.glob(os.path.join(full_imgs_path, '*.[jpJP][pnPN]*[gG]')))
    
    if len(input_img_list) == 0:
        print(f"Ошибка: не найдено изображений в {full_imgs_path}")
        sys.exit(1)
    
    # Извлечение ландмарков
    print("Извлечение ландмарков...")
    coord_list, frame_list = get_landmark_and_bbox(input_img_list, bbox_shift)
    input_latent_list = []
    idx = -1
    coord_placeholder = (0.0, 0.0, 0.0, 0.0)
    
    for bbox, frame in zip(coord_list, frame_list):
        idx = idx + 1
        if bbox == coord_placeholder:
            continue
        x1, y1, x2, y2 = bbox
        if version == "v15":
            y2 = y2 + extra_margin
            y2 = min(y2, frame.shape[0])
            coord_list[idx] = [x1, y1, x2, y2]
        crop_frame = frame[y1:y2, x1:x2]
        resized_crop_frame = cv2.resize(crop_frame, (256, 256), interpolation=cv2.INTER_LANCZOS4)
        latents = vae.get_latents_for_unet(resized_crop_frame)
        input_latent_list.append(latents)
    
    # Создание циклических списков
    frame_list_cycle = frame_list + frame_list[::-1]
    coord_list_cycle = coord_list + coord_list[::-1]
    input_latent_list_cycle = input_latent_list + input_latent_list[::-1]
    mask_coords_list_cycle = []
    mask_list_cycle = []
    
    # Обработка кадров и создание масок
    print("Создание масок...")
    for i, frame in enumerate(tqdm(frame_list_cycle)):
        cv2.imwrite(f"{full_imgs_path}/{str(i).zfill(8)}.png", frame)
        
        x1, y1, x2, y2 = coord_list_cycle[i]
        if version == "v15":
            mode = parsing_mode
        else:
            mode = "raw"
        mask, crop_box = get_image_prepare_material(frame, [x1, y1, x2, y2], fp=fp, mode=mode)
        
        cv2.imwrite(f"{mask_out_path}/{str(i).zfill(8)}.png", mask)
        mask_coords_list_cycle += [crop_box]
        mask_list_cycle.append(mask)
    
    # Сохранение результатов
    print("Сохранение результатов...")
    with open(mask_coords_path, 'wb') as f:
        pickle.dump(mask_coords_list_cycle, f)
    
    with open(coords_path, 'wb') as f:
        pickle.dump(coord_list_cycle, f)
    
    torch.save(input_latent_list_cycle, latents_out_path)
    
    print(f"✓ Аватар {avatar_id} успешно подготовлен!")
    print(f"  Результаты сохранены в: {avatar_path}")


def main():
    parser = argparse.ArgumentParser(description="Скрипт для подготовки аватара к генерации")
    
    # Конфиг файл (альтернатива параметрам командной строки)
    parser.add_argument("--config", type=str, default=None, help="Путь к конфиг файлу (YAML)")
    
    # Основные параметры
    parser.add_argument("--avatar_id", type=str, default=None, help="ID аватара")
    parser.add_argument("--video_path", type=str, default=None, help="Путь к видео или директории с изображениями")
    parser.add_argument("--bbox_shift", type=int, default=5, help="Смещение bounding box")
    
    # Параметры модели
    parser.add_argument("--version", type=str, default="v15", choices=["v1", "v15"], help="Версия MuseTalk: v1 или v15")
    parser.add_argument("--gpu_id", type=int, default=0, help="ID GPU")
    parser.add_argument("--vae_type", type=str, default="sd-vae", help="Тип VAE модели")
    parser.add_argument("--use_float16", action="store_true", help="Использовать float16")
    
    # Параметры обработки
    parser.add_argument("--extra_margin", type=int, default=10, help="Дополнительный отступ для v15")
    parser.add_argument("--parsing_mode", type=str, default="jaw", help="Режим парсинга лица")
    parser.add_argument("--left_cheek_width", type=int, default=90, help="Ширина левой щеки")
    parser.add_argument("--right_cheek_width", type=int, default=90, help="Ширина правой щеки")
    
    # FFmpeg
    parser.add_argument("--ffmpeg_path", type=str, default="./ffmpeg-4.4-amd64-static/", help="Путь к ffmpeg")
    
    args = parser.parse_args()
    
    # Загрузка параметров из конфига, если указан
    if args.config:
        if not os.path.exists(args.config):
            print(f"Ошибка: конфиг файл не найден: {args.config}")
            sys.exit(1)
        config = OmegaConf.load(args.config)
        print(f"Загружен конфиг: {args.config}")
        print(config)
        
        # Проверка ffmpeg (один раз для всех аватаров)
        if not fast_check_ffmpeg():
            print("Добавление ffmpeg в PATH")
            path_separator = ';' if sys.platform == 'win32' else ':'
            os.environ["PATH"] = f"{args.ffmpeg_path}{path_separator}{os.environ['PATH']}"
            if not fast_check_ffmpeg():
                print("Предупреждение: Не удалось найти ffmpeg, убедитесь что он установлен")
        
        # Устройство
        device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
        print(f"Используется устройство: {device}")
        
        # Загрузка моделей (один раз для всех аватаров)
        print("Загрузка VAE модели...")
        vae_model_path = os.path.join("models", args.vae_type)
        vae = VAE(model_path=vae_model_path, use_float16=args.use_float16)
        if args.use_float16:
            vae.vae = vae.vae.half()
        vae.vae = vae.vae.to(device)
        vae.vae.eval()
        vae.vae.requires_grad_(False)
        
        # Инициализация FaceParsing
        print("Инициализация FaceParsing...")
        if args.version == "v15":
            fp = FaceParsing(
                left_cheek_width=args.left_cheek_width,
                right_cheek_width=args.right_cheek_width
            )
        else:  # v1
            fp = FaceParsing()
        
        # Обработка всех аватаров из конфига (игнорируем флаг preparation при использовании конфига)
        for avatar_id in config:
            avatar_config = config[avatar_id]
            
            # Использование параметров из конфига или аргументов командной строки
            final_avatar_id = args.avatar_id if args.avatar_id else avatar_id
            final_video_path = args.video_path if args.video_path else avatar_config.get("video_path")
            final_bbox_shift = args.bbox_shift if args.bbox_shift != 5 else avatar_config.get("bbox_shift", 5)
            
            if not final_video_path:
                print(f"Ошибка: video_path не указан для {avatar_id}")
                continue
            
            # Подготовка аватара
            prepare_avatar(
                avatar_id=final_avatar_id,
                video_path=final_video_path,
                bbox_shift=final_bbox_shift if args.version != "v15" else 0,
                version=args.version,
                vae=vae,
                fp=fp,
                extra_margin=args.extra_margin,
                parsing_mode=args.parsing_mode
            )
        
        return
    
    # Если конфиг не указан, используем параметры командной строки
    if not args.avatar_id or not args.video_path:
        parser.error("Необходимо указать --avatar_id и --video_path, либо использовать --config")
    
    # Проверка ffmpeg
    if not fast_check_ffmpeg():
        print("Добавление ffmpeg в PATH")
        path_separator = ';' if sys.platform == 'win32' else ':'
        os.environ["PATH"] = f"{args.ffmpeg_path}{path_separator}{os.environ['PATH']}"
        if not fast_check_ffmpeg():
            print("Предупреждение: Не удалось найти ffmpeg, убедитесь что он установлен")
    
    # Устройство
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"Используется устройство: {device}")
    
    # Загрузка VAE
    print("Загрузка VAE модели...")
    vae_model_path = os.path.join("models", args.vae_type)
    vae = VAE(model_path=vae_model_path, use_float16=args.use_float16)
    if args.use_float16:
        vae.vae = vae.vae.half()
    vae.vae = vae.vae.to(device)
    vae.vae.eval()
    vae.vae.requires_grad_(False)
    
    # Инициализация FaceParsing
    print("Инициализация FaceParsing...")
    if args.version == "v15":
        fp = FaceParsing(
            left_cheek_width=args.left_cheek_width,
            right_cheek_width=args.right_cheek_width
        )
    else:  # v1
        fp = FaceParsing()
    
    # Подготовка аватара
    prepare_avatar(
        avatar_id=args.avatar_id,
        video_path=args.video_path,
        bbox_shift=args.bbox_shift if args.version != "v15" else 0,
        version=args.version,
        vae=vae,
        fp=fp,
        extra_margin=args.extra_margin,
        parsing_mode=args.parsing_mode
    )


if __name__ == "__main__":
    main()

