import os
import subprocess
import tempfile
from omegaconf import OmegaConf
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse

api = FastAPI()

AUDIO_DIR = "data/audio"
VIDEO_DIR = "data/video"
CONFIG_DIR = "configs/inference"
RESULT_BASE_NORMAL = "results/test"
RESULT_BASE_REALTIME = "results/realtime"

os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(VIDEO_DIR, exist_ok=True)
os.makedirs(CONFIG_DIR, exist_ok=True)


@api.post("/run")
async def run_inference(
    param: str = Form(...),
    input_file: UploadFile = File(...),
    version: str = Form(default="v1.5"),
    mode: str = Form(default="realtime")
):
    """
    Запускает инференс через inference_fast.sh
    
    Args:
        param: Имя аватара и задачи (используется для поиска видео data/video/{param}.mp4 и как task_id в конфиге)
        input_file: Аудио файл для обработки
        version: Версия модели (v1.0 или v1.5, по умолчанию v1.5)
        mode: Режим работы (normal или realtime, по умолчанию realtime)
    """
    # Валидация версии
    if version not in ["v1.0", "v1.5"]:
        raise HTTPException(
            status_code=400,
            detail="Неверная версия. Используйте v1.0 или v1.5"
        )
    
    # Валидация режима
    if mode not in ["normal", "realtime"]:
        raise HTTPException(
            status_code=400,
            detail="Неверный режим. Используйте normal или realtime"
        )
    
    # Сохраняем аудио файл
    audio_path = os.path.join(AUDIO_DIR, f"{param}.wav")
    with open(audio_path, "wb") as f:
        f.write(await input_file.read())
    
    # Формируем путь к видео файлу на основе param (имя аватара)
    video_path = os.path.join(VIDEO_DIR, f"{param}.mp4")
    
    # Проверяем существование видео файла
    if not os.path.exists(video_path):
        raise HTTPException(
            status_code=400,
            detail=f"Видео файл не найден: {video_path}"
        )
    
    # Создаем временный конфиг YAML
    config_data = OmegaConf.create({
        f"task_{param}": {
            "video_path": video_path,
            "audio_path": audio_path
        }
    })
    
    # Создаем временный файл конфига
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False, dir=CONFIG_DIR) as f:
        OmegaConf.save(config_data, f.name)
        config_path = f.name
    
    try:
        # Определяем путь к результатам в зависимости от режима
        if mode == "normal":
            result_dir = RESULT_BASE_NORMAL
        else:
            result_dir = RESULT_BASE_REALTIME
        
        # Преобразуем версию для пути (v1.5 -> v15, v1.0 -> v1)
        version_path = "v15" if version == "v1.5" else "v1"
        
        # Определяем ожидаемый путь к результату
        # Используем param как имя аватара
        audio_basename = os.path.splitext(os.path.basename(audio_path))[0]
        output_basename = f"{param}_{audio_basename}"
        
        # Команда для запуска inference_fast.sh
        cmd = [
            "conda", "run", "-n", "MuseTalk",
            "sh", "inference_fast.sh", config_path, version, mode
        ]
        
        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                cwd=os.path.abspath(os.path.dirname(__file__)),
            )
        except subprocess.CalledProcessError as e:
            raise HTTPException(
                status_code=500,
                detail={
                    "message": "Script failed",
                    "returncode": e.returncode,
                    "cmd": e.cmd,
                    "stdout": e.stdout,
                    "stderr": e.stderr,
                },
            )
        
        # Ищем результат в директории результатов
        # Результат сохраняется в result_dir/{version}/{output_basename}.mp4
        expected_output = os.path.join(result_dir, version_path, f"{output_basename}.mp4")
        
        # Также проверяем альтернативные пути
        possible_outputs = [
            expected_output,
            os.path.join(result_dir, version_path, f"{output_basename}_concat.mp4"),
            os.path.join(result_dir, f"{output_basename}.mp4"),
        ]
        
        # Ищем все mp4 файлы в result_dir, содержащие output_basename
        output_path = None
        for path in possible_outputs:
            if os.path.exists(path):
                output_path = path
                break
        
        # Если не нашли по ожидаемому пути, ищем рекурсивно
        if not output_path and os.path.exists(result_dir):
            for root, dirs, files in os.walk(result_dir):
                for file in files:
                    if file.endswith('.mp4') and output_basename in file:
                        output_path = os.path.join(root, file)
                        break
                if output_path:
                    break
        
        if not output_path or not os.path.exists(output_path):
            raise HTTPException(
                status_code=500,
                detail=f"Результирующий файл не был создан. Ожидался: {expected_output}. Проверьте директорию: {result_dir}"
            )
        
        return FileResponse(
            path=output_path,
            media_type="video/mp4",
            filename=f"{param}.mp4"
        )
    
    finally:
        # Удаляем временный конфиг
        if os.path.exists(config_path):
            os.remove(config_path)
