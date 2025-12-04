#!/bin/bash

# Скрипт для запуска сервиса предзагрузки моделей в фоне
# Использование: sh start_model_service.sh [v1.0|v1.5] [normal|realtime] [gpu_id]

# Проверка аргументов
if [ "$#" -lt 2 ]; then
    echo "Использование: $0 <version> <mode> [gpu_id]"
    echo "Пример: $0 v1.5 normal 0"
    exit 1
fi

version=$1
mode=$2
gpu_id=${3:-0}

# Валидация версии
if [ "$version" != "v1.0" ] && [ "$version" != "v1.5" ]; then
    echo "Неверная версия. Используйте v1.0 или v1.5"
    exit 1
fi

# Валидация режима
if [ "$mode" != "normal" ] && [ "$mode" != "realtime" ]; then
    echo "Неверный режим. Используйте normal или realtime"
    exit 1
fi

# Определение путей к конфигам
if [ "$mode" = "normal" ]; then
    config_path="./configs/inference/test.yaml"
    result_dir="./results/test"
else
    config_path="./configs/inference/realtime.yaml"
    result_dir="./results/realtime"
fi

# Определение путей к моделям
if [ "$version" = "v1.0" ]; then
    model_dir="./models/musetalk"
    unet_model_path="$model_dir/pytorch_model.bin"
    unet_config="$model_dir/musetalk.json"
    version_arg="v1.0"
elif [ "$version" = "v1.5" ]; then
    model_dir="./models/musetalkV15"
    unet_model_path="$model_dir/unet.pth"
    unet_config="$model_dir/musetalk.json"
    version_arg="v1.5"
fi

# Проверка существования моделей
if [ ! -f "$unet_model_path" ]; then
    echo "Ошибка: Модель не найдена: $unet_model_path"
    exit 1
fi

# Создание директории для логов
log_dir="./logs"
mkdir -p "$log_dir"

# Имя файла лога
log_file="$log_dir/model_service_${version}_${mode}_$(date +%Y%m%d_%H%M%S).log"
pid_file="$log_dir/model_service_${version}_${mode}.pid"

# Проверка, не запущен ли уже сервис
if [ -f "$pid_file" ]; then
    old_pid=$(cat "$pid_file")
    if ps -p "$old_pid" > /dev/null 2>&1; then
        echo "Сервис уже запущен с PID: $old_pid"
        echo "Остановите его перед запуском нового: kill $old_pid"
        exit 1
    else
        rm -f "$pid_file"
    fi
fi

echo "Запуск сервиса предзагрузки моделей..."
echo "Версия: $version"
echo "Режим: $mode"
echo "GPU ID: $gpu_id"
echo "Лог файл: $log_file"
echo "PID файл: $pid_file"

# Запуск демон-сервиса в фоне
# Используем -u для unbuffered output, чтобы логи сразу попадали в файл
nohup python3 -u -m scripts.model_daemon_service \
    --version "$version_arg" \
    --gpu_id "$gpu_id" \
    --use_float16 \
    --request_dir "./requests" \
    --result_dir "$result_dir" \
    > "$log_file" 2>&1 &

# Сохранение PID
echo $! > "$pid_file"

echo "Сервис запущен с PID: $(cat $pid_file)"
echo "Логи сохраняются в: $log_file"
echo ""
echo "Для остановки сервиса выполните:"
echo "  kill \$(cat $pid_file)"

