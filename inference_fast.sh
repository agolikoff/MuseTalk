#!/bin/bash

# Быстрый скрипт инференса с использованием предзагруженных моделей
# Использование: sh inference_fast.sh <config_path> [version] [mode] [gpu_id]
# Требует запущенный демон-сервис через start_model_service.sh

# Проверка аргументов
if [ "$#" -lt 1 ]; then
    echo "Использование: $0 <config_path> [version] [mode]"
    echo "Пример: $0 ./configs/inference/test.yaml v1.5 normal"
    echo ""
    echo "ВАЖНО: Сначала запустите демон-сервис:"
    echo "  sh start_model_service.sh v1.5 normal 0"
    exit 1
fi

config_path=$1
version=${2:-"v1.5"}
mode=${3:-"normal"}

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

# Проверка существования конфига
if [ ! -f "$config_path" ]; then
    echo "Ошибка: Конфиг не найден: $config_path"
    exit 1
fi

# Проверка, запущен ли демон-сервис
pid_file="./logs/model_service_${version}_${mode}.pid"
if [ ! -f "$pid_file" ]; then
    echo "Ошибка: Демон-сервис не запущен!"
    echo "Запустите сервис: sh start_model_service.sh $version $mode 0"
    exit 1
fi

pid=$(cat "$pid_file")
if ! ps -p "$pid" > /dev/null 2>&1; then
    echo "Ошибка: Демон-сервис не запущен (PID $pid не найден)!"
    echo "Запустите сервис: sh start_model_service.sh $version $mode 0"
    exit 1
fi

# Определение result_dir
if [ "$mode" = "normal" ]; then
    result_dir="./results/test"
else
    result_dir="./results/realtime"
fi

echo "=========================================="
echo "Быстрый инференс через демон-сервис"
echo "=========================================="
echo "Конфиг: $config_path"
echo "Версия: $version"
echo "Режим: $mode"
echo "Результаты: $result_dir"
echo "=========================================="
echo ""

# Запись времени начала
start_time=$(date +%s)
start_time_readable=$(date '+%Y-%m-%d %H:%M:%S')
echo "Время начала: $start_time_readable"
echo ""

# Создаем директорию для запросов
request_dir="./requests"
mkdir -p "$request_dir"

# Копируем конфиг в директорию запросов
request_file="$request_dir/request_$(date +%Y%m%d_%H%M%S).yaml"
cp "$config_path" "$request_file"

echo "Запрос отправлен в демон-сервис: $request_file"
echo "Ожидание обработки..."
echo ""

# Ждем обработки (проверяем каждые 2 секунды)
max_wait=3600  # Максимум 1 час
waited=0
while [ $waited -lt $max_wait ]; do
    if [ ! -f "$request_file" ]; then
        # Файл перемещен в processed - обработка завершена
        end_time=$(date +%s)
        end_time_readable=$(date '+%Y-%m-%d %H:%M:%S')
        elapsed_time=$((end_time - start_time))
        hours=$((elapsed_time / 3600))
        minutes=$(((elapsed_time % 3600) / 60))
        seconds=$((elapsed_time % 60))
        
        echo ""
        echo "=========================================="
        echo "Обработка завершена!"
        echo "=========================================="
        echo "Время начала:  $start_time_readable"
        echo "Время окончания: $end_time_readable"
        echo "----------------------------------------"
        if [ $hours -gt 0 ]; then
            printf "Время выполнения: %02d:%02d:%02d (ч:м:с)\n" $hours $minutes $seconds
        else
            printf "Время выполнения: %02d:%02d (м:с)\n" $minutes $seconds
        fi
        echo "Результаты сохранены в: $result_dir"
        echo "=========================================="
        exit 0
    fi
    sleep 2
    waited=$((waited + 2))
    echo -n "."
done

end_time=$(date +%s)
end_time_readable=$(date '+%Y-%m-%d %H:%M:%S')
elapsed_time=$((end_time - start_time))
hours=$((elapsed_time / 3600))
minutes=$(((elapsed_time % 3600) / 60))
seconds=$((elapsed_time % 60))

echo ""
echo "=========================================="
echo "Предупреждение: Превышено время ожидания"
echo "=========================================="
echo "Время начала:  $start_time_readable"
echo "Время окончания: $end_time_readable"
if [ $hours -gt 0 ]; then
    printf "Время ожидания: %02d:%02d:%02d (ч:м:с)\n" $hours $minutes $seconds
else
    printf "Время ожидания: %02d:%02d (м:с)\n" $minutes $seconds
fi
echo "Проверьте логи сервиса: ./logs/model_service_${version}_${mode}_*.log"
echo "=========================================="

