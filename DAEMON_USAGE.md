# Быстрое руководство по использованию демон-сервиса

## Как это работает

1. **Запускаете демон один раз** - он загружает модели в память
2. **Отправляете запросы** - они обрабатываются быстро, без повторной загрузки моделей
3. **Модели остаются в памяти** - пока демон работает

## Пошаговая инструкция

### 1. Запуск демона (один раз)

```bash
# Запуск для версии v1.5 в режиме normal
sh start_model_service.sh v1.5 normal 0
```

Вы увидите:
```
[Daemon] Инициализация сервиса для версии v1.5...
[Daemon] Загрузка VAE, UNet и PositionalEncoding...
[Daemon] Загрузка Whisper модели...
[Daemon] Все модели загружены в память!
[Daemon] Сервис готов к обработке запросов из: ./requests
```

### 2. Отправка запросов (многократно)

```bash
# Отправка запроса на обработку
sh inference_fast.sh ./configs/inference/test.yaml v1.5 normal
```

Скрипт:
- Копирует конфиг в `./requests/`
- Ждет обработки демоном
- Показывает результат

### 3. Проверка статуса

```bash
# Проверка, что демон работает
ps aux | grep model_daemon_service

# Просмотр логов
tail -f ./logs/model_service_v1.5_normal_*.log
```

### 4. Остановка демона

```bash
# Мягкая остановка
kill $(cat ./logs/model_service_v1.5_normal.pid)

# Или принудительная
pkill -f model_daemon_service
```

## Примеры конфигов для запросов

### Простой запрос (normal mode)

Создайте файл `./requests/my_request.yaml`:

```yaml
task_1:
  video_path: "data/video/elon.mp4"
  audio_path: "data/audio/eng.wav"
  batch_size: 8
  fps: 25
  use_saved_coord: true
```

Демон автоматически обнаружит и обработает его!

### Множественные аудио (realtime mode)

```yaml
avatar_1:
  video_path: "data/video/yongen.mp4"
  audio_clips:
    audio_0: "data/audio/yongen.wav"
    audio_1: "data/audio/eng.wav"
  batch_size: 20
  fps: 25
```

## Преимущества

✅ **Быстро**: Модели загружаются один раз, не при каждом запросе  
✅ **Удобно**: Просто копируете конфиг в `./requests/`  
✅ **Эффективно**: Можно обрабатывать множество запросов подряд  
✅ **Надежно**: Логи всех операций сохраняются  

## Устранение проблем

### Демон не запускается

```bash
# Проверьте, не запущен ли уже
ps aux | grep model_daemon_service

# Проверьте логи
cat ./logs/model_service_*.log
```

### Запрос не обрабатывается

1. Убедитесь, что демон запущен: `ps aux | grep model_daemon_service`
2. Проверьте, что конфиг скопирован в `./requests/`
3. Проверьте логи демона: `tail -f ./logs/model_service_*.log`
4. Убедитесь, что пути в конфиге правильные

### Модели не загружаются

1. Проверьте наличие моделей:
   - v1.5: `./models/musetalkV15/unet.pth`
   - v1.0: `./models/musetalk/pytorch_model.bin`
2. Проверьте доступность GPU: `nvidia-smi`
3. Проверьте логи загрузки

