# Информация об аудио треках

## Какой трек используется?

По умолчанию используется **`DelayedMediaPlayerTrack`**, а не `SynchronizedAudioTrack`.

### DelayedMediaPlayerTrack (по умолчанию)
- Использует `MediaPlayer` из aiortc
- Логи: `[DelayedMediaPlayerTrack]`
- Преимущества: лучшее качество звука, автоматическая обработка форматов
- Недостатки: меньше контроля над синхронизацией

### SynchronizedAudioTrack (опционально)
- Кастомный трек с полным контролем
- Логи: `[SynchronizedAudioTrack]`
- Включается через переменную окружения: `USE_CUSTOM_AUDIO_TRACK=true`
- Преимущества: полный контроль над синхронизацией и PTS
- Недостатки: нужно вручную обрабатывать форматы

## Как переключиться на SynchronizedAudioTrack?

Установите переменную окружения перед запуском:
```bash
export USE_CUSTOM_AUDIO_TRACK=true
python streaming_webrtc_api.py
```

Или в коде:
```python
import os
os.environ["USE_CUSTOM_AUDIO_TRACK"] = "true"
```

## Исправления для DelayedMediaPlayerTrack

Я применил те же исправления, что и для SynchronizedAudioTrack:

1. **Копирование кадров**: Кадры от MediaPlayer теперь копируются перед отправкой
2. **Правильный PTS**: Начальный PTS сброшен на 0 для плавного старта
3. **Диагностика**: Добавлено логирование размеров кадров и PTS

## Что проверить в логах

### Для DelayedMediaPlayerTrack (по умолчанию):
```
[DelayedMediaPlayerTrack] recv вызван #1, audio_started=False
[DelayedMediaPlayerTrack] ✓ Аудио синхронизировано с видео
[DelayedMediaPlayerTrack] Получен кадр из MediaPlayer (samples=480, pts=..., duration=10.00ms)
```

### Для SynchronizedAudioTrack (если включен):
```
[SynchronizedAudioTrack] Инициализация: sample_rate=48000, channels=2
[SynchronizedAudioTrack] Аудио загружено: 1500 кадров
[SynchronizedAudioTrack] ✓ Все кадры имеют одинаковый размер: 480 samples
```

## Если заикание сохраняется

1. **Проверьте логи DelayedMediaPlayerTrack** - должны быть сообщения о получении кадров
2. **Попробуйте переключиться на SynchronizedAudioTrack** - может дать лучший контроль
3. **Проверьте размеры кадров** - должны быть постоянными (обычно 480 samples)
4. **Проверьте PTS** - должны увеличиваться последовательно

## Рекомендации

- **Для большинства случаев**: Используйте DelayedMediaPlayerTrack (по умолчанию)
- **Если нужен полный контроль**: Используйте SynchronizedAudioTrack
- **Если заикание**: Попробуйте оба варианта и сравните результаты


