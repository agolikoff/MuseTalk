# WebRTC Streaming для MuseTalk

## Установка

Для работы WebRTC стриминга необходимо установить дополнительные библиотеки:

```bash
pip install aiortc av
```

## Требования к портам на сервере

Для работы WebRTC необходимо открыть следующие порты:

### Обязательные порты:

1. **HTTP/HTTPS порт** (для API):
   - По умолчанию: **8002** (TCP)
   - Используется для REST API и WebSocket соединений
   - Можно изменить через параметр `--port`

2. **UDP порты для RTP/RTCP** (для медиа потока):
   - Диапазон: **10000-20000** (UDP)
   - Используются для передачи аудио и видео данных
   - Можно настроить через переменные окружения

3. **STUN сервер** (опционально, но рекомендуется):
   - Порт: **3478** (UDP/TCP)
   - Используется для NAT traversal
   - В коде используется публичный STUN сервер Google: `stun:stun.l.google.com:19302`

### Настройка файрвола:

**Для Linux (ufw):**
```bash
# HTTP порт для API
sudo ufw allow 8002/tcp

# UDP порты для RTP/RTCP (диапазон)
sudo ufw allow 10000:20000/udp
```

**Для Linux (iptables):**
```bash
# HTTP порт для API
sudo iptables -A INPUT -p tcp --dport 8002 -j ACCEPT

# UDP порты для RTP/RTCP
sudo iptables -A INPUT -p udp --dport 10000:20000 -j ACCEPT
```

**Для Windows Firewall:**
1. Откройте "Брандмауэр Защитника Windows"
2. Добавьте правило для входящих подключений:
   - TCP порт 8002
   - UDP порты 10000-20000

**Для облачных провайдеров (AWS, GCP, Azure):**
- Добавьте правила в Security Groups / Firewall:
  - TCP: 8002
  - UDP: 10000-20000

### Настройка диапазона портов:

Можно настроить диапазон портов через переменные окружения:

```bash
export WEBRTC_RTC_MIN_PORT=10000
export WEBRTC_RTC_MAX_PORT=20000
```

Или в коде через параметры aiortc.

### Настройка TURN сервера:

По умолчанию используются публичные TURN серверы (openrelay.metered.ca). Для продакшена рекомендуется настроить собственный TURN сервер.

**Настройка через переменные окружения:**

```bash
export TURN_SERVER_URL="turn:your-turn-server.com:3478"
export TURN_SERVER_USERNAME="username"
export TURN_SERVER_CREDENTIAL="password"
```

**Отключение публичных TURN серверов:**

```bash
export USE_PUBLIC_TURN="false"
```

**Пример настройки собственного TURN сервера (coturn):**

1. Установите coturn:
```bash
sudo apt-get install coturn
```

2. Узнайте ваш публичный IP адрес:
```bash
curl ifconfig.me
# или
curl ipinfo.io/ip
```

3. Настройте `/etc/turnserver.conf`:
```conf
# Прослушиваемый порт
listening-port=3478

# ВАЖНО: Укажите ваш публичный IP адрес (можно использовать IP без домена)
external-ip=YOUR_PUBLIC_IP

# Realm можно указать любым (IP адрес или произвольное имя)
realm=YOUR_PUBLIC_IP
# или просто:
# realm=turnserver

# Пользователь и пароль для TURN
user=webrtc:your_secure_password_here

# Диапазон портов для релейного трафика
min-port=49152
max-port=65535

# Логирование (опционально, для отладки)
log-file=/var/log/turn.log
verbose
```

**Примечание:** Если у вас несколько сетевых интерфейсов или сервер за NAT, может потребоваться указать внутренний и внешний IP:
```conf
external-ip=INTERNAL_IP/EXTERNAL_IP
# Например:
# external-ip=10.0.0.5/93.173.41.137
```

4. Запустите coturn:
```bash
sudo systemctl enable coturn
sudo systemctl start coturn
```

5. Проверьте работу:
```bash
# Проверка статуса
sudo systemctl status coturn

# Просмотр логов
sudo tail -f /var/log/turn.log

# Проверка, что порт открыт
sudo netstat -tulpn | grep 3478
```

6. Настройте переменные окружения (используйте IP адрес):
```bash
export TURN_SERVER_URL="turn:YOUR_PUBLIC_IP:3478"
export TURN_SERVER_USERNAME="webrtc"
export TURN_SERVER_CREDENTIAL="your_secure_password_here"
export USE_PUBLIC_TURN="false"
```

**Пример с реальным IP:**
```bash
# Если ваш публичный IP: 93.173.41.137
export TURN_SERVER_URL="turn:93.173.41.137:3478"
export TURN_SERVER_USERNAME="webrtc"
export TURN_SERVER_CREDENTIAL="my_secure_password"
export USE_PUBLIC_TURN="false"
```

**Важно:** 
- Используйте IP адрес напрямую в URL (без домена)
- Убедитесь, что порты 3478 (UDP/TCP) и 49152-65535 (UDP) открыты в firewall
- Пароль должен совпадать с тем, что указан в `/etc/turnserver.conf`

## Запуск

1. Запустите WebRTC API сервер:
```bash
python streaming_webrtc_api.py --host 0.0.0.0 --port 8002 --version v15 --gpu_id 0
```

2. Откройте `streaming_webrtc_test.html` в браузере

3. Заполните форму и нажмите "Начать стриминг"

## Преимущества WebRTC

- ✅ Низкая задержка (real-time стриминг)
- ✅ Автоматическая адаптация качества
- ✅ Встроенная синхронизация аудио и видео
- ✅ Не требует сложной обработки сегментов
- ✅ Работает напрямую с медиа потоками

## Отличия от WebSocket стриминга

- WebSocket: отправка готовых видео сегментов через MSE
- WebRTC: передача кадров напрямую в медиа поток

WebRTC лучше подходит для real-time стриминга, так как он специально разработан для этого.

## Устранение проблем

### Проблема: "ICE connection failed" или соединение остается в состоянии "checking"
- **Проверьте логи сервера** - должны быть relay кандидаты от TURN сервера
- Если нет relay кандидатов - TURN сервер не работает:
  - Публичные TURN серверы могут быть перегружены или недоступны
  - Настройте собственный TURN сервер (см. раздел "Настройка TURN сервера")
- Проверьте, что UDP порты открыты (10000-20000)
- Убедитесь, что STUN сервер доступен
- Для сложных NAT/Firewall TURN сервер обязателен

### Проблема: "Connection timeout"
- Проверьте файрвол и порты
- Убедитесь, что сервер доступен извне
- Проверьте настройки роутера (если применимо)
- Убедитесь, что TURN сервер настроен и работает

### Проблема: "recv() не был вызван"
- Это означает, что WebRTC не может установить соединение
- Проверьте логи на наличие relay кандидатов
- Убедитесь, что TURN сервер настроен правильно
- Проверьте сетевые ограничения (NAT, Firewall)

