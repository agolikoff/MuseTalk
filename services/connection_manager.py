
import logging
import asyncio
import threading
from typing import Dict, Optional
import os

from aiortc import RTCPeerConnection, RTCSessionDescription, RTCConfiguration, RTCIceServer

logger = logging.getLogger("StreamingAPI")

class ConnectionManager:
    """Менеджер WebRTC соединений"""
    
    def __init__(self, config):
        self.config = config
        
        # Управление активными WebRTC соединениями
        self._active_connections: Dict[str, RTCPeerConnection] = {}
        self._connections_lock = threading.Lock()
        
        # Управление информацией о соединениях по task_id
        self._connection_info: Dict[str, Dict] = {}
        self._info_lock = threading.Lock()
        
    def get_connection(self, task_id: str) -> Optional[RTCPeerConnection]:
        with self._connections_lock:
            return self._active_connections.get(task_id)

    def get_connection_info(self, task_id: str) -> Optional[Dict]:
        with self._info_lock:
            return self._connection_info.get(task_id)
            
    def update_connection_info(self, task_id: str, info: Dict):
        with self._info_lock:
            if task_id not in self._connection_info:
                self._connection_info[task_id] = {}
            self._connection_info[task_id].update(info)

    async def create_connection(self, task_id: str):
        """Создает и настраивает RTCPeerConnection"""
        
        # Настройка ICE серверов
        ice_servers = []
        
        # 1. Всегда добавляем Google STUN серверы (как в оригинале)
        ice_servers.append(RTCIceServer(urls=["stun:stun.l.google.com:19302"]))
        ice_servers.append(RTCIceServer(urls=["stun:stun1.l.google.com:19302"]))
        ice_servers.append(RTCIceServer(urls=["stun:stun2.l.google.com:19302"]))
        ice_servers.append(RTCIceServer(urls=["stun:stun.relay.metered.ca:80"]))
        logger.info("Добавлены публичные STUN серверы Google")

        # 2. Добавляем Custom TURN сервер
        turn_servers_added = 0
        if self.config.turn_server_url:
            turn_urls = [self.config.turn_server_url]
            # Добавляем TCP вариант, если его нет (как в оригинале)
            if "?transport=" not in self.config.turn_server_url:
                turn_urls.append(f"{self.config.turn_server_url}?transport=tcp")
            
            turn_server = RTCIceServer(
                urls=turn_urls,
                username=self.config.turn_server_username,
                credential=self.config.turn_server_credential
            )
            ice_servers.append(turn_server)
            turn_servers_added += 1
            logger.info(f"Добавлен TURN сервер: {self.config.turn_server_url} (+TCP variant)")
        
        # 3. Добавляем публичные TURN серверы (Metered.ca) ТОЛЬКО если просили и нет своего
        # В оригинале логика: if use_public_turn and turn_servers_added == 0
        if self.config.use_public_turn or turn_servers_added == 0:
            logger.info(f"Используются публичные TURN серверы (Metered.ca)")
            # Добавляем ExpressTurn
            ice_servers.append(RTCIceServer(
                urls=["turn:free.expressturn.com:3478"],
                username="000000002083522617",
                credential="6tDY9S5LOEjq4QzVg0I4vdZv8hM="
            ))
            ice_servers.append(RTCIceServer(
                urls=["turn:standard.relay.metered.ca:80"],
                username="75a832015154748652ddb17e",
                credential="6Igq/7ph1nThXk4t"
            ))
            ice_servers.append(RTCIceServer(
                urls=["turn:standard.relay.metered.ca:80?transport=tcp"],
                username="75a832015154748652ddb17e",
                credential="6Igq/7ph1nThXk4t"
            ))
            ice_servers.append(RTCIceServer(
                urls=["turn:standard.relay.metered.ca:443"],
                username="75a832015154748652ddb17e",
                credential="6Igq/7ph1nThXk4t"
            ))
            ice_servers.append(RTCIceServer(
                urls=["turns:standard.relay.metered.ca:443?transport=tcp"],
                username="75a832015154748652ddb17e",
                credential="6Igq/7ph1nThXk4t"
            ))
            
            
            
        rtc_config = RTCConfiguration(iceServers=ice_servers)
        pc = RTCPeerConnection(configuration=rtc_config)
        
        with self._connections_lock:
            self._active_connections[task_id] = pc
            
        # Инициализируем info для таска
        self.update_connection_info(task_id, {"pc": pc})
            
        return pc

    async def close_connection(self, task_id: str):
        """Закрывает соединение и очищает ресурсы"""
        pc = None
        with self._connections_lock:
            if task_id in self._active_connections:
                pc = self._active_connections.pop(task_id)
        
        if pc:
            logger.info(f"Закрытие соединения для task_id={task_id}")
            try:
                await pc.close()
            except Exception as e:
                logger.error(f"Ошибка при закрытии соединения: {e}", exc_info=True)
                
    async def cleanup_all(self):
        """Закрывает все активные соединения"""
        tasks = []
        with self._connections_lock:
            for task_id in list(self._active_connections.keys()):
                tasks.append(self.close_connection(task_id))
        
        if tasks:
            await asyncio.gather(*tasks)
            
    def remove_audio_file(self, task_id: str):
        """Удаляет аудио файл, связанный с таском"""
        with self._info_lock:
            info = self._connection_info.get(task_id)
            if not info:
                return
                
            audio_path = info.get("audio_path")
            uploaded = info.get("uploaded_audio", False)
            
            if uploaded and audio_path and os.path.exists(audio_path):
                try:
                    os.remove(audio_path)
                    logger.info(f"Удален временный аудио файл: {audio_path}")
                except Exception as e:
                    logger.error(f"Ошибка при удалении аудио файла {audio_path}: {e}")
