#!/usr/bin/env python3
"""
Утилита для работы с memory-mapped файлами для передачи кадров между процессами.
Использует mmap для эффективной передачи кадров без сохранения на диск.
"""
import os
import mmap
import struct
import numpy as np
import json
import tempfile
import platform
from pathlib import Path
from typing import Optional, Tuple


# Структура заголовка mmap файла (все в байтах)
# Формат: <I - status (4 байта), <I - total_frames (4), <I - current_frame (4), 
#          <I - frame_width (4), <I - frame_height (4), <I - frame_channels (4),
#          <I - frame_size (4), <I - header_size (4)
HEADER_FORMAT = '<IIIIIIII'  # 8 * 4 = 32 байта
HEADER_SIZE = 32

# Статусы
STATUS_PROCESSING = 0
STATUS_COMPLETED = 1
STATUS_STOPPED = 2
STATUS_ERROR = 3


def get_mmap_dir():
    """Возвращает директорию для mmap файлов (использует /dev/shm на Linux, temp на Windows)"""
    if platform.system() == 'Linux' and os.path.exists('/dev/shm'):
        return '/dev/shm/musetalk_frames'
    else:
        # Используем системную временную директорию
        return os.path.join(tempfile.gettempdir(), 'musetalk_frames')


def get_mmap_path(task_id: str) -> str:
    """Возвращает путь к mmap файлу для task_id"""
    mmap_dir = get_mmap_dir()
    os.makedirs(mmap_dir, exist_ok=True)
    return os.path.join(mmap_dir, f"frames_{task_id}.mmap")


class MmapFrameWriter:
    """Класс для записи кадров в mmap файл"""
    
    def __init__(self, task_id: str, total_frames: int, frame_shape: Tuple[int, int, int], 
                 frame_dtype=np.uint8):
        """
        Инициализирует mmap файл для записи кадров.
        
        Args:
            task_id: ID задачи
            total_frames: Общее количество кадров
            frame_shape: Форма кадра (height, width, channels)
            frame_dtype: Тип данных кадра (по умолчанию uint8)
        """
        self.task_id = task_id
        self.total_frames = total_frames
        self.frame_shape = frame_shape
        self.frame_dtype = frame_dtype
        self.frame_height, self.frame_width, self.frame_channels = frame_shape
        
        # Размер одного кадра в байтах
        self.frame_size = int(np.prod(frame_shape) * np.dtype(frame_dtype).itemsize)
        
        # Размер всего файла: заголовок + данные кадров
        self.file_size = HEADER_SIZE + (self.frame_size * total_frames)
        
        # Путь к mmap файлу
        self.mmap_path = get_mmap_path(task_id)
        print(f"[MmapFrameWriter] Создание mmap файла для task_id={task_id}: {self.mmap_path}", flush=True)
        
        # Создаем файл нужного размера
        try:
            with open(self.mmap_path, 'wb') as f:
                f.write(b'\x00' * self.file_size)
            print(f"[MmapFrameWriter] Файл создан, размер: {self.file_size} байт", flush=True)
        except Exception as e:
            print(f"[MmapFrameWriter] ОШИБКА при создании файла: {e}", flush=True)
            raise
        
        # Открываем mmap для записи
        try:
            self.file = open(self.mmap_path, 'r+b')
            self.mmap = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_WRITE)
            print(f"[MmapFrameWriter] Mmap открыт для записи", flush=True)
        except Exception as e:
            print(f"[MmapFrameWriter] ОШИБКА при открытии mmap: {e}", flush=True)
            raise
        
        # Инициализируем заголовок
        self._write_header(STATUS_PROCESSING, 0)
        
        print(f"[MmapFrameWriter] Создан mmap файл: {self.mmap_path} (размер: {self.file_size} байт)", flush=True)
        print(f"[MmapFrameWriter] Кадр: {frame_shape}, размер кадра: {self.frame_size} байт", flush=True)
    
    def _write_header(self, status: int, current_frame: int):
        """Записывает заголовок в mmap"""
        if self.mmap is None or self.mmap.closed:
            raise ValueError("Mmap закрыт, невозможно записать заголовок")
        
        try:
            header = struct.pack(
                HEADER_FORMAT,
                status,                    # status
                self.total_frames,         # total_frames
                current_frame,              # current_frame
                self.frame_width,          # frame_width
                self.frame_height,         # frame_height
                self.frame_channels,       # frame_channels
                self.frame_size,           # frame_size
                HEADER_SIZE                # header_size
            )
            # Используем срез для записи вместо seek/write
            self.mmap[0:HEADER_SIZE] = header
            self.mmap.flush()
        except (ValueError, OSError, IndexError) as e:
            raise ValueError(f"Ошибка при записи заголовка: {e}")
    
    def write_frame(self, frame_index: int, frame_array: np.ndarray):
        """
        Записывает кадр в mmap файл.
        
        Args:
            frame_index: Индекс кадра (0-based)
            frame_array: numpy массив кадра (BGR формат)
        """
        if frame_index >= self.total_frames:
            raise ValueError(f"Индекс кадра {frame_index} превышает общее количество {self.total_frames}")
        
        # Проверяем форму кадра
        if frame_array.shape != self.frame_shape:
            raise ValueError(f"Неверная форма кадра: ожидается {self.frame_shape}, получено {frame_array.shape}")
        
        # Проверяем тип данных
        if frame_array.dtype != self.frame_dtype:
            frame_array = frame_array.astype(self.frame_dtype)
        
        # Вычисляем смещение для этого кадра
        offset = HEADER_SIZE + (frame_index * self.frame_size)
        
        # Проверяем границы
        if offset + self.frame_size > len(self.mmap):
            raise ValueError(f"Смещение {offset} + размер {self.frame_size} превышает размер mmap {len(self.mmap)}")
        
        # Проверяем, что mmap открыт
        if self.mmap is None or self.mmap.closed:
            raise ValueError("Mmap закрыт, невозможно записать кадр")
        
        try:
            # Записываем кадр используя срез для безопасности
            frame_bytes = frame_array.tobytes()
            self.mmap[offset:offset + self.frame_size] = frame_bytes
            self.mmap.flush()
            
            # Обновляем заголовок с текущим индексом
            self._write_header(STATUS_PROCESSING, frame_index + 1)
        except (ValueError, OSError, IndexError) as e:
            raise ValueError(f"Ошибка при записи кадра {frame_index}: {e}")
    
    def set_completed(self):
        """Устанавливает статус завершения"""
        self._write_header(STATUS_COMPLETED, self.total_frames)
        print(f"[MmapFrameWriter] Генерация завершена: {self.total_frames} кадров")
    
    def set_stopped(self, frames_generated: int):
        """Устанавливает статус остановки"""
        self._write_header(STATUS_STOPPED, frames_generated)
        print(f"[MmapFrameWriter] Генерация остановлена: {frames_generated} кадров")
    
    def set_error(self, error_message: str):
        """Устанавливает статус ошибки"""
        self._write_header(STATUS_ERROR, 0)
        print(f"[MmapFrameWriter] Ошибка: {error_message}")
    
    def close(self):
        """Закрывает mmap файл"""
        try:
            if self.mmap and not self.mmap.closed:
                self.mmap.close()
        except Exception as e:
            print(f"[MmapFrameWriter] Ошибка при закрытии mmap: {e}", flush=True)
        finally:
            try:
                if self.file:
                    self.file.close()
            except Exception as e:
                print(f"[MmapFrameWriter] Ошибка при закрытии файла: {e}", flush=True)
            self.mmap = None
            self.file = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


class MmapFrameReader:
    """Класс для чтения кадров из mmap файла"""
    
    def __init__(self, task_id: str):
        """
        Инициализирует чтение из mmap файла.
        
        Args:
            task_id: ID задачи
        """
        self.task_id = task_id
        self.mmap_path = get_mmap_path(task_id)
        self.file = None
        self.mmap = None
        self.header = None
        
        # Параметры из заголовка
        self.status = None
        self.total_frames = None
        self.current_frame = None
        self.frame_width = None
        self.frame_height = None
        self.frame_channels = None
        self.frame_size = None
        
        # Открываем mmap для чтения
        self._open()
    
    def _open(self):
        """Открывает mmap файл для чтения"""
        if not os.path.exists(self.mmap_path):
            raise FileNotFoundError(f"Mmap файл не найден: {self.mmap_path}")
        
        # Проверяем, что файл не пустой
        file_size = os.path.getsize(self.mmap_path)
        if file_size == 0:
            raise ValueError(f"Mmap файл пустой (0 байт): {self.mmap_path}. Демон еще не создал файл с данными.")
        
        if file_size < HEADER_SIZE:
            raise ValueError(f"Mmap файл слишком мал ({file_size} байт < {HEADER_SIZE} байт заголовка): {self.mmap_path}")
        
        self.file = open(self.mmap_path, 'rb')
        self.mmap = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_READ)
        
        # Читаем заголовок
        self._read_header()
    
    def _read_header(self):
        """Читает заголовок из mmap"""
        if self.mmap is None or self.mmap.closed:
            raise ValueError("Mmap закрыт")
        
        try:
            # Используем срез вместо seek/read для безопасности
            if len(self.mmap) < HEADER_SIZE:
                raise ValueError(f"Mmap слишком мал: {len(self.mmap)} < {HEADER_SIZE}")
            
            header_data = self.mmap[0:HEADER_SIZE]
            
            if len(header_data) < HEADER_SIZE:
                raise ValueError(f"Неверный размер заголовка: {len(header_data)} < {HEADER_SIZE}")
            
            self.header = struct.unpack(HEADER_FORMAT, header_data)
            self.status = self.header[0]
            self.total_frames = self.header[1]
            self.current_frame = self.header[2]
            self.frame_width = self.header[3]
            self.frame_height = self.header[4]
            self.frame_channels = self.header[5]
            self.frame_size = self.header[6]
            header_size = self.header[7]
            
            if header_size != HEADER_SIZE:
                raise ValueError(f"Неверный размер заголовка в файле: {header_size} != {HEADER_SIZE}")
        except (ValueError, OSError, IndexError) as e:
            raise ValueError(f"Ошибка при чтении заголовка: {e}")
    
    def get_status(self) -> dict:
        """Возвращает статус генерации"""
        if self.mmap:
            self._read_header()  # Обновляем заголовок
        
        status_map = {
            STATUS_PROCESSING: "processing",
            STATUS_COMPLETED: "completed",
            STATUS_STOPPED: "stopped",
            STATUS_ERROR: "error"
        }
        
        return {
            "status": status_map.get(self.status, "unknown"),
            "frames_generated": self.current_frame,
            "total_frames": self.total_frames
        }
    
    def read_frame(self, frame_index: int) -> Optional[np.ndarray]:
        """
        Читает кадр из mmap файла.
        
        Args:
            frame_index: Индекс кадра (0-based)
        
        Returns:
            numpy массив кадра или None, если кадр еще не готов
        """
        if frame_index >= self.total_frames:
            return None
        
        # Проверяем, что mmap открыт
        if self.mmap is None or self.mmap.closed:
            return None
        
        # Обновляем заголовок для проверки текущего состояния
        try:
            self._read_header()
        except (ValueError, OSError) as e:
            # Mmap закрыт или поврежден
            print(f"[MmapFrameReader] Ошибка при чтении заголовка: {e}", flush=True)
            return None
        
        # Проверяем, готов ли кадр
        if frame_index >= self.current_frame:
            return None
        
        # Вычисляем смещение для этого кадра
        offset = HEADER_SIZE + (frame_index * self.frame_size)
        
        # Проверяем границы
        if offset + self.frame_size > len(self.mmap):
            return None
        
        try:
            # Читаем данные кадра напрямую из mmap (без seek/read)
            # Используем срез для безопасного доступа
            frame_data = self.mmap[offset:offset + self.frame_size]
            
            if len(frame_data) < self.frame_size:
                return None
            
            # Преобразуем в numpy array - используем memoryview для безопасности
            frame_array = np.frombuffer(memoryview(frame_data), dtype=np.uint8)
            frame_array = frame_array.reshape((self.frame_height, self.frame_width, self.frame_channels))
            
            return frame_array.copy()  # Возвращаем копию для безопасности
        except (ValueError, OSError, IndexError) as e:
            print(f"[MmapFrameReader] Ошибка при чтении кадра {frame_index}: {e}", flush=True)
            return None
    
    def close(self):
        """Закрывает mmap файл"""
        try:
            if self.mmap and not self.mmap.closed:
                self.mmap.close()
        except Exception as e:
            print(f"[MmapFrameReader] Ошибка при закрытии mmap: {e}", flush=True)
        finally:
            try:
                if self.file:
                    self.file.close()
            except Exception as e:
                print(f"[MmapFrameReader] Ошибка при закрытии файла: {e}", flush=True)
            self.mmap = None
            self.file = None
    
    def cleanup(self):
        """Удаляет mmap файл"""
        self.close()
        if os.path.exists(self.mmap_path):
            try:
                os.remove(self.mmap_path)
                print(f"[MmapFrameReader] Удален mmap файл: {self.mmap_path}")
            except Exception as e:
                print(f"[MmapFrameReader] Ошибка при удалении mmap файла: {e}")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

