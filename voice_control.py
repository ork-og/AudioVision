# voice_control.py
# Голосовой контроллер для PyQt5-приложений: оффлайн-распознавание (Vosk) + разбор команд.
# Установка: pip install vosk sounddevice
# Скачайте ru-модель Vosk и укажите путь через аргумент model_path или env VOSK_MODEL.

import os
import json
import threading
import queue
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import sounddevice as sd
from PyQt5 import QtCore

try:
    from vosk import Model, KaldiRecognizer
except Exception as e:
    Model = None
    KaldiRecognizer = None

@dataclass
class VoiceConfig:
    samplerate: int = 16000
    channels: int = 1
    dtype: str = "int16"
    blocksize: int = 8000  # ~0.5s при 16kHz (подберите при необходимости)
    model_path: Optional[str] = None  # если None — попробуем из VOSK_MODEL

class VoiceController(QtCore.QObject):
    """
    Отдельный объект/поток, слушает микрофон, распознаёт речь (Vosk),
    парсит ключевые слова и отправляет команды в основной поток через сигнал.
    """

    # cmd: str, payload: object (например, число/строка/путь)
    voiceCommand = QtCore.pyqtSignal(str, object)

    def __init__(self, config: VoiceConfig = VoiceConfig(), parent=None):
        super().__init__(parent)
        self.cfg = config
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._audio_q: queue.Queue[bytes] = queue.Queue(maxsize=8)
        self._recognizer = None
        self._model = None
        self._stream = None
        # 🔹 Укажи путь к папке модели Vosk (не к файлу, а к самой папке, где лежат am/, conf/ и т.д.)
        # Пример для macOS:
        # config.model_path = "/Users/arkadij/Library/Application Support/vosk/vosk-model-small-ru-0.22"
        config.model_path = "/Users/arkadij/Users/arkadij/Library/Application Support/vosk-model-ru-0.42"

    def is_ready(self) -> bool:
        return (Model is not None) and (KaldiRecognizer is not None)

    def start(self) -> bool:
        """Запускает поток распознавания. Возвращает True, если старт удачный."""
        if not self.is_ready():
            # Vosk не установлен
            return False

        model_path = self.cfg.model_path or os.environ.get("VOSK_MODEL")
        if not model_path or not os.path.isdir(model_path):
            return False

        try:
            self._model = Model(model_path)
            self._recognizer = KaldiRecognizer(self._model, self.cfg.samplerate)
        except Exception:
            return False

        self._stop.clear()
        self._thread = threading.Thread(target=self._run_loop, name="VoiceController", daemon=True)
        self._thread.start()
        return True

    def stop(self):
        """Останавливает поток и микрофонный стрим."""
        self._stop.set()
        try:
            if self._stream:
                self._stream.stop()
                self._stream.close()
        except Exception:
            pass
        self._stream = None
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None
        self._audio_q = queue.Queue(maxsize=8)

    # ---------- Внутреннее ----------
    def _audio_callback(self, indata, frames, time_info, status):
        if status:
            # можно логировать status, но не спамить
            pass
        try:
            self._audio_q.put_nowait(bytes(indata))
        except queue.Full:
            # если не успеваем — просто пропускаем блок
            pass

    def _run_loop(self):
        try:
            self._stream = sd.RawInputStream(
                samplerate=self.cfg.samplerate,
                channels=self.cfg.channels,
                dtype=self.cfg.dtype,
                blocksize=self.cfg.blocksize,
                callback=self._audio_callback,
            )
            self._stream.start()
        except Exception:
            # не удалось открыть микрофон
            return

        # Основной цикл
        while not self._stop.is_set():
            try:
                chunk = self._audio_q.get(timeout=0.3)
            except queue.Empty:
                continue

            try:
                if self._recognizer.AcceptWaveform(chunk):
                    res = self._recognizer.Result()
                else:
                    res = self._recognizer.PartialResult()
                self._handle_result(res)
            except Exception:
                # игнорируем разовые сбои
                pass

        # финальный drain
        try:
            res = self._recognizer.FinalResult()
            self._handle_result(res)
        except Exception:
            pass

    def _handle_result(self, res_json: str):
        """
        Разбираем JSON от Vosk:
        - у Result есть поле "text" (итоговая фраза),
        - у PartialResult поле "partial".
        Нас интересуют только законченные фразы (Result).
        """
        try:
            data = json.loads(res_json)
        except Exception:
            return

        text = None
        if "text" in data:
            text = (data.get("text") or "").strip()
        # Частичные не коммитим, чтобы не спамить
        if not text:
            return

        # Парсим команду
        cmd, payload = self._parse_command(text.lower())
        if cmd:
            # отправляем сигнал в GUI (потокобезопасно)
            self.voiceCommand.emit(cmd, payload)

    # ---------- Разбор фраз на команды ----------
    def _parse_command(self, text: str) -> Tuple[Optional[str], Optional[object]]:
        """
        Очень простой парсер фраз на русском.
        Примеры:
          - 'вставить картинку' / 'добавить картинку ...'
          - 'загрузить аудио' / 'добавить аудио ...'
          - 'пуск' / 'старт' / 'пауза' / 'остановить'
          - 'режим столбцы' / 'режим окружность'
          - 'сохранить' / 'экспорт'
          - 'громкость 70 процентов' / 'громче' / 'тише'
          - 'скорость один точка пять' / 'скорость 1.25'
        Можно расширять по месту.
        """
        # Картинка
        if "вставить картинку" in text or "добавить картинку" in text or "вставить изображение" in text:
            path = self._extract_path_after_keyword(text, ["картинку", "изображение"])
            return ("image", path or None)

        # Аудио
        if "загрузить аудио" in text or "добавить аудио" in text or "вставить аудио" in text:
            path = self._extract_path_after_keyword(text, ["аудио"])
            return ("audio", path or None)

        # Пуск/Пауза
        if any(k in text for k in ["пуск", "старт", "запусти", "запустить", "проигрывай", "проиграть"]):
            return ("play", None)
        if any(k in text for k in ["пауза", "останови", "остановить", "стоп"]):
            return ("pause", None)

        # Режим
        if "режим столбцы" in text or "столбцы" in text:
            return ("mode", "bars")
        if "режим окружность" in text or "режим круг" in text or "окружность" in text or "круг" in text:
            return ("mode", "ring")

        # Экспорт
        if any(k in text for k in ["сохранить", "экспорт", "рендер"]):
            return ("export", None)

        # Громкость
        if "громкость" in text or "проценто" in text:
            v = self._extract_percent(text)
            if v is not None:
                return ("volume", v)
        if "громче" in text:
            return ("volume_step", +10)
        if "тише" in text:
            return ("volume_step", -10)

        # Скорость
        if "скорость" in text:
            spd = self._extract_speed(text)
            if spd is not None:
                return ("speed", spd)

        return (None, None)

    def _extract_path_after_keyword(self, text: str, keywords):
        """
        Примитивная попытка достать путь после слова-ключа (если диктуют путь).
        В реальности пути лучше выбирать диалогом.
        """
        for kw in keywords:
            if kw in text:
                # всё после ключевого слова
                pos = text.find(kw) + len(kw)
                tail = text[pos:].strip().strip(":").strip()
                if tail:
                    return tail
        return None

    def _extract_percent(self, text: str) -> Optional[int]:
        # ищем число: '70', '70 процентов'
        import re
        m = re.search(r"(?:процент|процентов|процента)?\\s*(\\d{1,3})", text)
        if m:
            v = int(m.group(1))
            return max(0, min(100, v))
        return None

    def _extract_speed(self, text: str) -> Optional[float]:
        # ищем число с точкой/запятой: 1.25 / 1,25 / 'один точка пять' — упрощённо
        import re
        # сначала цифры
        m = re.search(r"(\\d+(?:[\\.,]\\d+)?)", text)
        if m:
            s = m.group(1).replace(",", ".")
            try:
                v = float(s)
                return max(0.25, min(3.0, v))
            except Exception:
                pass
        # простейшие слова
        if "один точка пять" in text:
            return 1.5
        if "полтора" in text:
            return 1.5
        return None
