import sys
import os
import math
import numpy as np
import cv2
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtCore import QUrl
import shutil, subprocess, tempfile, platform
from PyQt5.QtCore import QSettings

# ---------------------------
# УТИЛИТЫ ДЛЯ АУДИО (WAV)
# ---------------------------
import wave


def read_wav_mono(path):
    """
    Читает WAV-файл и возвращает (audio_float_mono, sample_rate).
    Поддерживается PCM 8/16/24/32-bit и float32/float64.
    """
    with wave.open(path, 'rb') as wf:
        n_channels = wf.getnchannels()
        sr = wf.getframerate()
        n_frames = wf.getnframes()
        sampwidth = wf.getsampwidth()
        raw = wf.readframes(n_frames)

    # Преобразуем в numpy
    if sampwidth == 1:
        dtype = np.uint8  # 8-bit unsigned PCM
        data = np.frombuffer(raw, dtype=dtype).astype(np.float32)
        data = (data - 128.0) / 128.0
    elif sampwidth == 2:
        dtype = np.int16
        data = np.frombuffer(raw, dtype=dtype).astype(np.float32) / 32768.0
    elif sampwidth == 3:
        # 24-bit PCM: конвертируем вручную в int32 с выравниванием знака
        a = np.frombuffer(raw, dtype=np.uint8)
        a = a.reshape(-1, 3)
        b = (a[:, 0].astype(np.int32) | (a[:, 1].astype(np.int32) << 8) | (a[:, 2].astype(np.int32) << 16))
        # преобразуем знаковость 24-бит
        mask = b & 0x800000
        b = b - (mask << 1)
        data = b.astype(np.float32) / 8388608.0
    elif sampwidth == 4:
        # может быть int32 PCM или float32
        arr = np.frombuffer(raw, dtype=np.int32)
        if np.max(np.abs(arr)) > 1e8:
            data = np.frombuffer(raw, dtype=np.float32)
        else:
            data = arr.astype(np.float32) / 2147483648.0
    else:
        data = np.frombuffer(raw, dtype=np.float64).astype(np.float32)

    if n_channels > 1:
        data = data.reshape(-1, n_channels)
        data = data.mean(axis=1)

    #ghfh

    return data.astype(np.float32), sr


def make_bar_features(audio, sr, fps, n_bins=32, ref_median_frames=60):
    """
    Разбивает аудио по окнам длительностью 1/FPS и считает энергию по частотным полосам.
    Возвращает массив формы (n_video_frames, n_bins) с нормированными значениями 0..1.
    """
    eps = 1e-8
    samples_per_frame = max(1, int(round(sr / float(fps))))
    n_frames = int(math.ceil(len(audio) / samples_per_frame))

    # Окно Ханна
    hann = np.hanning(samples_per_frame).astype(np.float32)

    # Частоты для rFFT
    freqs = np.fft.rfftfreq(samples_per_frame, d=1.0 / sr)

    # Логарифмически распределённые границы полос
    f_min = 20.0
    f_max = min(sr / 2.0, 16000.0)
    edges = np.geomspace(f_min, f_max, n_bins + 1)

    # Предподсчёт индексов для каждой полосы
    band_indices = []
    for i in range(n_bins):
        f1, f2 = edges[i], edges[i + 1]
        idx = np.where((freqs >= f1) & (freqs < f2))[0]
        if len(idx) == 0:
            nearest = np.argmin(np.abs(freqs - (f1 + f2) * 0.5))
            idx = np.array([nearest])
        band_indices.append(idx)

    bars = np.zeros((n_frames, n_bins), dtype=np.float32)

    # Подсчёт спектральной энергии по кадрам
    for fi in range(n_frames):
        s = fi * samples_per_frame
        e = min(len(audio), s + samples_per_frame)
        frame = np.zeros(samples_per_frame, dtype=np.float32)
        seg = audio[s:e]
        frame[: len(seg)] = seg
        frame *= hann
        mag = np.abs(np.fft.rfft(frame))

        for b, idx in enumerate(band_indices):
            val = mag[idx].mean()
            bars[fi, b] = val

    # Лог-нормализация и адаптация к уровню
    bars = np.log1p(bars)
    ref = np.median(bars[: min(ref_median_frames, len(bars))], axis=0) + eps
    bars = (bars - ref[None, :])
    bars = np.clip(bars, 0.0, None)
    if np.max(bars) > eps:
        bars /= (np.max(bars) + eps)

    # Временное сглаживание (пик-холд)
    alpha = 0.35
    for b in range(n_bins):
        acc = 0.0
        for fi in range(n_frames):
            acc = alpha * bars[fi, b] + (1 - alpha) * acc
            bars[fi, b] = max(bars[fi, b], acc)

    return bars


# ---------------------------
# ГЛАВНОЕ ОКНО ПРИЛОЖЕНИЯ
# ---------------------------
class VideoAudioVisualizer(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Видеоплеер + аудио-визуализация (PyQt5)")
        self.resize(1100, 700)

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)

        self.video_label = QtWidgets.QLabel("Загрузите видео/картинку и аудио…")
        self.video_label.setAlignment(QtCore.Qt.AlignCenter)
        self.video_label.setStyleSheet("background:#111; color:#aaa; font-size:16px;")
        self.video_label.setMinimumSize(800, 450)

        self.btn_load_video = QtWidgets.QPushButton("Загрузить видео…")
        self.btn_load_image = QtWidgets.QPushButton("Загрузить картинку…")
        self.btn_load_audio = QtWidgets.QPushButton("Загрузить аудио (WAV)…")
        self.btn_play = QtWidgets.QPushButton("▶️ Пуск")
        self.btn_play.setCheckable(True)
        self.btn_play.setEnabled(False)
        self.btn_export = QtWidgets.QPushButton("Сохранить MP4…")

        self.combo_vis = QtWidgets.QComboBox()
        self.combo_vis.addItems(["Столбцы", "Пульсирующая окружность"])  # bars / ring

        # Кнопка выбора цвета визуализации
        self.btn_color = QtWidgets.QPushButton("Цвет…")
        self.vis_color = QtGui.QColor(255, 255, 255)
        self._apply_btn_color_style()

        self.slider_speed = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_speed.setRange(50, 200)
        self.slider_speed.setValue(100)
        self.lbl_speed = QtWidgets.QLabel("Скорость: 1.00x")

        self.slider_volume = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_volume.setRange(0, 100)
        self.slider_volume.setValue(80)
        self.lbl_volume = QtWidgets.QLabel("Громкость: 80%")

        controls = QtWidgets.QHBoxLayout()
        controls.addWidget(self.btn_load_video)
        controls.addWidget(self.btn_load_image)
        controls.addWidget(self.btn_load_audio)
        controls.addWidget(self.btn_play)
        controls.addWidget(self.btn_export)
        controls.addWidget(self.btn_color)
        controls.addStretch(1)
        controls.addWidget(QtWidgets.QLabel("Визуализация:"))
        controls.addWidget(self.combo_vis)
        controls.addSpacing(20)
        controls.addWidget(self.lbl_speed)
        controls.addWidget(self.slider_speed)
        controls.addSpacing(12)
        controls.addWidget(self.lbl_volume)
        controls.addWidget(self.slider_volume)

        layout = QtWidgets.QVBoxLayout(central)
        layout.addWidget(self.video_label, 1)
        layout.addLayout(controls)

        # Состояние
        self.cap = None
        self.video_path = None
        self.image_path = None
        self.audio_path = None
        self.fps = 30.0
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.next_frame)
        self.frame_index = 0
        self.n_video_frames = 0

        self.bars = None
        self.n_bins = 32

        self.player = None  # QMediaPlayer

        # Хранилище для статичной картинки (фон)
        self.still_image_bgr = None  # np.ndarray HxWx3 uint8

        # Сигналы
        self.btn_load_video.clicked.connect(self.load_video)
        self.btn_load_image.clicked.connect(self.load_image)
        self.btn_load_audio.clicked.connect(self.load_audio)
        self.btn_play.toggled.connect(self.toggle_play)
        self.slider_speed.valueChanged.connect(self.on_speed_changed)
        self.slider_volume.valueChanged.connect(self.on_volume_changed)
        self.combo_vis.currentIndexChanged.connect(self.on_vis_changed)
        self.btn_export.clicked.connect(self.export_mp4)
        self.btn_color.clicked.connect(self.choose_vis_color)

    def _settings(self) -> QSettings:
        return QSettings("YourOrg", "VideoAudioVisualizer")

    def _candidate_ffmpeg_paths(self):
        # Популярные места на macOS + Linux + Windows
        candidates = [
            "/opt/homebrew/bin/ffmpeg",  # Homebrew (Apple Silicon)
            "/usr/local/bin/ffmpeg",  # Homebrew (Intel)
            "/opt/local/bin/ffmpeg",  # MacPorts
            "/usr/bin/ffmpeg",  # Linux
            "C:/ffmpeg/bin/ffmpeg.exe",  # Windows (часто)
        ]
        return candidates

    def find_ffmpeg(self) -> str:
        """Пытается найти ffmpeg: 1) сохранённый путь из QSettings, 2) в PATH, 3) по типичным путям."""
        # 1) Сохранённый путь пользователя
        s = self._settings()
        saved = s.value("ffmpeg_path", type=str)
        if saved and os.path.isfile(saved) and os.access(saved, os.X_OK):
            return saved

        # 2) В текущем PATH
        p = shutil.which("ffmpeg")
        if p:
            return p

        # 3) Популярные места
        for cand in self._candidate_ffmpeg_paths():
            if os.path.isfile(cand) and os.access(cand, os.X_OK):
                return cand

        # 4) Попробуем системную `which ffmpeg`
        try:
            out = subprocess.run(["/usr/bin/which", "ffmpeg"], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                 check=False)
            cand = out.stdout.decode().strip()
            if cand and os.path.isfile(cand) and os.access(cand, os.X_OK):
                return cand
        except Exception:
            pass

        return None

    def ask_ffmpeg_path(self) -> str:
        """Просит пользователя указать ffmpeg вручную и сохраняет в QSettings."""
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Укажите бинарник ffmpeg", "", "Все файлы (*)")
        if not path:
            return None
        if not os.path.isfile(path):
            QtWidgets.QMessageBox.warning(self, "ffmpeg", "Указан несуществующий файл.")
            return None
        if not os.access(path, os.X_OK):
            # попробуем дать права на исполнение (mac/Linux)
            try:
                os.chmod(path, os.stat(path).st_mode | 0o111)
            except Exception:
                pass
            if not os.access(path, os.X_OK):
                QtWidgets.QMessageBox.warning(self, "ffmpeg",
                                              "Файл не исполняемый. Сделайте его исполняемым или выберите другой.")
                return None
        s = self._settings()
        s.setValue("ffmpeg_path", path)
        return path

    # ---------- Конвертеры/утилиты ----------
    def qimage_to_bgr(self, qimg):
        """QImage (Format_RGB888) -> NumPy BGR uint8"""
        w = qimg.width()
        h = qimg.height()
        ptr = qimg.bits()
        ptr.setsize(h * w * 3)
        arr = np.frombuffer(ptr, np.uint8).reshape((h, w, 3))
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

    def render_frame_with_overlay(self, frame_bgr, idx):
        """Вернёт кадр BGR с наложенной визуализацией (без масштабирования)."""
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w, _ = frame_rgb.shape
        qimg = QtGui.QImage(frame_rgb.data, w, h, 3 * w, QtGui.QImage.Format_RGB888).copy()
        painter = QtGui.QPainter(qimg)
        if self.bars is not None and self.bars.size > 0:
            if idx >= len(self.bars):
                idx = idx % len(self.bars)
            vals = self.bars[idx]
            mode = self.combo_vis.currentText()
            if mode == "Столбцы":
                self.draw_bars(painter, w, h, vals)
            else:
                self.draw_circle(painter, w, h, vals)
        painter.end()
        return self.qimage_to_bgr(qimg)

    # ---------- Цвет ---------
    def _with_alpha(self, color: QtGui.QColor, alpha: int) -> QtGui.QColor:
        c = QtGui.QColor(color)
        c.setAlpha(max(0, min(255, int(alpha))))
        return c

    def _apply_btn_color_style(self):
        # мини-превью цвета на кнопке
        c = self.vis_color
        self.btn_color.setFixedWidth(90)
        self.btn_color.setStyleSheet(
            f"QPushButton{{padding:6px 10px; border-radius:6px; border:1px solid #444;"
            f"background-color: rgba({c.red()},{c.green()},{c.blue()},255); color: #000;}}"
            f"QPushButton:hover{{filter: brightness(1.08);}}"
        )

    def choose_vis_color(self):
        c = QtWidgets.QColorDialog.getColor(self.vis_color, self, "Выберите цвет визуализации")
        if c.isValid():
            self.vis_color = c
            self._apply_btn_color_style()
            self.on_vis_changed(0)

    # ---------- Отрисовка ----------
    def draw_bars(self, painter, w, h, vals):
        # тёмная подложка снизу для читаемости (не зависит от цвета визуализации)
        grad = QtGui.QLinearGradient(0, int(h * 0.6), 0, h)
        grad.setColorAt(0.0, self._with_alpha(QtGui.QColor(0, 0, 0), 0))
        grad.setColorAt(1.0, self._with_alpha(QtGui.QColor(0, 0, 0), 140))
        painter.fillRect(0, int(h * 0.6), w, int(h * 0.4), QtGui.QBrush(grad))

        margin_lr = int(0.05 * w)
        margin_bottom = int(0.06 * h)
        area_w = w - 2 * margin_lr
        area_h = int(0.30 * h)
        base_y = h - margin_bottom
        n = len(vals)
        gap = max(1, int(area_w * 0.002))
        bar_w = max(2, int((area_w - gap * (n - 1)) / n))

        pen = QtGui.QPen(self._with_alpha(self.vis_color, 220))
        brush = QtGui.QBrush(self._with_alpha(self.vis_color, 200))
        painter.setPen(pen)
        painter.setBrush(brush)

        x = margin_lr
        for v in vals:
            h_pix = int(v * area_h)
            rect = QtCore.QRect(x, base_y - h_pix, bar_w, h_pix)
            painter.drawRect(rect)
            x += bar_w + gap

        painter.setPen(QtGui.QPen(self._with_alpha(self.vis_color, 80), 1))
        painter.drawLine(margin_lr, base_y, margin_lr + area_w, base_y)

    def draw_circle(self, painter, w, h, vals):
        cx, cy = w // 2, int(h * 0.53)
        r_inner = int(0.12 * h)
        max_len = int(0.16 * h)

        #jjj

        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)

        # Базовое тонкое кольцо
        base_pen = QtGui.QPen(self._with_alpha(self.vis_color, 70))
        base_pen.setWidth(2)
        painter.setPen(base_pen)
        painter.setBrush(QtCore.Qt.NoBrush)
        painter.drawEllipse(QtCore.QPoint(cx, cy), r_inner, r_inner)

        # Радиальные лучи
        n = len(vals)
        for i, v in enumerate(vals):
            left = vals[i - 1] if i > 0 else vals[-1]
            right = vals[(i + 1) % n]
            vv = (0.6 * v + 0.2 * left + 0.2 * right)
            L = int(vv * max_len * 0.7)

            angle = (2.0 * math.pi) * (i / n)
            ca, sa = math.cos(angle), math.sin(angle)
            x1 = cx + int(ca * r_inner)
            y1 = cy + int(sa * r_inner)
            x2 = cx + int(ca * (r_inner + L))
            y2 = cy + int(sa * (r_inner + L))

            pen = QtGui.QPen(self._with_alpha(self.vis_color, 180))
            pen.setWidth(2)
            painter.setPen(pen)
            painter.drawLine(x1, y1, x2, y2)

        # «Ушки» НЧ/ВЧ — дополнительные дуги по энергии краёв спектра
        low_edge = max(1, int(0.15 * n))
        high_edge = max(1, int(0.15 * n))
        low_energy = float(np.mean(vals[:low_edge])) if low_edge < n else float(np.mean(vals))
        high_energy = float(np.mean(vals[n - high_edge:])) if high_edge < n else float(np.mean(vals))

        ears = [
            {"center": 0.0,           "amp": low_energy,  "alpha": 230},
            {"center": math.pi,       "amp": low_energy,  "alpha": 230},
            {"center": math.pi * 0.5, "amp": high_energy, "alpha": 210},
            {"center": math.pi * 1.5, "amp": high_energy, "alpha": 210},
        ]
        ear_half = math.radians(26)
        ear_max  = int(max_len * 0.9)

        for e in ears:
            c = e["center"]
            amp = e["amp"]
            steps = 36
            for k in range(-steps, steps + 1):
                t = k / steps
                ang = c + t * ear_half
                window = 0.5 * (1 + math.cos(math.pi * t))
                L = int(amp * ear_max * (window ** 1.2))
                ca, sa = math.cos(ang), math.sin(ang)
                x1 = cx + int(ca * r_inner)
                y1 = cy + int(sa * r_inner)
                x2 = cx + int(ca * (r_inner + L))
                y2 = cy + int(sa * (r_inner + L))

                pen = QtGui.QPen(self._with_alpha(self.vis_color, e["alpha"]))
                pen.setWidth(4)
                painter.setPen(pen)
                painter.drawLine(x1, y1, x2, y2)

        # Мягкое внешнее свечение
        glow_pen = QtGui.QPen(self._with_alpha(self.vis_color, 50))
        glow_pen.setWidth(6)
        painter.setPen(glow_pen)
        painter.drawEllipse(QtCore.QPoint(cx, cy), r_inner + int(0.5 * max_len), r_inner + int(0.5 * max_len))

    # ---------- Основной цикл отрисовки ----------
    def next_frame(self):
        # Источник кадра: видео (если открыто) или статичная картинка
        if self.cap is not None:
            ret, frame_bgr = self.cap.read()
            if not ret:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.frame_index = 0
                ret, frame_bgr = self.cap.read()
                if not ret:
                    return
        elif self.still_image_bgr is not None:
            frame_bgr = self.still_image_bgr.copy()
        else:
            return

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w, _ = frame_rgb.shape
        qimg = QtGui.QImage(frame_rgb.data, w, h, 3 * w, QtGui.QImage.Format_RGB888).copy()

        painter = QtGui.QPainter(qimg)
        if self.bars is not None and self.bars.size > 0:
            if self.player is not None:
                pos_ms = self.player.position()
                idx = int((pos_ms / 1000.0) * self.fps)
            else:
                idx = self.frame_index
            if idx >= len(self.bars):
                idx = idx % len(self.bars)
            vals = self.bars[idx]

            mode = self.combo_vis.currentText()
            if mode == "Столбцы":
                self.draw_bars(painter, w, h, vals)
            else:
                self.draw_circle(painter, w, h, vals)
        painter.end()

        pix = QtGui.QPixmap.fromImage(qimg)
        pix = pix.scaled(self.video_label.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        self.video_label.setPixmap(pix)
        self.frame_index += 1

    # ---------- Загрузка видео ----------
    def load_video(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Выберите видеофайл", "", "Видео (*.mp4 *.avi *.mkv *.mov *.webm);;Все файлы (*.*)")
        if not path:
            return
        if self.cap is not None:
            self.cap.release()
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            QtWidgets.QMessageBox.critical(self, "Ошибка", "Не удалось открыть видео.")
            self.cap = None
            return
        self.video_path = path
        self.image_path = None
        self.still_image_bgr = None
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.fps = fps if fps > 1e-3 else 30.0
        self.n_video_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_index = 0
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        self.update_window_title()
        self.update_play_button_state()
        self.draw_placeholder()

    # ---------- Загрузка КАРТИНКИ ----------
    def load_image(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Выберите изображение",
            "",
            "Изображения (*.png *.jpg *.jpeg *.bmp *.webp);;Все файлы (*.*)"
        )
        if not path:
            return

        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            QtWidgets.QMessageBox.critical(self, "Ошибка", "Не удалось открыть изображение.")
            return

        # Если загружаем картинку — сбрасываем видео-кап
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
            self.n_video_frames = 0
            self.video_path = None

        self.still_image_bgr = img
        self.image_path = path
        self.frame_index = 0
        # fps оставляем как есть (по умолчанию 30), либо пользователь может поменять слайдером скорости

        # Отрисуем превью
        self.on_vis_changed(0)
        self.update_window_title()
        self.update_play_button_state()

    # ---------- Загрузка аудио ----------
    def load_audio(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Выберите аудиофайл (WAV)", "", "Аудио WAV (*.wav);;Все файлы (*.*)")
        if not path:
            return
        try:
            audio, sr = read_wav_mono(path)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Ошибка аудио", f"Не удалось прочитать WAV.\n{e}")
            return

        fps = self.fps if (self.cap is not None or self.still_image_bgr is not None) else 30.0
        self.bars = make_bar_features(audio, sr, fps=fps, n_bins=self.n_bins)
        self.audio_path = path

        # Инициализируем и настраиваем аудиоплеер
        if self.player is None:
            self.player = QMediaPlayer(self)
            self.player.mediaStatusChanged.connect(self.on_media_status)
        self.player.setMedia(QMediaContent(QUrl.fromLocalFile(path)))
        self.player.setVolume(self.slider_volume.value())
        self.player.setPlaybackRate(self.slider_speed.value() / 100.0)

        self.update_window_title()
        self.update_play_button_state()

    def update_window_title(self):
        vname = os.path.basename(self.video_path) if self.video_path else (os.path.basename(self.image_path) if self.image_path else "(нет видео/картинки)")
        aname = os.path.basename(self.audio_path) if self.audio_path else "(нет аудио)"
        self.setWindowTitle(f"Фон: {vname}  |  Аудио: {aname}  |  FPS: {self.fps:.2f}")

    def update_play_button_state(self):
        # Можно играть, если есть видео ИЛИ картинка
        self.btn_play.setEnabled((self.cap is not None) or (self.still_image_bgr is not None))

    def on_speed_changed(self, val):
        spd = val / 100.0
        self.lbl_speed.setText(f"Скорость: {spd:.2f}x")
        if self.timer.isActive():
            interval_ms = max(1, int(1000.0 / (self.fps * spd)))
            self.timer.setInterval(interval_ms)
        if self.player is not None:
            self.player.setPlaybackRate(spd)

    def toggle_play(self, checked):
        if checked:
            if (self.cap is None) and (self.still_image_bgr is None):
                QtWidgets.QMessageBox.information(self, "Нет источника", "Сначала загрузите видео или картинку.")
                self.btn_play.setChecked(False)
                return
            spd = self.slider_speed.value() / 100.0
            interval_ms = max(1, int(1000.0 / (self.fps * spd)))
            self.timer.start(interval_ms)
            if self.player is not None:
                self.player.setPlaybackRate(spd)
                self.player.play()
            self.btn_play.setText("⏸ Пауза")
        else:
            self.timer.stop()
            if self.player is not None:
                self.player.pause()
            self.btn_play.setText("▶️ Пуск")

    def draw_placeholder(self):
        w = max(800, self.video_label.width())
        h = max(450, self.video_label.height())
        img = QtGui.QImage(w, h, QtGui.QImage.Format_RGB32)
        img.fill(QtGui.QColor(17, 17, 17))
        painter = QtGui.QPainter(img)
        painter.setPen(QtGui.QColor(200, 200, 200))
        painter.setFont(QtGui.QFont("Arial", 16))
        msg = "Нажмите ▶️ для воспроизведения"
        rect = QtCore.QRect(0, 0, w, h)
        painter.drawText(rect, QtCore.Qt.AlignCenter, msg)
        painter.end()
        self.video_label.setPixmap(QtGui.QPixmap.fromImage(img))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.video_label.pixmap() is not None:
            self.video_label.setPixmap(self.video_label.pixmap().scaled(self.video_label.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))

    def on_volume_changed(self, val):
        self.lbl_volume.setText(f"Громкость: {val}%")
        if self.player is not None:
            self.player.setVolume(val)

    def on_media_status(self, status):
        if status == QMediaPlayer.EndOfMedia:
            self.player.setPosition(0)
            if self.btn_play.isChecked():
                self.player.play()

    def on_vis_changed(self, idx):
        if self.timer.isActive():
            return
        # перерисуем текущий кадр для предпросмотра
        if self.cap is not None:
            cur = max(0, self.frame_index - 1)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, cur)
            prev_index = self.frame_index
            self.next_frame()
            self.frame_index = prev_index
        elif self.still_image_bgr is not None:
            # Сгенерируем один кадр поверх картинки
            frame_bgr = self.still_image_bgr.copy()
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            h, w, _ = frame_rgb.shape
            qimg = QtGui.QImage(frame_rgb.data, w, h, 3 * w, QtGui.QImage.Format_RGB888).copy()
            painter = QtGui.QPainter(qimg)
            if self.bars is not None and self.bars.size > 0:
                idx = 0
                vals = self.bars[idx]
                mode = self.combo_vis.currentText()
                if mode == "Столбцы":
                    self.draw_bars(painter, w, h, vals)
                else:
                    self.draw_circle(painter, w, h, vals)
            painter.end()
            pix = QtGui.QPixmap.fromImage(qimg)
            pix = pix.scaled(self.video_label.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
            self.video_label.setPixmap(pix)
        else:
            self.draw_placeholder()

    # ---------- Экспорт MP4 ----------
    def export_mp4(self):
        if (self.cap is None) and (self.still_image_bgr is None):
            QtWidgets.QMessageBox.information(self, "Нет источника", "Сначала загрузите видео или картинку.")
            return
        if self.audio_path is None:
            QtWidgets.QMessageBox.information(self, "Нет аудио", "Сначала загрузите аудиофайл (WAV).")
            return

        out_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Сохранить как", "render.mp4", "MP4 Video (*.mp4)"
        )
        if not out_path:
            return
        if not out_path.lower().endswith(".mp4"):
            out_path += ".mp4"

        ffmpeg_bin = self.find_ffmpeg()
        if ffmpeg_bin is None:
            # покажем текущий PATH для отладки
            cur_path = os.environ.get("PATH", "")
            btn = QtWidgets.QMessageBox.question(
                self,
                "ffmpeg не найден",
                "Приложение не видит ffmpeg в PATH.\n"
                f"Текущий PATH внутри приложения:\n{cur_path}\n\n"
                "Выбрать путь к ffmpeg вручную?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
            )
            if btn == QtWidgets.QMessageBox.Yes:
                ffmpeg_bin = self.ask_ffmpeg_path()
            if ffmpeg_bin is None:
                return

        # --- Рендер временного видео БЕЗ звука ---
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        if self.cap is not None:
            w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = float(self.fps)
            # заново от начала
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            n_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        else:
            # Источник — статичная картинка
            h, w = self.still_image_bgr.shape[:2]
            fps = float(self.fps)
            # если есть анализ аудио, синхронизируем длину видео с длиной аудио (bars)
            if self.bars is None or len(self.bars) == 0:
                QtWidgets.QMessageBox.critical(self, "Ошибка", "Нет рассчитанных фич аудио.")
                return
            n_frames = int(len(self.bars))

        prog = QtWidgets.QProgressDialog("Экспорт видео…", "Отмена", 0, n_frames, self)
        prog.setWindowModality(QtCore.Qt.WindowModal)
        prog.setMinimumDuration(0)

        # временная директория для файлов
        tmpdir = tempfile.mkdtemp(prefix="va_export_")
        tmp_video = os.path.join(tmpdir, "video_only.mp4")

        vw = cv2.VideoWriter(tmp_video, fourcc, fps, (w, h))
        if not vw.isOpened():
            QtWidgets.QMessageBox.critical(self, "Ошибка", "Не удалось открыть VideoWriter для MP4.")
            return

        if self.cap is not None:
            # Рендер по кадрам видео
            for i in range(n_frames):
                ret, frame_bgr = self.cap.read()
                if not ret:
                    break
                # индекс фич по fps
                idx = i
                # отрисовка выбранной визуализации поверх кадра
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                qimg = QtGui.QImage(frame_rgb.data, w, h, 3 * w, QtGui.QImage.Format_RGB888).copy()
                painter = QtGui.QPainter(qimg)
                if self.bars is not None and self.bars.size > 0:
                    use_idx = idx % len(self.bars)
                    vals = self.bars[use_idx]
                    mode = self.combo_vis.currentText()
                    if mode == "Столбцы":
                        self.draw_bars(painter, w, h, vals)
                    else:
                        self.draw_circle(painter, w, h, vals)
                painter.end()

                # обратно в BGR для записи
                ptr = qimg.bits()
                ptr.setsize(h * w * 3)
                arr = np.frombuffer(ptr, np.uint8).reshape((h, w, 3))
                frame_bgr_out = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
                vw.write(frame_bgr_out)

                if i % 5 == 0:
                    prog.setValue(i)
                    QtWidgets.QApplication.processEvents()
                    if prog.wasCanceled():
                        break
        else:
            # Рендер повторяющегося кадра-картинки
            base_bgr = self.still_image_bgr
            for i in range(n_frames):
                # отрисовка выбранной визуализации поверх статичной картинки
                frame_rgb = cv2.cvtColor(base_bgr, cv2.COLOR_BGR2RGB)
                qimg = QtGui.QImage(frame_rgb.data, w, h, 3 * w, QtGui.QImage.Format_RGB888).copy()
                painter = QtGui.QPainter(qimg)
                if self.bars is not None and self.bars.size > 0:
                    use_idx = i % len(self.bars)
                    vals = self.bars[use_idx]
                    mode = self.combo_vis.currentText()
                    if mode == "Столбцы":
                        self.draw_bars(painter, w, h, vals)
                    else:
                        self.draw_circle(painter, w, h, vals)
                painter.end()

                ptr = qimg.bits()
                ptr.setsize(h * w * 3)
                arr = np.frombuffer(ptr, np.uint8).reshape((h, w, 3))
                frame_bgr_out = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
                vw.write(frame_bgr_out)

                if i % 50 == 0:
                    prog.setValue(i)
                    QtWidgets.QApplication.processEvents()
                    if prog.wasCanceled():
                        break

        vw.release()
        prog.setValue(n_frames)

        if prog.wasCanceled():
            try:
                shutil.rmtree(tmpdir)
            except Exception:
                pass
            return

        # --- Сведение через ffmpeg ---
        # Важно: явно указываем маппинг потоков, кодеки и -shortest.
        cmd = [
            ffmpeg_bin, "-y",
            "-i", tmp_video,
            "-i", self.audio_path,
            "-map", "0:v:0",  # видео из первого входа
            "-map", "1:a:0",  # аудио из второго входа
            "-c:v", "copy",  # видео копируем (без повторной компрессии)
            "-c:a", "aac",  # перекодируем аудио в AAC
            "-b:a", "192k",
            "-shortest",  # обрезаем по корочеcму потоку
            out_path
        ]

        try:
            # stdout/stderr ловим для диагностики
            proc = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as e:
            # Покажем stderr ffmpeg — поможет понять, что именно ему не понравилось
            QtWidgets.QMessageBox.critical(
                self, "Ошибка ffmpeg",
                "ffmpeg не смог собрать видео с аудио.\n\n"
                f"Команда:\n{' '.join(cmd)}\n\n"
                f"stderr:\n{e.stderr.decode(errors='ignore')[:2000]}"
            )
            # оставим временные файлы для разбора
            return
        finally:
            try:
                shutil.rmtree(tmpdir)
            except Exception:
                pass

        QtWidgets.QMessageBox.information(self, "Готово", f"Экспорт завершён: {out_path}")


def main():
    app = QtWidgets.QApplication(sys.argv)
    w = VideoAudioVisualizer()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
