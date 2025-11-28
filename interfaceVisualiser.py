import sys
import os
import math

import numpy as np
import cv2

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtCore import QUrl, QSettings

from voice_control import VoiceController, VoiceConfig
from EngineAV import AVVisualizerEngine


def qcolor(r, g, b, a=255):
    c = QtGui.QColor(int(r), int(g), int(b), int(a))
    return c


class VideoAudioVisualizer(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Видеоплеер + аудио-визуализация (PyQt5)")
        self.resize(1100, 700)

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)

        self.video_label = QtWidgets.QLabel("Загрузите видео/картинку и аудио…\nили просто перетащите файл сюда")
        self.video_label.setAlignment(QtCore.Qt.AlignCenter)
        self.video_label.setStyleSheet("background:#111; color:#aaa; font-size:16px; border: 2px solid #222;")
        self.video_label.setMinimumSize(800, 450)

        self.btn_load_video = QtWidgets.QPushButton("Загрузить видео…")
        self.btn_load_image = QtWidgets.QPushButton("Загрузить картинку…")
        self.btn_load_audio = QtWidgets.QPushButton("Загрузить аудио (WAV)…")
        self.btn_play = QtWidgets.QPushButton("▶️ Пуск")
        self.btn_play.setCheckable(True)
        self.btn_play.setEnabled(False)
        self.btn_export = QtWidgets.QPushButton("Сохранить MP4…")

        self.combo_vis = QtWidgets.QComboBox()
        self.combo_vis.addItems(["Столбцы", "Пульсирующая окружность"])

        self.btn_color = QtWidgets.QPushButton("Цвет…")
        self.vis_color = QtGui.QColor(255, 255, 255)
        self._apply_btn_color_style()

        self.btn_col_bass = QtWidgets.QPushButton("Бас/Кик")
        self.btn_col_low  = QtWidgets.QPushButton("Низы")
        self.btn_col_mid  = QtWidgets.QPushButton("Средние")
        self.btn_col_high = QtWidgets.QPushButton("ВЧ")
        self.btn_col_top  = QtWidgets.QPushButton("СверхВЧ")

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
        controls.addWidget(self.btn_col_bass)
        controls.addWidget(self.btn_col_low)
        controls.addWidget(self.btn_col_mid)
        controls.addWidget(self.btn_col_high)
        controls.addWidget(self.btn_col_top)

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

        # Кнопка голосового управления
        self.btn_voice = QtWidgets.QPushButton("🎤 Голос")
        self.btn_voice.setCheckable(True)
        controls.addWidget(self.btn_voice)

        # Голосовой контроллер
        self.voice = VoiceController(VoiceConfig())
        self.voice.voiceCommand.connect(self.on_voice_command)
        self.btn_voice.toggled.connect(self.toggle_voice)

        # ---------- Состояние (GUI + движок) ----------
        self.n_bins = 32
        self.engine = AVVisualizerEngine(n_bins=self.n_bins, fps_default=30.0)

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.next_frame)
        self.frame_index = 0

        self.player: QMediaPlayer | None = None  # для проигрывания аудио

        # Частотные группы и цвета
        self.freq_split = {
            "basskick_max": 100.0,
            "low_max":      500.0,
            "mid_max":      2000.0,
            "high_max":     6000.0,
        }
        self.color_basskick = qcolor(255, 94, 168)
        self.color_low     = qcolor(50, 140, 255)
        self.color_mid     = qcolor(72, 245, 139)
        self.color_high    = qcolor(255, 217, 102)
        self.color_ultra   = qcolor(255, 255, 255)

        self.width_basskick = 6
        self.width_low      = 4
        self.width_mid      = 3
        self.width_high     = 2
        self.width_ultra    = 1

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

        self.btn_col_bass.clicked.connect(lambda: self._choose_group_color('bass'))
        self.btn_col_low.clicked.connect(lambda: self._choose_group_color('low'))
        self.btn_col_mid.clicked.connect(lambda: self._choose_group_color('mid'))
        self.btn_col_high.clicked.connect(lambda: self._choose_group_color('high'))
        self.btn_col_top.clicked.connect(lambda: self._choose_group_color('ultra'))

        for b, c in [
            (self.btn_col_bass, self.color_basskick),
            (self.btn_col_low,  self.color_low),
            (self.btn_col_mid,  self.color_mid),
            (self.btn_col_high, self.color_high),
            (self.btn_col_top,  self.color_ultra),
        ]:
            self._refresh_group_btn(b, c)

        # Drag-and-Drop
        self.setAcceptDrops(True)
        self.video_label.setAcceptDrops(True)
        self.video_label.installEventFilter(self)
        self._dnd_highlight_on = False

    # ---------- Настройки / ffmpeg ----------

    def _settings(self) -> QSettings:
        return QSettings("YourOrg", "VideoAudioVisualizer")

    def find_ffmpeg(self) -> str | None:
        s = self._settings()
        saved = s.value("ffmpeg_path", type=str)
        if saved and os.path.isfile(saved) and os.access(saved, os.X_OK):
            return saved
        return self.engine.find_ffmpeg()

    def ask_ffmpeg_path(self) -> str | None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Укажите бинарник ffmpeg", "", "Все файлы (*)")
        if not path:
            return None
        if not os.path.isfile(path):
            QtWidgets.QMessageBox.warning(self, "ffmpeg", "Указан несуществующий файл.")
            return None
        if not os.access(path, os.X_OK):
            try:
                os.chmod(path, os.stat(path).st_mode | 0o111)
            except Exception:
                pass
            if not os.access(path, os.X_OK):
                QtWidgets.QMessageBox.warning(
                    self, "ffmpeg",
                    "Файл не исполняемый. Сделайте его исполняемым или выберите другой."
                )
                return None
        s = self._settings()
        s.setValue("ffmpeg_path", path)
        return path

    # ---------- Drag & Drop ----------

    def is_image_file(self, path: str) -> bool:
        ext = os.path.splitext(path)[1].lower()
        return ext in [".png", ".jpg", ".jpeg", ".bmp", ".webp"]

    def is_wav_file(self, path: str) -> bool:
        return os.path.splitext(path)[1].lower() == ".wav"

    def _set_drop_highlight(self, on: bool):
        if on == self._dnd_highlight_on:
            return
        self._dnd_highlight_on = on
        if on:
            self.video_label.setStyleSheet(
                "background:#111; color:#aaa; font-size:16px; border: 2px dashed #4da3ff;"
            )
        else:
            self.video_label.setStyleSheet(
                "background:#111; color:#aaa; font-size:16px; border: 2px solid #222;"
            )

    def _extract_local_paths(self, event) -> list:
        urls = event.mimeData().urls()
        paths = []
        for u in urls:
            if u.isLocalFile():
                paths.append(u.toLocalFile())
        return paths

    def handle_dropped_paths(self, paths: list):
        if not paths:
            return

        img_loaded = False
        wav_loaded = False
        other_files = []

        for p in paths:
            if os.path.isdir(p):
                continue
            if self.is_image_file(p) and not img_loaded:
                self.open_image_path(p)
                img_loaded = True
            elif self.is_wav_file(p) and not wav_loaded:
                self.open_audio_path(p)
                wav_loaded = True
            else:
                other_files.append(p)

        if other_files:
            pretty = "\n".join(os.path.basename(x) for x in other_files)
            QtWidgets.QMessageBox.information(
                self, "Не поддерживается",
                "Поддерживаются изображения (.png .jpg .jpeg .bmp .webp) и аудио WAV (.wav).\n"
                f"Пропущены файлы:\n{pretty}"
            )

    def dragEnterEvent(self, event: QtGui.QDragEnterEvent):
        if event.mimeData().hasUrls():
            paths = self._extract_local_paths(event)
            if any(self.is_image_file(p) or self.is_wav_file(p) for p in paths):
                event.acceptProposedAction()
                self._set_drop_highlight(True)
                return
        event.ignore()

    def dragLeaveEvent(self, event: QtGui.QDragLeaveEvent):
        self._set_drop_highlight(False)
        event.accept()

    def dropEvent(self, event: QtGui.QDropEvent):
        paths = self._extract_local_paths(event)
        self.handle_dropped_paths(paths)
        self._set_drop_highlight(False)
        event.acceptProposedAction()

    def eventFilter(self, obj, ev):
        if obj is self.video_label:
            if ev.type() == QtCore.QEvent.DragEnter:
                if ev.mimeData().hasUrls():
                    paths = self._extract_local_paths(ev)
                    if any(self.is_image_file(p) or self.is_wav_file(p) for p in paths):
                        ev.acceptProposedAction()
                        self._set_drop_highlight(True)
                        return True
                ev.ignore()
                return True
            elif ev.type() == QtCore.QEvent.DragLeave:
                self._set_drop_highlight(False)
                ev.accept()
                return True
            elif ev.type() == QtCore.QEvent.Drop:
                paths = self._extract_local_paths(ev)
                self.handle_dropped_paths(paths)
                self._set_drop_highlight(False)
                ev.acceptProposedAction()
                return True
        return super().eventFilter(obj, ev)

    # ---------- Конвертеры/утилиты ----------

    def qimage_to_bgr_safe(self, qimg: QtGui.QImage) -> np.ndarray:
        qimg = qimg.convertToFormat(QtGui.QImage.Format_RGB888)
        w = qimg.width()
        h = qimg.height()
        bpl = qimg.bytesPerLine()
        ptr = qimg.bits()
        ptr.setsize(bpl * h)
        buf = np.frombuffer(ptr, np.uint8).reshape((h, bpl))
        rgb = buf[:, : w * 3].reshape((h, w, 3))
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        return bgr

    # ---------- Цвет / визуал ----------

    def _with_alpha(self, color: QtGui.QColor, alpha: int) -> QtGui.QColor:
        c = QtGui.QColor(color)
        c.setAlpha(max(0, min(255, int(alpha))))
        return c

    def _apply_btn_color_style(self):
        c = self.vis_color
        self.btn_color.setFixedWidth(90)
        self.btn_color.setStyleSheet(
            f"QPushButton{{padding:6px 10px; border-radius:6px; border:1px solid #444;"
            f"background-color: rgba({c.red()},{c.green()},{c.blue()},255); color: #000;}}"
            f"QPushButton:hover{{filter: brightness(1.08);}}"
        )

    def _refresh_group_btn(self, btn, color):
        btn.setStyleSheet(
            f"QPushButton{{padding:6px 10px; border-radius:6px; border:1px solid #444;"
            f"background-color: rgba({color.red()},{color.green()},{color.blue()},255); color: #000;}}"
            f"QPushButton:hover{{filter: brightness(1.08);}}"
        )

    def _choose_group_color(self, which: str):
        cur = {
            'bass': self.color_basskick,
            'low':  self.color_low,
            'mid':  self.color_mid,
            'high': self.color_high,
            'ultra':self.color_ultra
        }[which]
        c = QtWidgets.QColorDialog.getColor(cur, self, "Цвет диапазона")
        if not c.isValid():
            return
        if which == 'bass':
            self.color_basskick = c
            self._refresh_group_btn(self.btn_col_bass, c)
        elif which == 'low':
            self.color_low = c
            self._refresh_group_btn(self.btn_col_low, c)
        elif which == 'mid':
            self.color_mid = c
            self._refresh_group_btn(self.btn_col_mid, c)
        elif which == 'high':
            self.color_high = c
            self._refresh_group_btn(self.btn_col_high, c)
        else:
            self.color_ultra = c
            self._refresh_group_btn(self.btn_col_top, c)
        self.on_vis_changed(0)

    def choose_vis_color(self):
        c = QtWidgets.QColorDialog.getColor(self.vis_color, self, "Выберите цвет визуализации")
        if c.isValid():
            self.vis_color = c
            self._apply_btn_color_style()
            self.on_vis_changed(0)

    def _color_and_width_by_freq(self, f_hz: float):
        if f_hz <= self.freq_split["basskick_max"]:
            return self.color_basskick, self.width_basskick
        elif f_hz <= self.freq_split["low_max"]:
            return self.color_low, self.width_low
        elif f_hz <= self.freq_split["mid_max"]:
            return self.color_mid, self.width_mid
        elif f_hz <= self.freq_split["high_max"]:
            return self.color_high, self.width_high
        else:
            return self.color_ultra, self.width_ultra

    # ---------- Отрисовка ----------

    def draw_bars(self, painter, w, h, vals):
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

        x = margin_lr
        centers = self.engine.band_centers
        for i, v in enumerate(vals):
            h_pix = int(v * area_h)
            rect = QtCore.QRect(x, base_y - h_pix, bar_w, h_pix)

            if centers is not None and i < len(centers):
                f = float(centers[i])
                c, _w = self._color_and_width_by_freq(f)
            else:
                c, _w = self.vis_color, 2

            pen = QtGui.QPen(self._with_alpha(c, 220))
            brush = QtGui.QBrush(self._with_alpha(c, 200))
            painter.setPen(pen)
            painter.setBrush(brush)
            painter.drawRect(rect)

            x += bar_w + gap

        painter.setPen(QtGui.QPen(self._with_alpha(QtGui.QColor(255, 255, 255), 80), 1))
        painter.drawLine(margin_lr, base_y, margin_lr + area_w, base_y)

    def draw_circle(self, painter, w, h, vals):
        cx, cy = w // 2, int(h * 0.53)
        r_inner = int(0.12 * h)
        max_len = int(0.16 * h)

        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)

        base_pen = QtGui.QPen(self._with_alpha(self.vis_color, 70))
        base_pen.setWidth(2)
        painter.setPen(base_pen)
        painter.setBrush(QtCore.Qt.NoBrush)
        painter.drawEllipse(QtCore.QPoint(cx, cy), r_inner, r_inner)

        n = len(vals)
        centers = self.engine.band_centers

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

            if centers is not None and i < len(centers):
                f = float(centers[i])
                c, wline = self._color_and_width_by_freq(f)
            else:
                c, wline = self.vis_color, 2

            pen = QtGui.QPen(self._with_alpha(c, 180))
            pen.setWidth(int(wline))
            painter.setPen(pen)
            painter.drawLine(x1, y1, x2, y2)

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
            c_center = e["center"]
            amp = e["amp"]
            steps = 36
            for k in range(-steps, steps + 1):
                t = k / steps
                ang = c_center + t * ear_half
                window = 0.5 * (1 + math.cos(math.pi * t))
                L = int(amp * ear_max * (window ** 1.2))
                ca, sa = math.cos(ang), math.sin(ang)
                x1 = cx + int(ca * r_inner)
                y1 = cy + int(sa * r_inner)
                x2 = cx + int(ca * (r_inner + L))
                y2 = cy + int(sa * (r_inner + L))

                ear_color = self.color_low if c_center in (0.0, math.pi) else self.color_ultra
                pen = QtGui.QPen(self._with_alpha(ear_color, e["alpha"]))
                pen.setWidth(4)
                painter.setPen(pen)
                painter.drawLine(x1, y1, x2, y2)

        glow_pen = QtGui.QPen(self._with_alpha(self.vis_color, 50))
        glow_pen.setWidth(6)
        painter.setPen(glow_pen)
        painter.drawEllipse(QtCore.QPoint(cx, cy),
                            r_inner + int(0.5 * max_len),
                            r_inner + int(0.5 * max_len))

    # ---------- Голос ----------

    def toggle_voice(self, checked: bool):
        if checked:
            if not self.voice.cfg.model_path and not os.environ.get("VOSK_MODEL"):
                QtWidgets.QMessageBox.information(
                    self, "Vosk",
                    "Укажите путь к модели Vosk (переменная окружения VOSK_MODEL) "
                    "или пропишите voice.cfg.model_path в коде."
                )
                self.btn_voice.setChecked(False)
                return
            ok = self.voice.start()
            if not ok:
                QtWidgets.QMessageBox.critical(
                    self, "Vosk",
                    "Не удалось запустить распознавание. Проверьте модель/микрофон."
                )
                self.btn_voice.setChecked(False)
                return
            self.btn_voice.setText("🛑 Голос")
        else:
            self.voice.stop()
            self.btn_voice.setText("🎤 Голос")

    @QtCore.pyqtSlot(str, object)
    def on_voice_command(self, cmd: str, payload):
        try:
            if cmd == "image":
                if isinstance(payload, str) and os.path.exists(payload):
                    self.open_image_path(payload)
                else:
                    self.load_image()
            elif cmd == "audio":
                if isinstance(payload, str) and os.path.exists(payload):
                    self.open_audio_path(payload)
                else:
                    self.load_audio()
            elif cmd == "play":
                if not self.btn_play.isChecked():
                    self.btn_play.setChecked(True)
            elif cmd == "pause":
                if self.btn_play.isChecked():
                    self.btn_play.setChecked(False)
            elif cmd == "mode":
                if payload == "bars":
                    self.combo_vis.setCurrentIndex(0)
                elif payload == "ring":
                    self.combo_vis.setCurrentIndex(1)
            elif cmd == "export":
                self.export_mp4()
            elif cmd == "volume":
                v = int(payload)
                self.slider_volume.setValue(v)
            elif cmd == "volume_step":
                cur = self.slider_volume.value()
                self.slider_volume.setValue(max(0, min(100, cur + int(payload))))
            elif cmd == "speed":
                spd = float(payload)
                self.slider_speed.setValue(int(round(100 * spd)))
        except Exception:
            pass

    # ---------- Основной цикл отрисовки ----------

    def next_frame(self):
        try:
            if self.engine.cap is not None and self.engine.n_video_frames > 0:
                if self.frame_index >= self.engine.n_video_frames:
                    self.frame_index = 0
            frame_bgr = self.engine.get_background_frame(self.frame_index)
        except RuntimeError:
            return

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w, _ = frame_rgb.shape
        qimg = QtGui.QImage(frame_rgb.data, w, h, 3 * w, QtGui.QImage.Format_RGB888).copy()

        painter = QtGui.QPainter(qimg)

        bars_vec = None
        if self.engine.bars is not None and self.engine.bars.size > 0:
            if self.player is not None:
                pos_ms = self.player.position()
                t_sec = pos_ms / 1000.0
                bars_vec = self.engine.get_bar_values_for_time(t_sec, wrap=True)
            else:
                bars_vec = self.engine.get_bar_values_for_frame(self.frame_index, wrap=True)

        if bars_vec is not None:
            mode = self.combo_vis.currentText()
            if mode == "Столбцы":
                self.draw_bars(painter, w, h, bars_vec)
            else:
                self.draw_circle(painter, w, h, bars_vec)
        painter.end()

        pix = QtGui.QPixmap.fromImage(qimg)
        pix = pix.scaled(self.video_label.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        self.video_label.setPixmap(pix)
        self.frame_index += 1

    # ---------- Загрузка видео / картинки / аудио ----------

    def load_video(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Выберите видеофайл",
            "",
            "Видео (*.mp4 *.avi *.mkv *.mov *.webm);;Все файлы (*.*)"
        )
        if not path:
            return
        try:
            self.engine.load_video(path)
        except RuntimeError as e:
            QtWidgets.QMessageBox.critical(self, "Ошибка", str(e))
            return

        self.frame_index = 0
        self.update_window_title()
        self.update_play_button_state()
        self.draw_placeholder()

    def load_image(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Выберите изображение",
            "",
            "Изображения (*.png *.jpg *.jpeg *.bmp *.webp);;Все файлы (*.*)"
        )
        if not path:
            return
        self.open_image_path(path)

    def open_image_path(self, path: str):
        try:
            self.engine.load_image(path)
        except RuntimeError as e:
            QtWidgets.QMessageBox.critical(self, "Ошибка", str(e))
            return

        self.frame_index = 0
        self.on_vis_changed(0)
        self.update_window_title()
        self.update_play_button_state()

    def load_audio(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Выберите аудиофайл (WAV)",
            "",
            "Аудио WAV (*.wav);;Все файлы (*.*)"
        )
        if not path:
            return
        self.open_audio_path(path)

    def open_audio_path(self, path: str):
        try:
            self.engine.load_audio_wav(path)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Ошибка аудио", f"Не удалось прочитать WAV.\n{e}")
            return

        if self.player is None:
            self.player = QMediaPlayer(self)
            self.player.mediaStatusChanged.connect(self.on_media_status)

        self.player.setMedia(QMediaContent(QUrl.fromLocalFile(path)))
        self.player.setVolume(self.slider_volume.value())
        self.player.setPlaybackRate(self.slider_speed.value() / 100.0)

        self.update_window_title()
        self.update_play_button_state()
        self.on_vis_changed(0)

    def update_window_title(self):
        if self.engine.video_path:
            vname = os.path.basename(self.engine.video_path)
        elif self.engine.image_path:
            vname = os.path.basename(self.engine.image_path)
        else:
            vname = "(нет видео/картинки)"

        aname = os.path.basename(self.engine.audio_path) if self.engine.audio_path else "(нет аудио)"
        self.setWindowTitle(f"Фон: {vname}  |  Аудио: {aname}  |  FPS: {self.engine.fps:.2f}")

    def update_play_button_state(self):
        self.btn_play.setEnabled(
            (self.engine.cap is not None) or (self.engine.still_image_bgr is not None)
        )

    # ---------- Управление плеером / скоростью ----------

    def on_speed_changed(self, val):
        spd = val / 100.0
        self.lbl_speed.setText(f"Скорость: {spd:.2f}x")
        if self.timer.isActive():
            interval_ms = max(1, int(1000.0 / (self.engine.fps * spd)))
            self.timer.setInterval(interval_ms)
        if self.player is not None:
            self.player.setPlaybackRate(spd)

    def toggle_play(self, checked):
        if checked:
            if (self.engine.cap is None) and (self.engine.still_image_bgr is None):
                QtWidgets.QMessageBox.information(self, "Нет источника", "Сначала загрузите видео или картинку.")
                self.btn_play.setChecked(False)
                return
            spd = self.slider_speed.value() / 100.0
            interval_ms = max(1, int(1000.0 / (self.engine.fps * spd)))
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
            self.video_label.setPixmap(
                self.video_label.pixmap().scaled(
                    self.video_label.size(),
                    QtCore.Qt.KeepAspectRatio,
                    QtCore.Qt.SmoothTransformation
                )
            )

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

        if self.engine.cap is not None:
            cur = max(0, self.frame_index - 1)
            try:
                frame_bgr = self.engine.get_background_frame(cur)
            except RuntimeError:
                self.draw_placeholder()
                return
            bars_vec = self.engine.get_bar_values_for_frame(cur, wrap=True)
        elif self.engine.still_image_bgr is not None:
            frame_bgr = self.engine.still_image_bgr.copy()
            bars_vec = self.engine.get_bar_values_for_frame(0, wrap=True)
        else:
            self.draw_placeholder()
            return

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w, _ = frame_rgb.shape
        qimg = QtGui.QImage(frame_rgb.data, w, h, 3 * w, QtGui.QImage.Format_RGB888).copy()

        painter = QtGui.QPainter(qimg)
        if bars_vec is not None:
            mode = self.combo_vis.currentText()
            if mode == "Столбцы":
                self.draw_bars(painter, w, h, bars_vec)
            else:
                self.draw_circle(painter, w, h, bars_vec)
        painter.end()

        pix = QtGui.QPixmap.fromImage(qimg)
        pix = pix.scaled(
            self.video_label.size(),
            QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.SmoothTransformation
        )
        self.video_label.setPixmap(pix)

    # ---------- Экспорт MP4 ----------

    def export_mp4(self):
        if (self.engine.cap is None) and (self.engine.still_image_bgr is None):
            QtWidgets.QMessageBox.information(self, "Нет источника", "Сначала загрузите видео или картинку.")
            return
        if self.engine.audio_path is None or self.engine.bars is None:
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

        try:
            n_frames = self.engine.estimate_export_frame_count()
        except RuntimeError as e:
            QtWidgets.QMessageBox.critical(self, "Ошибка", str(e))
            return

        prog = QtWidgets.QProgressDialog("Экспорт видео…", "", 0, n_frames, self)
        prog.setWindowModality(QtCore.Qt.WindowModal)
        prog.setMinimumDuration(0)
        prog.setCancelButton(None)

        def progress_callback(i, n):
            prog.setValue(i)
            QtWidgets.QApplication.processEvents()

        def overlay_callback(frame_bgr, frame_index, bars_vec):
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            h, w, _ = frame_rgb.shape
            qimg = QtGui.QImage(frame_rgb.data, w, h, 3 * w, QtGui.QImage.Format_RGB888).copy()
            painter = QtGui.QPainter(qimg)
            if bars_vec is not None:
                mode = self.combo_vis.currentText()
                if mode == "Столбцы":
                    self.draw_bars(painter, w, h, bars_vec)
                else:
                    self.draw_circle(painter, w, h, bars_vec)
            painter.end()
            return self.qimage_to_bgr_safe(qimg)

        try:
            self.engine.export_mp4(
                out_path=out_path,
                ffmpeg_bin=ffmpeg_bin,
                loop_bars=True,
                progress_callback=progress_callback,
                overlay_callback=overlay_callback,
            )
        except RuntimeError as e:
            QtWidgets.QMessageBox.critical(
                self, "Ошибка ffmpeg",
                f"Не удалось экспортировать видео:\n{e}"
            )
            return

        prog.setValue(n_frames)
        QtWidgets.QMessageBox.information(self, "Готово", f"Экспорт завершён: {out_path}")


def main():
    app = QtWidgets.QApplication(sys.argv)
    w = VideoAudioVisualizer()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
