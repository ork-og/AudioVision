#!/usr/bin/env python3
# sdxl_gui_simple.py
import sys, os, threading, time
from PyQt5 import QtCore, QtGui, QtWidgets

# --- импорт нейросети ---
import torch
from diffusers import StableDiffusionXLPipeline

class SDXLGui(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Audio Vision  — Генератор картинок")
        self.resize(860, 620)

        # ---------- Вводы ----------
        self.ed_prompt = QtWidgets.QPlainTextEdit()
        self.ed_prompt.setPlaceholderText("Введите промпт...")
        self.ed_prompt.setMinimumHeight(100)

        self.sp_steps = QtWidgets.QSpinBox()
        self.sp_steps.setRange(1, 200)
        self.sp_steps.setValue(15)
        self.sp_steps.setToolTip("Число диффузионных шагов (эпох)")

        self.sp_width = QtWidgets.QSpinBox()
        self.sp_width.setRange(256, 2048)
        self.sp_width.setSingleStep(64)
        self.sp_width.setValue(1920)

        self.sp_height = QtWidgets.QSpinBox()
        self.sp_height.setRange(256, 2048)
        self.sp_height.setSingleStep(64)
        self.sp_height.setValue(1080)

        self.cmb_device = QtWidgets.QComboBox()
        # по умолчанию — mps (для Apple Silicon); оставил варианты на всякий
        self.cmb_device.addItems(["mps", "cuda", "cpu"])
        self.cmb_device.setCurrentText("mps")

        self.btn_generate = QtWidgets.QPushButton("Сгенерировать")
        self.btn_generate.setEnabled(False)

        self.lbl_status = QtWidgets.QLabel("Инициализация модели…")
        self.lbl_status.setStyleSheet("color:#aaa")

        # предпросмотр
        self.preview = QtWidgets.QLabel("Предпросмотр появится здесь")
        self.preview.setAlignment(QtCore.Qt.AlignCenter)
        self.preview.setStyleSheet("background:#111; color:#777; border:1px solid #333;")
        self.preview.setMinimumSize(480, 320)

        # ---------- Разметка ----------
        grid = QtWidgets.QGridLayout()
        grid.addWidget(QtWidgets.QLabel("Эпохи:"), 0, 0)
        grid.addWidget(self.sp_steps, 0, 1)
        grid.addWidget(QtWidgets.QLabel("Ширина:"), 0, 2)
        grid.addWidget(self.sp_width, 0, 3)
        grid.addWidget(QtWidgets.QLabel("Высота:"), 0, 4)
        grid.addWidget(self.sp_height, 0, 5)
        grid.addWidget(QtWidgets.QLabel("Device:"), 0, 6)
        grid.addWidget(self.cmb_device, 0, 7)

        buttons = QtWidgets.QHBoxLayout()
        buttons.addWidget(self.btn_generate)
        buttons.addStretch(1)
        buttons.addWidget(self.lbl_status)

        main = QtWidgets.QVBoxLayout(self)
        main.addWidget(QtWidgets.QLabel("Промпт:"))
        main.addWidget(self.ed_prompt)
        main.addLayout(grid)
        main.addWidget(self.preview, 1)
        main.addLayout(buttons)

        # ---------- Состояние ----------
        self.pipe = None
        self._load_thread = None
        self._gen_thread = None

        # ---------- Сигналы ----------
        self.btn_generate.clicked.connect(self.on_generate)
        self.cmb_device.currentTextChanged.connect(self.on_device_change)

        # Стартуем загрузку модели (асинхронно)
        self._start_load_pipeline(device=self.cmb_device.currentText())

    # ----------- Загрузка модели -----------
    def _start_load_pipeline(self, device: str):
        self.btn_generate.setEnabled(False)
        self.lbl_status.setText("Загрузка модели… (первый запуск может быть долгим)")
        self.lbl_status.setStyleSheet("color:#fb0")
        self.pipe = None

        def load():
            try:
                # dtype/variant подбираем под устройство
                use_fp16 = device in ("mps", "cuda")
                dtype = torch.float16 if use_fp16 else torch.float32

                pipe = StableDiffusionXLPipeline.from_pretrained(
                    "stabilityai/stable-diffusion-xl-base-1.0",
                    torch_dtype=dtype,
                    variant="fp16" if use_fp16 else None,
                ).to(device)

            except Exception as e:
                QtCore.QMetaObject.invokeMethod(
                    self, "_on_loaded", QtCore.Qt.QueuedConnection,
                    QtCore.Q_ARG(object, (None, str(e)))
                )
                return

            QtCore.QMetaObject.invokeMethod(
                self, "_on_loaded", QtCore.Qt.QueuedConnection,
                QtCore.Q_ARG(object, (pipe, None))
            )

        self._load_thread = threading.Thread(target=load, daemon=True)
        self._load_thread.start()

    @QtCore.pyqtSlot(object)
    def _on_loaded(self, payload):
        pipe, err = payload
        if err:
            self.lbl_status.setText("Ошибка загрузки модели")
            self.lbl_status.setStyleSheet("color:#f55")
            QtWidgets.QMessageBox.critical(self, "SDXL", f"Не удалось загрузить модель.\n\n{err}")
            return
        self.pipe = pipe
        self.lbl_status.setText("Модель готова")
        self.lbl_status.setStyleSheet("color:#0a0")
        self.btn_generate.setEnabled(True)

    # ----------- Генерация -----------
    def on_generate(self):
        if self.pipe is None:
            QtWidgets.QMessageBox.information(self, "SDXL", "Модель ещё не готова.")
            return

        prompt = self.ed_prompt.toPlainText().strip()
        if not prompt:
            QtWidgets.QMessageBox.information(self, "SDXL", "Введите промпт.")
            return

        steps = self.sp_steps.value()
        w = self.sp_width.value()
        h = self.sp_height.value()

        # UI lock
        self.btn_generate.setEnabled(False)
        self.lbl_status.setText("Генерация…")
        self.lbl_status.setStyleSheet("color:#fb0")
        self.preview.setText("Генерация...")

        def run():
            try:
                torch.manual_seed(int(time.time()))
                img = self.pipe(
                    prompt=prompt,
                    num_inference_steps=int(steps),
                    width=int(w),
                    height=int(h),
                ).images[0]

                out_name = f"output_{w}x{h}.png"
                img.save(out_name)
                result = (out_name, None)
            except Exception as e:
                result = (None, str(e))

            QtCore.QMetaObject.invokeMethod(
                self, "_on_generated", QtCore.Qt.QueuedConnection,
                QtCore.Q_ARG(object, result)
            )

        self._gen_thread = threading.Thread(target=run, daemon=True)
        self._gen_thread.start()

    @QtCore.pyqtSlot(object)
    def _on_generated(self, payload):
        out_path, err = payload
        self.btn_generate.setEnabled(True)

        if err:
            self.lbl_status.setText("Ошибка генерации")
            self.lbl_status.setStyleSheet("color:#f55")
            self.preview.setText("Ошибка")
            QtWidgets.QMessageBox.critical(self, "SDXL", f"Не удалось сгенерировать.\n\n{err}")
            return

        self.lbl_status.setText(f"Готово: {os.path.basename(out_path)}")
        self.lbl_status.setStyleSheet("color:#0a0")

        pix = QtGui.QPixmap(out_path)
        if not pix.isNull():
            pix = pix.scaled(self.preview.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
            self.preview.setPixmap(pix)
        else:
            self.preview.setText("Не удалось открыть изображение")

    # ----------- Перезагрузка при смене устройства -----------
    def on_device_change(self, dev):
        # Перезагружаем пайплайн под новое устройство
        self._start_load_pipeline(device=dev)

def main():
    app = QtWidgets.QApplication(sys.argv)
    w = SDXLGui()
    w.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
#Стандартное разрешение для YouTube 720p — это:

# 1280 × 720 пикселей
# Соотношение сторон: 16:9



#Ширина: 1280 px

#Высота: 720 px

#Найдено оптимальное данные для генерации во-первых это средний стандарт для
#YouTube во-вторых при 15 слоях картинки все достаточно быстро генерируется и что самое главное хватает оперативной памятиВозможно при таком разрешении можно даже будет увеличить количество наложений до 20 – 25