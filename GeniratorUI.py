#!/usr/bin/env python3
# sdxl_gui_simple_refactored_async_translate.py

import sys
import os
import threading
import time

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog

from AImodelImage import ImageAI
from AImodelTranslate import TranslateAI


class SDXLGui(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Audio Vision — Генератор картинок")
        self.resize(860, 620)

        # ---------- Вводы ----------
        self.ed_prompt = QtWidgets.QPlainTextEdit()
        self.ed_prompt.setPlaceholderText("Введите промпт (можно по-русски)...")
        self.ed_prompt.setMinimumHeight(100)

        self.sp_steps = QtWidgets.QSpinBox()
        self.sp_steps.setRange(1, 200)
        self.sp_steps.setValue(15)

        self.sp_width = QtWidgets.QSpinBox()
        self.sp_width.setRange(256, 2048)
        self.sp_width.setSingleStep(64)
        self.sp_width.setValue(1280)

        self.sp_height = QtWidgets.QSpinBox()
        self.sp_height.setRange(256, 2048)
        self.sp_height.setSingleStep(64)
        self.sp_height.setValue(720)

        self.cmb_device = QtWidgets.QComboBox()
        self.cmb_device.addItems(["mps", "cuda", "cpu"])
        self.cmb_device.setCurrentText("mps")

        self.btn_generate = QtWidgets.QPushButton("Сгенерировать")
        self.btn_generate.setEnabled(False)

        self.btn_saveas = QtWidgets.QPushButton("Сохранить как")

        self.lbl_status = QtWidgets.QLabel("Инициализация моделей…")
        self.lbl_status.setStyleSheet("color:#aaa")

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
        buttons.addWidget(self.btn_saveas)
        buttons.addStretch(1)
        buttons.addWidget(self.lbl_status)

        main = QtWidgets.QVBoxLayout(self)
        main.addWidget(QtWidgets.QLabel("Промпт:"))
        main.addWidget(self.ed_prompt)
        main.addLayout(grid)
        main.addWidget(self.preview, 1)
        main.addLayout(buttons)

        # ---------- Состояние ----------
        self.image_ai = ImageAI()
        self.translate_ai = TranslateAI()

        self._load_thread = None
        self._gen_thread = None
        self.models_ready = False

        # ---------- Сигналы ----------
        self.btn_generate.clicked.connect(self.on_generate)
        self.cmb_device.currentTextChanged.connect(self.on_device_change)
        self.btn_saveas.clicked.connect(self.savePictureAs)

        # асинхронная инициализация моделей
        self._start_load_models(device=self.cmb_device.currentText())

    # ----------- Загрузка моделей -----------
    def _start_load_models(self, device: str):
        self.btn_generate.setEnabled(False)
        self.models_ready = False
        self.lbl_status.setText("Загрузка моделей… (первый запуск может занять время)")
        self.lbl_status.setStyleSheet("color:#fb0")

        def load():
            try:
                pipe = self.image_ai.load(device)
                if not pipe:
                    raise RuntimeError("Не удалось инициализировать SDXL-пайплайн")

                # LLM для перевода — загружаем один раз
                if not hasattr(self.translate_ai, "llm"):
                    self.translate_ai.load()

                payload = (True, None)
            except Exception as e:
                payload = (False, str(e))

            QtCore.QMetaObject.invokeMethod(
                self, "_on_loaded", QtCore.Qt.QueuedConnection,
                QtCore.Q_ARG(object, payload)
            )

        self._load_thread = threading.Thread(target=load, daemon=True)
        self._load_thread.start()

    @QtCore.pyqtSlot(object)
    def _on_loaded(self, payload):
        ok, err = payload
        if not ok:
            self.lbl_status.setText("Ошибка загрузки моделей")
            self.lbl_status.setStyleSheet("color:#f55")
            QtWidgets.QMessageBox.critical(self, "Модели", f"Не удалось загрузить модели.\n\n{err}")
            return

        self.models_ready = True
        self.lbl_status.setText("Модели готовы")
        self.lbl_status.setStyleSheet("color:#0a0")
        self.btn_generate.setEnabled(True)

    # ----------- Перевод (обёртка) -----------
    def translate_prompt(self, txt: str) -> str:
        try:
            return self.translate_ai.translatePromt(txt)
        except Exception as e:
            print("Translate error:", e)
            return txt

    # ----------- Сохранение картинки -----------
    def savePictureAs(self, checked=False):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Сохранить файл как",
            "",
            "PNG Images (*.png)",
            options=options
        )

        if file_path and getattr(self.image_ai, "img", None) is not None:
            self.image_ai.img.save(file_path)

    # ----------- Генерация (ПЕРЕВОД + КАРТИНКА В ОДНОМ ПОТОКЕ) -----------
    def on_generate(self):
        if not self.models_ready:
            QtWidgets.QMessageBox.information(self, "SDXL", "Модели ещё не готовы.")
            return

        prompt_src = self.ed_prompt.toPlainText().strip()
        if not prompt_src:
            QtWidgets.QMessageBox.information(self, "SDXL", "Введите промпт.")
            return

        steps = self.sp_steps.value()
        w = self.sp_width.value()
        h = self.sp_height.value()

        # блокируем UI
        self.btn_generate.setEnabled(False)
        self.lbl_status.setText("Перевод и генерация…")
        self.lbl_status.setStyleSheet("color:#fb0")
        self.preview.setText("Перевод и генерация...")

        def run():
            try:
                # 1) ПЕРЕВОД В ФОНОВОМ ПОТОКЕ
                prompt_en = self.translate_prompt(prompt_src)
                print("PROMPT (EN):", prompt_en)

                # 2) ГЕНЕРАЦИЯ КАРТИНКИ В ЭТОМ ЖЕ ПОТОКЕ
                result = self.image_ai.run(steps=steps, prompt=prompt_en, w=w, h=h)
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
        self._start_load_models(device=dev)


def main():
    app = QtWidgets.QApplication(sys.argv)
    w = SDXLGui()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
