import os
import math
import wave
import shutil
import tempfile
import subprocess

import numpy as np
import cv2


class AVVisualizerEngine:
    """
    Бэкенд для работы с видео/изображением и аудио (WAV) без привязки к GUI.

    Отвечает за:
      - загрузку видео/картинки;
      - загрузку WAV и расчёт баров;
      - выдачу кадров фона;
      - расчёт индекса бара по времени;
      - оценку числа кадров для экспорта;
      - экспорт MP4 через ffmpeg (с внешним overlay_callback).
    """

    def __init__(self, n_bins: int = 32, fps_default: float = 30.0):
        self.n_bins = int(n_bins)
        self.fps_default = float(fps_default)

        self.video_path: str | None = None
        self.image_path: str | None = None
        self.audio_path: str | None = None

        self.cap: cv2.VideoCapture | None = None
        self.still_image_bgr: np.ndarray | None = None

        self.fps: float = fps_default
        self.n_video_frames: int = 0

        self.bars: np.ndarray | None = None        # (n_frames, n_bins)
        self.band_edges: np.ndarray | None = None  # (n_bins+1,)
        self.band_centers: np.ndarray | None = None  # (n_bins,)

    # ---------------------------
    # АУДИО (WAV)
    # ---------------------------

    @staticmethod
    def read_wav_mono(path: str) -> tuple[np.ndarray, int]:
        with wave.open(path, 'rb') as wf:
            n_channels = wf.getnchannels()
            sr = wf.getframerate()
            n_frames = wf.getnframes()
            sampwidth = wf.getsampwidth()
            raw = wf.readframes(n_frames)

        if sampwidth == 1:
            dtype = np.uint8
            data = np.frombuffer(raw, dtype=dtype).astype(np.float32)
            data = (data - 128.0) / 128.0
        elif sampwidth == 2:
            dtype = np.int16
            data = np.frombuffer(raw, dtype=dtype).astype(np.float32) / 32768.0
        elif sampwidth == 3:
            a = np.frombuffer(raw, dtype=np.uint8)
            a = a.reshape(-1, 3)
            b = (
                a[:, 0].astype(np.int32)
                | (a[:, 1].astype(np.int32) << 8)
                | (a[:, 2].astype(np.int32) << 16)
            )
            mask = b & 0x800000
            b = b - (mask << 1)
            data = b.astype(np.float32) / 8388608.0
        elif sampwidth == 4:
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

        return data.astype(np.float32), sr

    @staticmethod
    def make_bar_features(
        audio: np.ndarray,
        sr: int,
        fps: float,
        n_bins: int = 32,
        ref_median_frames: int = 60
    ) -> np.ndarray:
        eps = 1e-8
        samples_per_frame = max(1, int(round(sr / float(fps))))
        n_frames = int(math.ceil(len(audio) / samples_per_frame))

        hann = np.hanning(samples_per_frame).astype(np.float32)
        freqs = np.fft.rfftfreq(samples_per_frame, d=1.0 / sr)

        f_min = 20.0
        f_max = min(sr / 2.0, 16000.0)
        edges = np.geomspace(f_min, f_max, n_bins + 1)

        band_indices = []
        for i in range(n_bins):
            f1, f2 = edges[i], edges[i + 1]
            idx = np.where((freqs >= f1) & (freqs < f2))[0]
            if len(idx) == 0:
                nearest = np.argmin(np.abs(freqs - (f1 + f2) * 0.5))
                idx = np.array([nearest])
            band_indices.append(idx)

        bars = np.zeros((n_frames, n_bins), dtype=np.float32)

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

        bars = np.log1p(bars)
        ref = np.median(bars[: min(ref_median_frames, len(bars))], axis=0) + eps
        bars = (bars - ref[None, :])
        bars = np.clip(bars, 0.0, None)
        if np.max(bars) > eps:
            bars /= (np.max(bars) + eps)

        alpha = 0.35
        for b in range(n_bins):
            acc = 0.0
            for fi in range(n_frames):
                acc = alpha * bars[fi, b] + (1 - alpha) * acc
                bars[fi, b] = max(bars[fi, b], acc)

        return bars

    @staticmethod
    def build_bandplan(sr: int, n_bins: int, f_min: float = 20.0, f_max_limit: float = 16000.0):
        f_max = min(sr / 2.0, f_max_limit)
        edges = np.geomspace(f_min, f_max, n_bins + 1)
        centers = np.sqrt(edges[:-1] * edges[1:])
        return edges.astype(np.float32), centers.astype(np.float32)

    # ---------------------------
    # ВИДЕО / ИЗОБРАЖЕНИЕ
    # ---------------------------

    def load_video(self, path: str):
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise RuntimeError(f"Не удалось открыть видео: {path}")

        self.cap = cap
        self.video_path = path
        self.image_path = None
        self.still_image_bgr = None

        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.fps = fps if fps > 1e-3 else self.fps_default
        self.n_video_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def load_image(self, path: str):
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Не удалось открыть изображение: {path}")

        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

        self.still_image_bgr = img
        self.image_path = path
        self.video_path = None
        self.n_video_frames = 0

    def get_background_frame(self, frame_index: int) -> np.ndarray:
        if self.cap is not None:
            if frame_index < 0:
                frame_index = 0
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame_bgr = self.cap.read()
            if not ret:
                raise RuntimeError(f"Не удалось прочитать кадр #{frame_index} из видео.")
            return frame_bgr

        if self.still_image_bgr is not None:
            return self.still_image_bgr.copy()

        raise RuntimeError("Фон не загружен (нет ни видео, ни картинки).")

    # ---------------------------
    # АУДИО + БАРЫ
    # ---------------------------

    def load_audio_wav(self, path: str):
        if not path.lower().endswith(".wav"):
            raise ValueError("Поддерживается только WAV (.wav).")

        audio, sr = self.read_wav_mono(path)

        fps = self.fps if (self.cap is not None or self.still_image_bgr is not None) else self.fps_default

        bars = self.make_bar_features(audio, sr, fps=fps, n_bins=self.n_bins)
        edges, centers = self.build_bandplan(sr, self.n_bins)

        self.audio_path = path
        self.bars = bars
        self.band_edges = edges
        self.band_centers = centers

    def get_bar_values_for_time(self, t_sec: float, wrap: bool = True):
        """
        Возвращает вектор баров для момента времени t_sec (секунды).
        Если wrap = True — индекс зацикливается по длине self.bars.
        """
        if self.bars is None or len(self.bars) == 0:
            return None
        idx = int(t_sec * self.fps)
        if wrap:
            idx = idx % len(self.bars)
        else:
            idx = max(0, min(len(self.bars) - 1, idx))
        return self.bars[idx]

    def get_bar_values_for_frame(self, frame_index: int, wrap: bool = True):
        """
        Возвращает вектор баров по номеру кадра (frame_index).
        """
        if self.bars is None or len(self.bars) == 0:
            return None
        if wrap:
            idx = frame_index % len(self.bars)
        else:
            idx = max(0, min(len(self.bars) - 1, frame_index))
        return self.bars[idx]

    # ---------------------------
    # FFMPEG
    # ---------------------------

    @staticmethod
    def _candidate_ffmpeg_paths():
        return [
            "/opt/homebrew/bin/ffmpeg",
            "/usr/local/bin/ffmpeg",
            "/opt/local/bin/ffmpeg",
            "/usr/bin/ffmpeg",
            "C:/ffmpeg/bin/ffmpeg.exe",
        ]

    def find_ffmpeg(self) -> str | None:
        p = shutil.which("ffmpeg")
        if p:
            return p

        for cand in self._candidate_ffmpeg_paths():
            if os.path.isfile(cand) and os.access(cand, os.X_OK):
                return cand

        try:
            out = subprocess.run(
                ["which", "ffmpeg"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False
            )
            cand = out.stdout.decode().strip()
            if cand and os.path.isfile(cand) and os.access(cand, os.X_OK):
                return cand
        except Exception:
            pass

        return None

    # ---------------------------
    # ЭКСПОРТ
    # ---------------------------

    def estimate_export_frame_count(self) -> int:
        """
        Оценивает, сколько кадров будет отрендерено при экспорте.
        Это чистая логика движка (main просто покажет прогресс).
        """
        if self.cap is not None and self.n_video_frames > 0:
            return self.n_video_frames
        if self.bars is not None:
            return len(self.bars)
        raise RuntimeError("Нечего экспортировать: нет видео/картинки или баров.")

    def export_mp4(
        self,
        out_path: str,
        ffmpeg_bin: str | None = None,
        loop_bars: bool = True,
        progress_callback=None,
        overlay_callback=None,
        video_codec: str = "mp4v",
        audio_bitrate: str = "192k",
    ):
        if (self.cap is None) and (self.still_image_bgr is None):
            raise RuntimeError("Нечего рендерить: не загружено ни видео, ни изображение.")
        if self.audio_path is None or self.bars is None:
            raise RuntimeError("Нет аудио или не рассчитаны бары (self.bars).")

        if ffmpeg_bin is None:
            ffmpeg_bin = self.find_ffmpeg()
            if ffmpeg_bin is None:
                raise RuntimeError("ffmpeg не найден. Укажите путь явно через ffmpeg_bin.")

        # размеры/кадры
        if self.cap is not None:
            w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = float(self.fps)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            n_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        else:
            h, w = self.still_image_bgr.shape[:2]
            fps = float(self.fps)
            n_frames = int(len(self.bars))

        tmpdir = tempfile.mkdtemp(prefix="va_export_")
        tmp_video = os.path.join(tmpdir, "video_only.mp4")

        fourcc = cv2.VideoWriter_fourcc(*video_codec)
        vw = cv2.VideoWriter(tmp_video, fourcc, fps, (w, h))
        if not vw.isOpened():
            shutil.rmtree(tmpdir, ignore_errors=True)
            raise RuntimeError("Не удалось открыть VideoWriter для MP4.")

        try:
            if self.cap is not None:
                for i in range(n_frames):
                    ret, frame_bgr = self.cap.read()
                    if not ret:
                        break

                    bars_vec = self.get_bar_values_for_frame(i, wrap=loop_bars)

                    if overlay_callback is not None:
                        frame_bgr_out = overlay_callback(frame_bgr, i, bars_vec)
                    else:
                        frame_bgr_out = frame_bgr

                    vw.write(frame_bgr_out)

                    if progress_callback is not None:
                        progress_callback(i + 1, n_frames)
            else:
                base_bgr = self.still_image_bgr.copy()
                for i in range(n_frames):
                    frame_bgr = base_bgr.copy()
                    bars_vec = self.get_bar_values_for_frame(i, wrap=loop_bars)

                    if overlay_callback is not None:
                        frame_bgr_out = overlay_callback(frame_bgr, i, bars_vec)
                    else:
                        frame_bgr_out = frame_bgr

                    vw.write(frame_bgr_out)

                    if progress_callback is not None:
                        progress_callback(i + 1, n_frames)
        finally:
            vw.release()

        cmd = [
            ffmpeg_bin, "-y",
            "-i", tmp_video,
            "-i", self.audio_path,
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", audio_bitrate,
            "-shortest",
            out_path
        ]

        try:
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as e:
            err = e.stderr.decode(errors='ignore')
            raise RuntimeError(f"ffmpeg не смог собрать видео с аудио:\n{err[:2000]}") from e
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
