"""Camada de câmera: uma interface única para todos os tipos.

Trocar de câmera = mudar `camera.type` no config.yaml:

    simulator  esteira simulada com imagens do dataset (sem hardware)
    usb        webcam / câmera UVC                     (OpenCV)
    rtsp       câmera IP (RTSP/ONVIF)                  (OpenCV + FFmpeg)
    genicam    câmera industrial GigE Vision / USB3    (pacote `harvesters` + .cti do fabricante)
    video      arquivo de vídeo (teste/reprodução)
    images     pasta de imagens (teste/reprodução)

Todas as câmeras "ao vivo" rodam numa thread que sempre guarda o quadro
mais recente (sem acumular atraso) e reconectam sozinhas se cair.
"""
from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from .config import IMG_EXTS, Config, resolve

log = logging.getLogger("camera")


class Camera:
    """Interface: open() / read() -> (ok, frame BGR) / close()."""

    name = "base"

    def open(self) -> None: ...

    def read(self) -> tuple[bool, np.ndarray | None]:
        raise NotImplementedError

    def close(self) -> None: ...

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *exc):
        self.close()


# ------------------------------------------------------------------ OpenCV
class OpenCVCamera(Camera):
    """USB e RTSP via cv2.VideoCapture."""

    def __init__(self, source, backend: int, width=None, height=None, fps=None, name="opencv"):
        self.source, self.backend, self.name = source, backend, name
        self.width, self.height, self.fps = width, height, fps
        self.cap: cv2.VideoCapture | None = None

    def open(self) -> None:
        self.cap = cv2.VideoCapture(self.source, self.backend)
        if self.width:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        if self.height:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        if self.fps:
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self.cap.isOpened():
            raise ConnectionError(f"Não foi possível abrir a câmera {self.name}: {self.source}")

    def read(self):
        if self.cap is None:
            return False, None
        return self.cap.read()

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None


# ------------------------------------------------------------------ GenICam
class GenICamCamera(Camera):
    """Câmeras industriais (Basler, Hikrobot, FLIR, Allied Vision, ...) via
    GenICam/GenTL. Requer `pip install harvesters` e o arquivo .cti do SDK
    do fabricante (ex.: pylon da Basler, MVS da Hikrobot)."""

    name = "genicam"

    def __init__(self, cti: str, serial: str = "", exposure_us: float | None = None,
                 width: int | None = None, height: int | None = None):
        self.cti, self.serial, self.exposure_us = cti, serial, exposure_us
        self.width, self.height = width, height
        self.h = self.ia = None

    def open(self) -> None:
        try:
            from harvesters.core import Harvester
        except ImportError as e:
            raise ImportError("Instale o suporte GenICam:  uv pip install harvesters") from e
        if not Path(self.cti).exists():
            raise FileNotFoundError(f"Arquivo GenTL (.cti) não encontrado: {self.cti}")
        self.h = Harvester()
        self.h.add_file(self.cti)
        self.h.update()
        if not self.h.device_info_list:
            raise ConnectionError("Nenhuma câmera GenICam encontrada na rede/USB.")
        sel = {"serial_number": self.serial} if self.serial else 0
        self.ia = self.h.create(sel)
        nm = self.ia.remote_device.node_map
        for node, val in (("Width", self.width), ("Height", self.height), ("ExposureTime", self.exposure_us)):
            if val:
                try:
                    getattr(nm, node).value = val
                except Exception as e:  # nó pode não existir neste modelo
                    log.warning("GenICam: não foi possível ajustar %s: %s", node, e)
        try:
            nm.PixelFormat.value = "BGR8"
        except Exception:
            pass  # usa o formato padrão e convertemos abaixo
        self.ia.start()

    def read(self):
        if self.ia is None:
            return False, None
        with self.ia.fetch(timeout=3.0) as buf:
            comp = buf.payload.components[0]
            fmt = comp.data_format
            img = comp.data.reshape(comp.height, comp.width, -1).copy()
        if img.shape[2] == 1:
            code = {"BayerRG8": cv2.COLOR_BayerRG2BGR, "BayerBG8": cv2.COLOR_BayerBG2BGR,
                    "BayerGR8": cv2.COLOR_BayerGR2BGR, "BayerGB8": cv2.COLOR_BayerGB2BGR}.get(fmt, cv2.COLOR_GRAY2BGR)
            img = cv2.cvtColor(img, code)
        elif fmt.startswith("RGB"):
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return True, img

    def close(self) -> None:
        if self.ia is not None:
            self.ia.stop()
            self.ia.destroy()
            self.ia = None
        if self.h is not None:
            self.h.reset()
            self.h = None


# ------------------------------------------------------------------ arquivos
class VideoFileCamera(OpenCVCamera):
    def __init__(self, path: str, loop: bool = True):
        super().__init__(str(resolve(path)), cv2.CAP_ANY, name="video")
        self.loop = loop

    def read(self):
        ok, frame = super().read()
        if not ok and self.loop and self.cap is not None:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.cap.read()
        return ok, frame


class ImageFolderCamera(Camera):
    name = "images"

    def __init__(self, folder: str, loop: bool = True):
        self.files = sorted(p for p in resolve(folder).rglob("*") if p.suffix.lower() in IMG_EXTS)
        self.loop, self.i = loop, 0

    def read(self):
        from .data import imread
        if not self.files:
            return False, None
        if self.i >= len(self.files):
            if not self.loop:
                return False, None
            self.i = 0
        img = imread(self.files[self.i])
        self.i += 1
        return img is not None, img


# ------------------------------------------------------------------ thread
class ThreadedCamera(Camera):
    """Lê a câmera numa thread, mantém só o quadro mais novo e reconecta."""

    def __init__(self, cam: Camera, reconnect_s: float = 2.0):
        self.cam, self.reconnect_s = cam, reconnect_s
        self.name = cam.name
        self._frame = None
        self._seq = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.connected = False

    def open(self) -> None:
        self.cam.open()
        self.connected = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while not self._stop.is_set():
            try:
                ok, frame = self.cam.read()
            except Exception as e:
                ok, frame = False, None
                log.warning("Erro lendo câmera %s: %s", self.name, e)
            if ok and frame is not None:
                with self._lock:
                    self._frame, self._seq = frame, self._seq + 1
                continue
            self.connected = False
            log.warning("Câmera %s sem sinal, reconectando em %.0fs...", self.name, self.reconnect_s)
            self.cam.close()
            time.sleep(self.reconnect_s)
            try:
                self.cam.open()
                self.connected = True
                log.info("Câmera %s reconectada.", self.name)
            except Exception as e:
                log.warning("Falha ao reconectar: %s", e)

    def read(self):
        with self._lock:
            return (self._frame is not None), (None if self._frame is None else self._frame.copy())

    def read_new(self, last_seq: int, timeout: float = 1.0):
        """Espera um quadro mais novo que last_seq. Retorna (seq, frame)."""
        t0 = time.time()
        while time.time() - t0 < timeout:
            with self._lock:
                if self._seq != last_seq and self._frame is not None:
                    return self._seq, self._frame.copy()
            time.sleep(0.002)
        return last_seq, None

    def close(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)
        self.cam.close()


# ------------------------------------------------------------------ fábrica
def create_camera(cfg: Config, threaded: bool | None = None) -> Camera:
    c = cfg.camera
    t = c.type.lower()
    if t == "simulator":
        from .simulator import ConveyorSimulator
        s = c.simulator
        cam: Camera = ConveyorSimulator(resolve(cfg.data.prepared) / s.source_split, s.width, s.height,
                                        s.object_px, s.foreign_rate, s.speed_px,
                                        set(cfg.normal_classes), fps=c.fps)
        return cam  # gera sob demanda, não precisa de thread
    if t == "usb":
        cam = OpenCVCamera(int(c.usb.index), cv2.CAP_DSHOW, c.width, c.height, c.fps, name="usb")
    elif t == "rtsp":
        cam = OpenCVCamera(c.rtsp.url, cv2.CAP_FFMPEG, name="rtsp")
    elif t == "genicam":
        g = c.genicam
        cam = GenICamCamera(g.cti, g.get("serial", ""), g.get("exposure_us"), c.width, c.height)
    elif t == "video":
        return VideoFileCamera(c.video.path, c.video.get("loop", True))
    elif t == "images":
        return ImageFolderCamera(c.images.folder)
    else:
        raise ValueError(f"camera.type desconhecido: {c.type}")
    return ThreadedCamera(cam) if threaded in (None, True) else cam
