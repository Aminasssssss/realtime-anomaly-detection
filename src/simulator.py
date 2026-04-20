from __future__ import annotations

import time
import logging
import threading
from collections import deque
from datetime import datetime, timezone
from typing import Generator, Optional

import numpy as np

from src.config import WINDOW_SIZE, STEP_SIZE, STREAM_INTERVAL_SEC, SMAP_CHANNEL
from src.data_loader import load_entity, normalize

logger = logging.getLogger(__name__)

class SensorStreamSimulator:
    def __init__(self, entity=SMAP_CHANNEL, interval_sec=STREAM_INTERVAL_SEC, loop=True):
        self.entity = entity
        self.interval_sec = interval_sec
        self.loop = loop

        raw_data = load_entity(entity, split="test")
        train_data = load_entity(entity, split="train")

        _, _, mean, std = normalize(train_data)
        self.data = (raw_data - mean) / (std + 1e-10)
        self.raw_data = raw_data
        self.D = self.data.shape[1]
        self.T = self.data.shape[0]

        self._position = 0
        self._tick = 0
        self._lock = threading.Lock()
        self._history: deque = deque(maxlen=500)

    @property
    def position(self):
        return self._position

    @property
    def tick(self):
        return self._tick

    def _next_window(self):
        with self._lock:
            start = self._position
            end = start + WINDOW_SIZE

            if end > self.T:
                if self.loop:
                    self._position = 0
                    start = 0
                    end = WINDOW_SIZE
                else:
                    return None

            window = self.data[start:end]
            raw_window = self.raw_data[start:end]

            metadata = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "tick": self._tick,
                "position_start": int(start),
                "position_end": int(end),
                "progress": round(start / self.T, 3),
                "primary_channel_last": float(raw_window[-1, 0]),
                "primary_channel_mean": float(raw_window[:, 0].mean()),
                "primary_channel_std": float(raw_window[:, 0].std()),
            }

            self._position += STEP_SIZE
            self._tick += 1
            self._history.append((window.copy(), metadata.copy()))

            return window, metadata

    def stream(self) -> Generator[tuple[np.ndarray, dict], None, None]:
        while True:
            tick_start = time.monotonic()
            result = self._next_window()
            if result is None:
                break
            yield result
            elapsed = time.monotonic() - tick_start
            time.sleep(max(0.0, self.interval_sec - elapsed))

    def get_history(self, last_n=200):
        with self._lock:
            return list(self._history)[-last_n:]

    def reset(self):
        with self._lock:
            self._position = 0
            self._tick = 0
            self._history.clear()
