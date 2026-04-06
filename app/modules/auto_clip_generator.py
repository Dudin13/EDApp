# auto_clip_generator.py

import cv2
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict

class AutoClipGenerator:
    """Genera clips automáticamente basados en eventos detectados."""

    def __init__(self, clips_dir: Path, ffmpeg_path: str):
        self.clips_dir = clips_dir
        self.ffmpeg_path = ffmpeg_path
        self.clips_dir.mkdir(exist_ok=True)

        # Configuración de clips por evento
        self.clip_configs = {
            "Gol": {"before": 5, "after": 3, "quality": "high"},
            "Corner": {"before": 3, "after": 5, "quality": "medium"},
            "Tiro a puerta": {"before": 2, "after": 4, "quality": "medium"},
            "Falta": {"before": 3, "after": 3, "quality": "low"},
            "Pase clave": {"before": 2, "after": 3, "quality": "low"}
        }

    def generate_clips_from_events(self, video_path: str, events: List[Dict],
                                 progress_callback=None) -> List[str]:
        """Genera clips para todos los eventos detectados."""
        generated_clips = []

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            for i, event in enumerate(events):
                future = executor.submit(
                    self._generate_single_clip,
                    video_path, event, i, len(events)
                )
                futures.append(future)

            for future in futures:
                result = future.result()
                if result:
                    generated_clips.append(result)
                    if progress_callback:
                        progress_callback(len(generated_clips), len(events))

        return generated_clips

    def _generate_single_clip(self, video_path: str, event: Dict,
                            index: int, total: int) -> str:
        """Genera un clip individual para un evento."""
        event_type = event.get("action", "Evento")
        timestamp = event.get("timestamp", 0)

        config = self.clip_configs.get(event_type, {"before": 3, "after": 3, "quality": "medium"})

        start_time = max(0, timestamp - config["before"])
        duration = config["before"] + config["after"]

        clip_filename = f"auto_{event_type.lower()}_{index:03d}_{timestamp:.1f}s.mp4"
        output_path = self.clips_dir / clip_filename

        # Generar clip con FFmpeg
        cmd = [
            self.ffmpeg_path, "-y",
            "-i", video_path,
            "-ss", str(start_time),
            "-t", str(duration),
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "23",
            str(output_path)
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            if result.returncode == 0 and output_path.exists():
                return str(output_path)
        except Exception as e:
            print(f"Error generando clip {clip_filename}: {e}")

        return None