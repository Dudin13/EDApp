import pytest
import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from core.config_loader import config

def test_config_loads():
    # Verifica que todos los campos principales existen
    assert hasattr(config, "detection")
    assert hasattr(config, "tracking")
    assert hasattr(config, "team_classifier")
    assert hasattr(config, "events")
    assert hasattr(config, "reporting")

    # Verifica campos de detection
    assert hasattr(config.detection, "conf_threshold")
    assert hasattr(config.detection, "min_height_px")
    assert hasattr(config.detection, "imgsz_players")
    assert hasattr(config.detection, "imgsz_ball")
    assert hasattr(config.detection, "models")
    assert hasattr(config.detection.models, "players")
    assert hasattr(config.detection.models, "ball")
    assert hasattr(config.detection.models, "pitch")

    # Verifica tracking
    assert hasattr(config.tracking, "engine")
    assert hasattr(config.tracking, "lost_track_buffer")
    assert hasattr(config.tracking, "gta_link")

    # Verifica team_classifier
    assert hasattr(config.team_classifier, "engine")
    assert hasattr(config.team_classifier, "k")
    assert hasattr(config.team_classifier, "grass_hue")

    # Verifica events
    assert hasattr(config.events, "ball_proximity_duel")
    assert hasattr(config.events, "min_ball_speed")
    assert hasattr(config.events, "event_cooldown")
    assert hasattr(config.events, "shot_speed")

    # Verifica reporting
    assert hasattr(config.reporting, "llm_provider")
    assert hasattr(config.reporting, "model")
