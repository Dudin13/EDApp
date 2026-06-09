import pytest
import json
import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

def test_events_structure():
    events_path = Path(_REPO_ROOT) / "output" / "events.json"
    if not events_path.exists():
        pytest.skip(f"No se encontró el archivo de ejemplo {events_path}")
        
    with open(events_path, "r", encoding="utf-8") as f:
        events = json.load(f)
        
    assert isinstance(events, dict), "events.json debe contener un diccionario principal"
    assert "events" in events, "El diccionario debe tener una clave 'events'"
    
    events_list = events["events"]
    assert isinstance(events_list, list), "'events' debe ser una lista"
    
    if len(events_list) > 0:
        ev = events_list[0]
        # Verifica estructura de campos principales
        assert "minute" in ev or "timestamp" in ev or "second" in ev, "Falta campo de tiempo en el evento"
        assert "action" in ev or "type" in ev, "Falta el tipo de accion en el evento"
        assert "equipo" in ev or "team" in ev, "Falta el equipo en el evento"
        assert "track_id" in ev or "from_tid" in ev, "Falta el track_id en el evento"
