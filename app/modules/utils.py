import base64
from pathlib import Path

def get_logo_base64():
    """Reads the logo.png and returns it as base64 string."""
    base_dir = Path(__file__).parent.parent
    logo_path = base_dir / "logo.png"
    if logo_path.exists():
        with open(logo_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return None
