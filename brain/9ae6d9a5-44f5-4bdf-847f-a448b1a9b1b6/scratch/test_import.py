import sys
from pathlib import Path
ROOT = Path("C:/apped")
sys.path.insert(0, str(ROOT))
try:
    from core.config.settings import settings
    print(f"Import success! Team A anchor: {settings.TEAM_A_COLOR_HSV}")
except Exception as e:
    print(f"Import failed: {e}")
