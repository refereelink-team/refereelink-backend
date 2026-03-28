from pathlib import Path


APP_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT_DIR = APP_DIR.parent
ASSETS_DIR = REPO_ROOT_DIR / 'assets'
DATA_DIR = ASSETS_DIR / 'data'
WEIGHTS_DIR = ASSETS_DIR / 'weights'

PLAYER_DETECTION_MODEL_PATH = str(WEIGHTS_DIR / 'football-player-detection.pt')
PITCH_DETECTION_MODEL_PATH = str(WEIGHTS_DIR / 'football-pitch-detection.pt')
BALL_DETECTION_MODEL_PATH = str(WEIGHTS_DIR / 'football-ball-detection.pt')
