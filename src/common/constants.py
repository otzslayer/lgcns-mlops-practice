import os
import pathlib

# 경로
LOG_FILEPATH: str = os.path.join(
    pathlib.Path(__file__).parent.parent.parent.absolute(), "logs"
)
DATA_PATH: str = os.path.join(
    pathlib.Path(__file__).parent.parent.parent.absolute(), "data"
)
ARTIFACT_PATH: str = os.path.join(
    pathlib.Path(__file__).parent.parent.parent.absolute(), "artifacts"
)

if not os.path.exists(LOG_FILEPATH):
   os.makedirs(LOG_FILEPATH)

if not os.path.exists(ARTIFACT_PATH):
   os.makedirs(ARTIFACT_PATH)