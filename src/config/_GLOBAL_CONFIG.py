import os

current_file = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file)

ROOT_DIR = os.path.join(current_dir, "../../")

CONFIG_DIR = os.path.join(ROOT_DIR, "config")
DATA_PATH = os.path.join(ROOT_DIR, "data")
THIRD_PARTY_PATH = os.path.join(ROOT_DIR, "third_party")
LOG_PATH = os.path.join(ROOT_DIR, "logs")