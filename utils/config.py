"""
utils/config.py
============================================================
Configuration loader for perception-bev.

Loads params.yaml from the config/ directory using an absolute
path derived from this file's location — works regardless of
the working directory from which the script is launched.

Usage:
    from utils.config import load_config

    cfg = load_config()                      # default params.yaml
    cfg = load_config("my_custom.yaml")      # custom config
"""
import yaml
from pathlib import Path

_PROJECT_ROOT_ = Path(__file__).parent.parent

def load_config(path = None) -> dict:


    if path is None:
        cfg_path = _PROJECT_ROOT_/ "config" /"params.yaml"
    else:
        cfg_path = Path(path)

    if not cfg_path.exists():
        FileNotFoundError(f"Config file not find : {cfg_path}")

    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
        
    return cfg