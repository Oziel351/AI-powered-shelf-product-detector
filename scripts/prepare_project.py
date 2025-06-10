#!/usr/bin/env python3
import os
import yaml
from roboflow import Roboflow

def main():
    cfg = yaml.safe_load(open("config/config.yaml"))
    rf = Roboflow(api_key=cfg["api_key"])
    project = rf.workspace(cfg["workspace"]).project(cfg["dataset"])
    dataset = project.version(cfg["version"]).download("yolov8")

    src = dataset.location
    dst = os.path.abspath( cfg["dataset"])
    if os.path.exists(dst):
        print(f"{dst} ya existe, se omite descarga.")
    else:
        os.rename(src, dst)
        print(f"✅ Dataset listo en {dst}")

if __name__ == "__main__":
    main()
