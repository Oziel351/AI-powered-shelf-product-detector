import os
import shutil

base_dir = "runs/detect"

for name in os.listdir(base_dir):
    train_dir = os.path.join(base_dir, name)
    weights_path = os.path.join(train_dir, "weights", "best.pt")
    if os.path.isdir(train_dir) and name.startswith("train"):
        if not os.path.exists(weights_path):
            print(f"JKDFHDS {train_dir} (no tiene best.pt)")
            shutil.rmtree(train_dir)