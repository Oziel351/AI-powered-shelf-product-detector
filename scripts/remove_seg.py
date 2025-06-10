import os

files_to_delete = [
    "C:/Users/omont/Desktop/IA-Counter/dataset-counter-products/train/labels/lata02_png.rf.3883cdc8b75fd21ca25a9a3f02966900.txt",
    "C:/Users/omont/Desktop/IA-Counter/dataset-counter-products/train/labels/lata02_png.rf.6794f1e86ef28ecb43ba6ffa3c0acf08.txt",
    "C:/Users/omont/Desktop/IA-Counter/dataset-counter-products/valid/labels/descarga_png.rf.5b0ab326247968327d6eb78bfdbf7b01.txt"
]

for path in files_to_delete:
    if os.path.exists(path):
        os.remove(path)
        print(f"Eliminado: {path}")
    else:
        print(f"No encontrado: {path}")
