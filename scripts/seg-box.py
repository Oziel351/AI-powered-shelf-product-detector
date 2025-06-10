import os

def check_for_segments(label_dir):
    seg_files = []
    for filename in os.listdir(label_dir):
        if filename.endswith('.txt'):
            with open(os.path.join(label_dir, filename), 'r') as f:
                lines = f.readlines()
                for line in lines:
                    values = line.strip().split()
                    if len(values) > 5:
                        seg_files.append(filename)
                        break
    return seg_files

train_labels = "C:/Users/omont/Desktop/IA-Counter/dataset-counter-products/train/labels"
val_labels = "C:/Users/omont/Desktop/IA-Counter/dataset-counter-products/valid/labels"

train_seg = check_for_segments(train_labels)
val_seg = check_for_segments(val_labels)

print("Archivos con segmentación en train:", train_seg)
print("Archivos con segmentación en valid:", val_seg)
