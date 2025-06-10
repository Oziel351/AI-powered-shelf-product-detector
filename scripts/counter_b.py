from ultralytics import YOLO 
import cv2
from collections import defaultdict
import os

# Ruta de imagen y modelo
image_path = "productos-estante-4.jpg"
#image_path = "productos-estante-5.jpg"
model_path = "runs/detect/train26/weights/best.pt" 

if not os.path.exists(image_path):
    print(f"No se encontró la imagen: {image_path}")
    exit()

if not os.path.exists(model_path):
    print(f"No se encontró el modelo: {model_path}")
    exit()

# Cargar modelo
model = YOLO(model_path)

# Lista de clases según tu dataset
class_names = ['botella', 'lata', 'caja']

# Leer imagen
img = cv2.imread(image_path)
if img is None:
    print("Error al cargar la imagen.")
    exit()

# Inferencia
#imagen estante 4
results = model(img, conf=0.02)
#imagen estante 5
#results = model(img, conf=0.005)


print("\nDetecciones brutas:")
detections = results[0].boxes
if detections is None or len(detections) == 0:
    print("No se detectaron objetos.")
else:
    for box in detections:
        cls_id = int(box.cls)
        conf = box.conf.item()
        xyxy = box.xyxy.numpy().astype(int)
        print(f"Clase: {class_names[cls_id]}, Confianza: {conf:.2f}, BBox: {xyxy.tolist()}")

# Conteo de clases detectadas
counter = defaultdict(int)
for box in detections:
    class_id = int(box.cls)
    class_name = class_names[class_id]
    counter[class_name] += 1

print("\nConteo de productos por clase:")
if counter:
    for cls, count in counter.items():
        print(f"{cls}: {count}")
else:
    print("No se detectó ninguna clase válida.")

#Estimacion si se conoce la profundidad física del estante, en otro caso setear un valor por defecto
profundidad_estante_cm = 30  

# Profundidad por producto en cm
profundidad_producto_cm = {
    'botella': 10,
    'lata': 7,
    'caja': 15
}

print("\nEstimación total de productos considerando profundidad física:")
if counter:
    for cls, count in counter.items():
        prof_prod = profundidad_producto_cm.get(cls, 10)  
        filas_en_profundidad = max(1, profundidad_estante_cm // prof_prod)
        estimado_total = count * filas_en_profundidad
        print(f"{cls}: detectados {count}, profundidad estante: {profundidad_estante_cm}cm, "
              f"el producto ocupa: {prof_prod}cm → estimado total: {estimado_total}")
else:
    print("No se detectó ninguna clase válida.")

annotated_img = results[0].plot()

try:
    cv2.imshow("Detecciones", annotated_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
except:
    print("No se puede mostrar la imagen (probablemente sin entorno gráfico).")

output_dir = "pruebas"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, f"output_{os.path.basename(image_path)}")
cv2.imwrite(output_path, annotated_img)
print(f"Imagen guardada como {output_path}")
