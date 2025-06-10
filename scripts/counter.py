from inference import get_model
import supervision as sv
import cv2
import requests
import numpy as np

# URL de la imagen para inferencia
image_url = "https://l450v.alamy.com/450vfr/2j2acfg/novi-sad-serbie-14-mars-2022-bouteilles-et-canettes-de-differentes-marques-de-biere-sur-le-rayon-de-supermarche-idea-a-novi-sad-editorial-illustratif-2j2acfg.jpg"

# Descargar la imagen y convertirla a formato OpenCV
resp = requests.get(image_url)
image_np = np.asarray(bytearray(resp.content), dtype=np.uint8)
image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

# Cargar modelo con tu API key
model = get_model(model_id="dataset-counter-products/3", api_key="3jQAu52F9uiAV5Wsw0GJ")

# Inferir usando la URL
results = model.infer(image_url, conf=0.0001)[0]

# Crear detecciones para supervision
detections = sv.Detections.from_inference(results)

# Contar productos detectados
num_productos = len(detections)
print(f"Cantidad de productos detectados: {num_productos}")

# Crear anotadores
bounding_box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

# Anotar la imagen
annotated_image = bounding_box_annotator.annotate(scene=image, detections=detections)
annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)

# Mostrar imagen anotada
sv.plot_image(annotated_image)
