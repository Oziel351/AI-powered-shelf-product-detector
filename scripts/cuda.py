import torch

print("Torch CUDA version:", torch.version.cuda)       # debería salir "12.6"
print("cuDNN version:", torch.backends.cudnn.version()) 
print("CUDA available:", torch.cuda.is_available())    # debería ser True
print("Número de GPUs:", torch.cuda.device_count())     # >= 1
if torch.cuda.is_available():
    print("Nombre de GPU:", torch.cuda.get_device_name(0))
