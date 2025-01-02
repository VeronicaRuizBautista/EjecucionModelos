import torch
from PIL import Image
from transformers import AutoModelForImageClassification
from torchvision import transforms

# Cargar modelo desde Hugging Face
model_name = "google/vit-base-patch16-224-in21k"
model = AutoModelForImageClassification.from_pretrained(model_name)

#Cargar el modelo local
# ejecutar 
# transformers-cli download google/vit-base-patch16-224-in21k
# model_path = "ruta_local_al_modelo"  # Ruta donde descargaste el modelo


# Cargar la imagen
image_path = "nemo.jpg" 
image = Image.open(image_path)

# Preprocesamiento de la imagen para el modelo
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Aplicar transformaciones
image_tensor = preprocess(image).unsqueeze(0)

# Hacer predicción
with torch.no_grad():
    outputs = model(image_tensor)

# Obtener la clase con la mayor probabilidad
logits = outputs.logits
predicted_class_idx = torch.argmax(logits, dim=1).item()

# Mostrar el resultado
print(f"Predicted class index: {predicted_class_idx}")
import requests

# Descargar las etiquetas de ImageNet
url = "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json"
response = requests.get(url)
class_idx = response.json()

# Obtener el nombre de la clase correspondiente
predicted_class_name = class_idx[str(predicted_class_idx)][1]
print(f"Predicted class: {predicted_class_name}")
