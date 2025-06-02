import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, send_from_directory
import torch
import torch.nn as nn
import torchvision.transforms as transforms

app = Flask(__name__)

class_names = ['Avião', 'Formiga', 'Maçã', 'Machado', 'Bola de basquete', 'Abelha', 'Bicicleta', 'Pássaro', 'Gato', 'Círculo',
               'Relógio', 'Cachorro', 'Rosquinha', 'Torre Eiffel', 'Flor', 'Casa', 'Rosto sorridente', 'Caracol', 'Quadrado', 'Árvore']

class Net(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Net(num_classes=len(class_names)).to(device)
model.load_state_dict(torch.load("modelo_pytorch.pt", map_location=device))
model.eval()

@app.route('/')
def index():
    return send_from_directory('.', 'templates/index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "Nenhuma imagem enviada"}), 400

    image_file = request.files['image']
    img = Image.open(image_file).convert("L").resize((28, 28), Image.LANCZOS)
    img_array = np.array(img).reshape(1, 784).astype(np.float32) / 255.0

    if np.all(img_array == 0):
        return jsonify({"error": "Imagem vazia"}), 400

    with torch.no_grad():
        input_tensor = torch.tensor(img_array).to(device)
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy().flatten()
    
    results = [
        {"class": name, "probability": float(prob)}
        for name, prob in zip(class_names, probs)
    ]

    return jsonify({"predictions": results})

if __name__ == '__main__':
    app.run(debug=True)