from flask import Flask, render_template, request
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

app = Flask(__name__)

# ------------ MODEL DEFINITION ------------
class MobileNetKidneyClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super(MobileNetKidneyClassifier, self).__init__()
        self.model = models.mobilenet_v2(weights=None)
        num_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.model(x)


# ------------ LOAD MODEL ------------
device = torch.device("cpu")
model = MobileNetKidneyClassifier()

model_path = "mobilenet_stone_detector.pth"   # <--- FIXED NAME

if not os.path.exists(model_path):
    print(f"\n❌ Model file NOT found at: {model_path}")
    print("Place your .pth file in the project folder.\n")
    exit()

model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()


# ------------ IMAGE TRANSFORM ------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])


def predict_image(img_path):
    img = Image.open(img_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, pred = torch.max(probs, 1)

    classes = ["Normal Kidney", "Stone Detected"]
    return classes[pred.item()], round(confidence.item() * 100, 2)


# ------------ ROUTES ------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["image"]

    if file:
        file_path = os.path.join("static", "uploaded.jpg")
        file.save(file_path)

        label, conf = predict_image(file_path)

        return render_template("result.html",
                               prediction=label,
                               confidence=conf,
                               img_path=file_path)
    else:
        return "No file uploaded!"


if __name__ == "__main__":
    app.run(debug=True)
