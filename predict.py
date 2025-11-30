import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import sys

MODEL_PATH = "mobilenet_kidney_classifier.pth"   # your trained model


# ----------------------------------------------------
# 1. Define the SAME model architecture used in training
# ----------------------------------------------------
class MobileNetKidneyClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super(MobileNetKidneyClassifier, self).__init__()
        self.model = models.mobilenet_v2(weights=None)   # IMPORTANT: weights=None
        num_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.model(x)


# ----------------------------------------------------
# 2. Image Preprocessing
# ----------------------------------------------------
def preprocess(img_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    img = Image.open(img_path).convert("RGB")
    img = transform(img).unsqueeze(0)
    return img


# ----------------------------------------------------
# 3. Prediction Function
# ----------------------------------------------------
def predict(img_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = MobileNetKidneyClassifier()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    # Load image
    img_tensor = preprocess(img_path).to(device)

    # Predict
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    classes = ["Normal Kidney", "Stone Detected"]

    return classes[predicted.item()], confidence.item() * 100


# ----------------------------------------------------
# 4. Run from Command Line
# ----------------------------------------------------
if __name__ == "__main__":
    img_path = sys.argv[1]
    label, confidence = predict(img_path)

    print("\n==============================")
    print(f" Prediction: {label}")
    print(f" Confidence: {confidence:.2f}%")
    print("==============================\n")
