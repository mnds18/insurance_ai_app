import torch
import os
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import random

# Define paths
TRAIN_DIR = "data/vision/train"
MODEL_PATH = "models/custom_damage_classifier.pth"
NUM_EPOCHS = 5
BATCH_SIZE = 16
NUM_CLASSES = 5  # adjust if needed

# Define transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Load training dataset
train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
class_names = train_dataset.classes

# Define model
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train_model():
    print("🚀 Training Started...")
    model.train()
    for epoch in range(NUM_EPOCHS):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {running_loss/len(train_loader):.4f}")

    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'class_names': class_names
    }, MODEL_PATH)
    print(f"✅ Model saved to: {MODEL_PATH}")

def classify_image(image_path):
    # Load model for inference
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    class_names = checkpoint['class_names']

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        predicted_idx = torch.argmax(output, dim=1).item()
        predicted_label = class_names[predicted_idx]

    return predicted_label

# Run training if called directly
if __name__ == "__main__":
    train_model()

    # Demo prediction
    TEST_DIR = "data/vision"
    test_images = [f for f in os.listdir(TEST_DIR) if f.endswith((".jpg", ".png"))]
    test_image_path = os.path.join(TEST_DIR, random.choice(test_images))
    print(f"\n🔍 Testing on: {test_image_path}")
    prediction = classify_image(test_image_path)
    print(f"🧠 Predicted Class: {prediction}")
