
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


test_transforms = transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])




print("\nLoading test dataset...")

test_dataset = datasets.ImageFolder("data/test", transform=test_transforms)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

class_names = test_dataset.classes

print("Test samples:", len(test_dataset))
print("Classes:", class_names)




print("\nLoading trained model...")

model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)

model.load_state_dict(torch.load("best_model.pth", map_location=device))
model = model.to(device)

model.eval()

print("Model loaded successfully!")




print("\nEvaluating on test set...")

correct = 0
total = 0

all_preds = []
all_labels = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())


test_accuracy = 100 * correct / total

print(f"Final Test Accuracy: {test_accuracy:.2f}%")




print("Generating confusion matrix...")

cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names)

plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("Confusion Matrix")

plt.savefig("confusion_matrix.png")
plt.show()

print("Confusion matrix saved as confusion_matrix.png")




print("\nDisplaying some sample predictions...")

def imshow(img, title):
    img = img.numpy().transpose((1, 2, 0))
    img = img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]
    img = np.clip(img, 0, 1)

    plt.imshow(img)
    plt.title(title)
    plt.axis("off")


# Get a small batch from test loader
dataiter = iter(test_loader)
images, labels = next(dataiter)

images = images.to(device)
outputs = model(images)
_, preds = torch.max(outputs, 1)

plt.figure(figsize=(12, 8))

for i in range(10):
    plt.subplot(2, 5, i+1)
    imshow(images[i].cpu(),
           f"Pred: {class_names[preds[i]]}\nActual: {class_names[labels[i]]}")

plt.tight_layout()
plt.show()


print("\nEvaluation complete!")


