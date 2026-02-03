import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread("/content/test1.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


blurred = cv2.GaussianBlur(image, (7, 7), 0)

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

# Canny Edge Detection
edges = cv2.Canny(gray, 100, 200)

# Binary Thresholding
_, thresholded = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# Display results in a 2x2 grid using axes[]
fig, axes = plt.subplots(2, 2, figsize=(8, 8))

axes[0, 0].set_title("Original Image")
axes[0, 0].imshow(image)
axes[0, 0].axis("off")

axes[0, 1].set_title("Gaussian Blur (7x7)")
axes[0, 1].imshow(blurred)
axes[0, 1].axis("off")

axes[1, 0].set_title("Canny Edges")
axes[1, 0].imshow(edges, cmap="gray")
axes[1, 0].axis("off")

axes[1, 1].set_title("Binary Threshold")
axes[1, 1].imshow(thresholded, cmap="gray")
axes[1, 1].axis("off")

plt.tight_layout()
plt.show()