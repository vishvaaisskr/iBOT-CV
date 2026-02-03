import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the grayscale image
image = cv2.imread("/content/test1.jpg", cv2.IMREAD_GRAYSCALE)



# Calculate histogram
hist = cv2.calcHist([image], [0], None, [256], [0, 256])

# Calculate statistics
mean_val = np.mean(image)
median_val = np.median(image)
std_val = np.std(image)


fig, axes = plt.subplots(1, 2, figsize=(10, 4))

axes[0].set_title("Grayscale Image")
axes[0].imshow(image, cmap="gray")
axes[0].axis("off")

axes[1].set_title("Pixel Intensity Histogram")
axes[1].plot(hist)
axes[1].set_xlim([0, 256])
axes[1].set_xlabel("Pixel Intensity")
axes[1].set_ylabel("Frequency")

plt.tight_layout()
plt.show()


print(f"Mean Intensity   : {mean_val:.2f}")
print(f"Median Intensity : {median_val:.2f}")
print(f"Standard Deviation: {std_val:.2f}")