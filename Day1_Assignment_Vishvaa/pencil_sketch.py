import cv2
import numpy as np
import matplotlib.pyplot as plt


def pencil_sketch(image_path, blur_kernel=21):
    """
    Convert an image to pencil sketch effect.
    """
    try:

        test_image = cv2.imread(image_path)
        if test_image is None:
            print("Image not found or invalid format")
            return None, None

        rgb_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)

        inverted = 255 - gray
        blurred = cv2.GaussianBlur(inverted, (blur_kernel, blur_kernel), 0)
        inverted_blur = 255 - blurred


        sketch = cv2.divide(
            gray.astype(np.float32),
            inverted_blur.astype(np.float32) + 1e-6,
            scale=256)

        sketch = np.clip(sketch, 0, 255).astype(np.uint8)

        return rgb_image, sketch

    except Exception as e:
        print("Error occurred:", e)
        return None, None



def display_result(original, sketch, save_path=None):
    """
    Display original and sketch side-by-side.

    Args:
        original: Original image (RGB)
        sketch: Sketch image (grayscale)
        save_path: Optional path to save the sketch
    """
    plt.figure(figsize=(8, 4))

    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(original)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Pencil Sketch")
    plt.imshow(sketch, cmap="gray")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    if save_path:
        cv2.imwrite(save_path, sketch)


def main():
    """Main function to run the pencil sketch converter."""
    image_path = input("Enter image path: ")
    blur_kernel = int(input("Enter blur kernel size (odd number): "))

    original, sketch = pencil_sketch(image_path, blur_kernel)

    if original is None or sketch is None:
        print("Failed to generate sketch")
        return

    display_result(original, sketch, "sketch_output.jpg")



if __name__ == "__main__":
    main()