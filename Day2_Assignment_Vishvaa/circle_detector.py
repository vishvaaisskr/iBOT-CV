import cv2
import numpy as np
import matplotlib.pyplot as plt


def auto_blur_kernel(image):
    """
    Automatically choose blur kernel size based on image size.
    """
    h, w = image.shape[:2]
    area = h * w

    if area > 1_000_000:
        return 31
    elif area > 500_000:
        return 21
    else:
        return 15


def pencil_sketch(image_path, blur_kernel=None):
    """
    Convert an image to pencil sketch effect.

    Args:
        image_path (str): Path to input image
        blur_kernel (int or None): Gaussian blur kernel size (odd)

    Returns:
        tuple: (original_rgb, sketch) or (None, None)
    """
    try:
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print("Image not found or invalid format")
            return None, None

        # Auto-tune blur kernel if not provided
        if blur_kernel is None:
            blur_kernel = auto_blur_kernel(image)
            print(f"Auto blur kernel selected: {blur_kernel}")

        # Convert for display
        original_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Step 1: Grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Step 2: Invert
        inverted = 255 - gray

        # Step 3: Gaussian blur
        blurred = cv2.GaussianBlur(inverted, (blur_kernel, blur_kernel), 0)

        # Step 4: Invert blurred image
        inverted_blur = 255 - blurred

        # Step 5: Divide and scale (type-safe)
        sketch = cv2.divide(
            gray.astype(np.float32),
            inverted_blur.astype(np.float32) + 1e-6,
            scale=256
        )

        sketch = np.clip(sketch, 0, 255).astype(np.uint8)

        return original_rgb, sketch

    except Exception as e:
        print("Error occurred:", e)
        return None, None


def display_result(original, sketch, save_path=None):
    """
    Display original and sketch side-by-side.
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
    """Main function."""
    image_path = input("Enter image path: ").strip()
    user_input = input("Enter blur kernel (odd) or press Enter for auto: ").strip()

    if user_input == "":
        blur_kernel = None
    else:
        try:
            blur_kernel = int(user_input)
            if blur_kernel % 2 == 0 or blur_kernel <= 0:
                print("Invalid kernel. Using auto-tuning.")
                blur_kernel = None
        except:
            blur_kernel = None

    original, sketch = pencil_sketch(image_path, blur_kernel)

    if original is None or sketch is None:
        print("Failed to generate sketch")
        return

    display_result(original, sketch, "sketch_output.jpg")
    print("Sketch saved as sketch_output.jpg")


if __name__ == "__main__":
    main()
