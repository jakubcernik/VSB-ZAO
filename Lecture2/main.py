import os
import cv2
import numpy as np
import matplotlib
matplotlib.use("TkAgg")  # or "Agg" if running headless
import matplotlib.pyplot as plt


input_folder = "test-images"
output_folder = "test-output"
color_threshold = 150

correct_count = 0
total_images = 0

for image_name in os.listdir(input_folder):
    if image_name.endswith('.png'):
        image_path = os.path.join(input_folder, image_name)

        original_image = cv2.imread(image_path)

        image = original_image.copy()

        # Set white pixels to black
        white_mask = (image > 100).all(axis=2)  # Pixels where all R, G, B > 100
        image[white_mask] = [0, 0, 0]

        _, green_channel, red_channel = cv2.split(image)

        red_pixels = np.count_nonzero(red_channel > color_threshold)
        green_pixels = np.count_nonzero(green_channel > color_threshold)

        expected_color = "red" if "red" in image_name else "green"
        detected_color = "red" if red_pixels > green_pixels else "green"

        if detected_color == expected_color:
            correct_count += 1

        total_images += 1

        # Create output image
        output_subfolder = os.path.join(output_folder, "out-red" if red_pixels > green_pixels else "out-green")
        output_image = np.zeros_like(image)  # Black background

        if red_pixels > green_pixels:
            output_image[:, :, 2] = np.where(red_channel > color_threshold, red_channel, 0)  # Keep red pixels
        else:
            output_image[:, :, 1] = np.where(green_channel > color_threshold, green_channel, 0)  # Keep green pixels

        os.makedirs(output_subfolder, exist_ok=True)
        cv2.imwrite(os.path.join(output_subfolder, image_name), output_image)

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        axes[0].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        axes[0].set_title("Original")
        axes[0].axis('off')

        axes[1].imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
        axes[1].set_title("Output")
        axes[1].axis('off')

        plt.tight_layout()
        #plt.show()

        # Print accuracy after processing all images
        if total_images == 130:
            accuracy = (correct_count / total_images) * 100
            print(f"Accuracy: {accuracy:.2f}% ({correct_count} correct out of {total_images})")
