import cv2
import os
import matplotlib.pyplot as plt

input_folder = 'parking_images_cv01'
output_folder = 'cropped'
os.makedirs(output_folder, exist_ok=True)

x1, y1 = 778, 725   # Starting point
w, h = 140, 220
num_spaces = 3

all_cropped_images = []

# Loop over images
for image_name in os.listdir(input_folder):
    if image_name.endswith('.jpg'):
        image_path = os.path.join(input_folder, image_name)
        image = cv2.imread(image_path)

        # Loop over parking spaces
        for i in range(num_spaces):
            x = x1 + i * w
            y = y1

            # Crop
            cropped = image[y:y+h, x:x+w]
            all_cropped_images.append(cropped)

            # Save cropped
            cropped_name = f"{os.path.splitext(image_name)[0]}_space_{i+1}.jpg"
            cv2.imwrite(os.path.join(output_folder, cropped_name), cropped)

plt.figure(figsize=(15, 10))
for idx, cropped_img in enumerate(all_cropped_images):
    plt.subplot(4, 3, idx+1)  # 4 rows, 3 columns
    plt.imshow(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))
    plt.title(f"Image {idx//3 + 1}, Space {idx%3 + 1}")
    plt.axis('off')
plt.suptitle("All Cropped Parking Spaces")
plt.tight_layout()
plt.show()