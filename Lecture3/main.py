import cv2
import numpy as np
import pyautogui

# Load the dart template
dart_template = cv2.imread('target.png', 0)  # Grayscale template

# Capture a screenshot of the game window
screenshot = pyautogui.screenshot()
screenshot = np.array(screenshot)
# Convert to grayscale
screenshot_gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)

# Apply edge detection
screenshot_gray = cv2.Canny(screenshot_gray, 50, 200)
dart_template = cv2.Canny(dart_template, 50, 200)

# Now apply template matching
result = cv2.matchTemplate(screenshot_gray, dart_template, cv2.TM_CCOEFF_NORMED)

min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

# Define threshold for detection
threshold = 0.2 # Adjust based on accuracy
if max_val >= threshold:
    dart_position = max_loc
    print(f"Dart Found at: {dart_position}")

    # Move mouse to the dart position and throw
    dart_x, dart_y = dart_position
    pyautogui.moveTo(dart_x, dart_y, duration=0.2)
    pyautogui.click()
    pyautogui.click()

else:
    print("Dart not found!")

# Show detection result
h, w = dart_template.shape
cv2.rectangle(screenshot, dart_position, (dart_position[0] + w, dart_position[1] + h), (0, 255, 0), 2)
cv2.imshow('Detected Dart', screenshot)
cv2.waitKey(0)
cv2.destroyAllWindows()
