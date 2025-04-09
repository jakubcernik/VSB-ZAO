import cv2
import numpy as np
import time
from skimage.feature import local_binary_pattern

def load_eye_states(file_path):
    try:
        with open(file_path, 'r') as f:
            return [line.strip().lower() for line in f]
    except Exception as e:
        print(f"Could not load '{file_path}'!", e)
        return []

def get_lbp_image(image):
    radius = 1; n_points = 8 * radius
    # radius = 2; n_points = 16 * radius
    # radius = 3; n_points = 24 * radius
    lbp = local_binary_pattern(image, n_points, radius, method='uniform')
    return lbp

def is_eye_open(eye_region_gray):
    if eye_region_gray.size == 0:
        return False

    lbp_image = get_lbp_image(eye_region_gray)

    # Sobel operator for edge detection
    sobel_x = cv2.Sobel(eye_region_gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(eye_region_gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel = np.sqrt(sobel_x**2 + sobel_y**2)

    mean_value = lbp_image.mean()
    edge_density = np.sum(sobel > 20) / sobel.size

    _, thresholded = cv2.threshold(eye_region_gray, 70, 255, cv2.THRESH_BINARY)
    white_ratio = cv2.countNonZero(thresholded) / thresholded.size

    is_open = (mean_value > 20 and edge_density > 0.15) or white_ratio > 0.35

    return is_open


def process_frame(frame, cascades, eye_states, frame_idx):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Face detection
    faces = []
    for cascade_name in ["face_frontal", "face_profile"]:
        cascade = cascades[cascade_name]
        detected, _, weights = cascade.detectMultiScale3(
            gray, scaleFactor=1.1, minNeighbors=3,
            minSize=(200, 200), maxSize=(500, 500), outputRejectLevels=True)
        faces.extend([face for face, weight in zip(detected, weights) if weight > 2.0])

    # Eye detection
    predicted_eye_state = "unknown"
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 255), 2)

        roi_gray = gray[y:y + h, x:x + w]

        face_upper_half = roi_gray[0:int(h/2), :]
        eyes = cascades["eye"].detectMultiScale(face_upper_half, 1.1, 2,
                                                minSize=(35, 35), maxSize=(70, 70))

        for (ex, ey, ew, eh) in eyes:
            eye_roi = roi_gray[ey:ey + eh, ex:ex + ew]
            eye_roi = cv2.equalizeHist(eye_roi)  # Better contrast

            is_open = is_eye_open(eye_roi)
            color = (150, 180, 120)
            cv2.rectangle(frame, (x + ex, y + ey), (x + ex + ew, y + ey + eh), color, 2)

            if is_open:
                predicted_eye_state = "open"
                break

        if predicted_eye_state == "open":
            break

    # Precision
    correct = total = 0
    if frame_idx < len(eye_states):
        total = 1
        ground_truth = eye_states[frame_idx]

        if predicted_eye_state == "unknown":
            predicted_eye_state = "close"
        correct = int(predicted_eye_state == ground_truth)
        print(f"Frame {frame_idx}: Predicted={predicted_eye_state}, Truth={ground_truth}, Match={correct==1}")

    return frame, (correct, total), predicted_eye_state


def main():
    eye_states = load_eye_states('eye-state.txt')
    total_time = 0

    cascades = {
        "face_frontal": cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml'),
        "face_profile": cv2.CascadeClassifier('haarcascades/haarcascade_profileface.xml'),
        "eye": cv2.CascadeClassifier('eye_cascade.xml')
    }

    cap = cv2.VideoCapture('car.avi')
    correct_predictions = total_predictions = 0
    frame_index = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        start_time = time.time()
        frame, (correct, total), predicted_state = process_frame(frame, cascades, eye_states, frame_index)
        detection_time = time.time() - start_time
        total_time += detection_time

        correct_predictions += correct
        total_predictions += total

        cv2.putText(frame, f"Time: {detection_time:.3f}s", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        state_text = eye_states[frame_index] if frame_index < len(eye_states) else "unknown"
        color = (0, 255, 0) if predicted_state == state_text else (0, 0, 255)
        cv2.putText(frame, f"Truth: {state_text}, Pred: {predicted_state}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.imshow("Detection", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

        frame_index += 1

    cap.release()
    cv2.destroyAllWindows()

    if total_predictions > 0:
        print(f"Accuracy: {(correct_predictions / total_predictions) * 100:.2f}%")
        print(f"Correct: {correct_predictions}, Total: {total_predictions}")
        print(f"Total processing time: {total_time:.2f}s")
    else:
        print("eye-state.txt missing or empty!")


if __name__ == "__main__":
    main()