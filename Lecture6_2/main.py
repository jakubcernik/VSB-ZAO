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
    radius = 1
    n_points = 8 * radius
    lbp = local_binary_pattern(image, n_points, radius, method='uniform')
    return lbp

def is_eye_open(eye_region_gray):
    if eye_region_gray.size == 0:
        return False

    lbp_image = get_lbp_image(eye_region_gray)

    # Sober operator for edge detection
    sobel_x = cv2.Sobel(eye_region_gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(eye_region_gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel = np.sqrt(sobel_x**2 + sobel_y**2)

    mean_value = lbp_image.mean()
    edge_density = np.sum(sobel > 20) / sobel.size

    # Výpočet poměru bílé plochy (otevřené oko má více jasných pixelů)
    _, thresholded = cv2.threshold(eye_region_gray, 55, 255, cv2.THRESH_BINARY)
    white_ratio = np.sum(thresholded > 0) / thresholded.size

    # Komplexnější pravidlo pro detekci otevřeného oka
    # Více vzorů hran a vyšší světlost znamenají otevřené oko
    is_open = (mean_value > 20 and edge_density > 0.15) or white_ratio > 0.45

    return is_open


def process_frame(frame, cascades, eye_states, frame_idx):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detekce obličeje
    faces = []
    for cascade_name in ["face_frontal", "face_profile"]:
        cascade = cascades[cascade_name]
        detected, _, weights = cascade.detectMultiScale3(
            gray, scaleFactor=1.1, minNeighbors=3,
            minSize=(200, 200), maxSize=(500, 500), outputRejectLevels=True)
        faces.extend([face for face, weight in zip(detected, weights) if weight > 2.0])

    # Detekce oka
    predicted_eye_state = "unknown"
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        eyes = cascades["eye"].detectMultiScale(roi_gray, 1.1, 2,
                                                minSize=(15, 15), maxSize=(70, 70))

        for (ex, ey, ew, eh) in eyes:
            eye_roi = roi_gray[ey:ey + eh, ex:ex + ew]
            # Předzpracování oblasti oka
            eye_roi = cv2.equalizeHist(eye_roi)  # Zlepšení kontrastu

# Detekce, zda je oko otevřené
            is_open = is_eye_open(eye_roi)
            color = (0, 255, 0) if is_open else (0, 0, 255)  # Zelená pro otevřené, červená pro zavřené
            cv2.rectangle(frame, (x + ex, y + ey), (x + ex + ew, y + ey + eh), color, 2)

            if is_open:
                predicted_eye_state = "open"
                break
                break

        if predicted_eye_state == "open":
            break

    # Vyhodnocení přesnosti
    # Vyhodnocení přesnosti
    correct = total = 0
    if frame_idx < len(eye_states):
        total = 1
        ground_truth = eye_states[frame_idx]
        # Pokud nebyl učiněn odhad, defaultně předpokládáme "close"
        if predicted_eye_state == "unknown":
            predicted_eye_state = "close"
        correct = int(predicted_eye_state == ground_truth)
        print(f"Frame {frame_idx}: Predicted={predicted_eye_state}, Truth={ground_truth}, Match={correct==1}")

    return frame, (correct, total), predicted_eye_state


def main():
    eye_states = load_eye_states('eye-state.txt')

    # Inicializace kaskád
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

        correct_predictions += correct
        total_predictions += total

        # Zobrazení výsledku
        cv2.putText(frame, f"Time: {detection_time:.3f}s", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        state_text = eye_states[frame_index] if frame_index < len(eye_states) else "unknown"
        cv2.putText(frame, f"Truth: {state_text}, Pred: {predicted_state}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Detection", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

        frame_index += 1

    cap.release()
    cv2.destroyAllWindows()

    if total_predictions > 0:
        print(f"Accuracy: {(correct_predictions / total_predictions) * 100:.2f}%")
        print(f"Correct: {correct_predictions}, Total: {total_predictions}")
    else:
        print("eye-state.txt missing or empty!")


if __name__ == "__main__":
    main()