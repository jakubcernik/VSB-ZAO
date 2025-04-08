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
    # Parametry pro LBP
    radius = 5
    n_points = 8 * radius

    # Výpočet LBP pomocí knihovní funkce
    lbp = local_binary_pattern(image, n_points, radius, method='default')

    # Převod na uint8 pro kompatibilitu s OpenCV
    lbp_image = np.uint8(lbp)

    return lbp_image


def is_eye_open(eye_region_gray, lbp_threshold=None, uniformity_threshold=0.6):
    if eye_region_gray.size == 0:
        return False

    lbp_image = get_lbp_image(eye_region_gray)
    hist, _ = np.histogram(lbp_image.flatten(), bins=256, range=(0, 256))
    hist = hist.astype(float) / (hist.sum() + 1e-7)

    # Dynamický práh založený na statistikách LBP obrazu
    if lbp_threshold is None:
        lbp_threshold = lbp_image.mean() * 0.9  # Dynamický práh

    return (lbp_image.mean() > lbp_threshold and
            np.sum(np.square(hist)) > uniformity_threshold)


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
    predicted_eye_state = "close"
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        eyes = cascades["eye"].detectMultiScale(roi_gray, 1.05, 3,
                                                minSize=(20, 20), maxSize=(60, 60))

        for (ex, ey, ew, eh) in eyes:
            eye_roi = roi_gray[ey:ey + eh, ex:ex + ew]
            cv2.rectangle(frame, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (0, 255, 0), 2)

            # Použití dynamického prahu pro klasifikaci
            if is_eye_open(eye_roi, lbp_threshold=None):
                predicted_eye_state = "open"
                break

        if predicted_eye_state == "open":
            break

    # Vyhodnocení přesnosti
    # Vyhodnocení přesnosti
    correct = total = 0
    if frame_idx < len(eye_states):
        total = 1
        ground_truth = eye_states[frame_idx]
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