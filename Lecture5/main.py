import cv2
import time

def load_eye_states(file_path):
    try:
        with open(file_path, 'r') as f:
            return [line.strip().lower() for line in f]
    except Exception as e:
        print(f"Could not load '{file_path}'!", e)
        return []


def initialize_cascades():
    return {
        "face_frontal": cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml'),
        "face_profile": cv2.CascadeClassifier('haarcascades/haarcascade_profileface.xml'),
        "eye": cv2.CascadeClassifier('eye_cascade_fusek.xml')
    }

def detect_faces(cascade, gray_frame, weight_threshold=2.0):
    faces, _, weights = cascade.detectMultiScale3(
            gray_frame, scaleFactor=1.1, minNeighbors=3, minSize=(200, 200), maxSize=(500, 500),
            outputRejectLevels=True)
    return [face for face, weight in zip(faces, weights) if weight > weight_threshold]

def is_eye_open(eye_region_gray, intensity_threshold=80):
    # Avg pixel intensity
    avg_intensity = eye_region_gray.mean()

    # Presuming "open"
    return avg_intensity > intensity_threshold

def process_frame(frame, cascades, eye_states, frame_index, correct_predictions, total_predictions):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detect_faces(cascades["face_frontal"], gray) + detect_faces(cascades["face_profile"], gray)
    predicted_eye_state = "close"

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]

        # Eye detection
        eyes = cascades["eye"].detectMultiScale(roi_gray, scaleFactor=1.05, minNeighbors=3, minSize=(20, 20), maxSize=(60, 60))
        for (ex, ey, ew, eh) in eyes:
            eye_roi_gray = roi_gray[ey:ey + eh, ex:ex + ew]

            cv2.rectangle(frame,
                          (x + ex, y + ey), (x + ex + ew, y + ey + eh),
                          (0, 255, 0), 2)

            if is_eye_open(eye_roi_gray):
                predicted_eye_state = "open"
                break

    if frame_index < len(eye_states):
        if predicted_eye_state == eye_states[frame_index]:
            correct_predictions += 1
        total_predictions += 1

    return correct_predictions, total_predictions


def main():
    eye_states = load_eye_states('eye-state.txt')
    cascades = initialize_cascades()
    cap = cv2.VideoCapture('fusek_face_car_01.avi')

    correct_predictions = 0
    total_predictions = 0
    frame_index = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        start_time = time.time()
        correct_predictions, total_predictions = process_frame(
            frame, cascades, eye_states, frame_index, correct_predictions, total_predictions
        )
        detection_time = time.time() - start_time

        cv2.putText(frame, f"Detection Time: {detection_time:.3f}s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 0, 255), 2)
        cv2.imshow("Face/Eye Detection", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

        frame_index += 1

    cap.release()
    cv2.destroyAllWindows()

    if total_predictions > 0:
        accuracy = (correct_predictions / total_predictions) * 100
        print(f"Overall eye state recognition accuracy: {accuracy:.2f}%")
    else:
        print("eye-state.txt missing!")


if __name__ == "__main__":
    main()