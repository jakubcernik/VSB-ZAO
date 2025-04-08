import cv2
import json
import numpy as np
import os
from skimage.feature import local_binary_pattern
from sklearn.svm import SVC

radius = 1
n_points = 8 * radius
correct_prediction = 0
total_predictions = 0

def extract_lbp_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, n_points, radius, method="uniform")
    histogram = np.histogram(lbp.ravel(), bins=n_points + 2, range=(0, n_points + 2))[0]
    return histogram / histogram.sum()


def train_classifier():
    X_train, y_train = [], []
    for folder, label in [("free", 0), ("full", 1)]:
        for fname in os.listdir(folder):
            img = cv2.imread(os.path.join(folder, fname))
            if img is not None:
                X_train.append(extract_lbp_features(img))
                y_train.append(label)

    clf = SVC(kernel='linear')
    clf.fit(np.array(X_train), np.array(y_train))
    return clf

with open('coordinates.json', 'r') as f:
    slots_data = json.load(f)

slot_width, slot_height = 64, 128
clf = train_classifier()

os.makedirs('test_results_zao', exist_ok=True)

for i in range(1, 25):
    image = cv2.imread(f'test_images_zao/test{i}.jpg')

    predictions = []
    for slot in slots_data:
        points = np.array(slot['points'], dtype='float32')
        dst = np.array([[0, 0], [slot_width - 1, 0], [slot_width - 1, slot_height - 1], [0, slot_height - 1]],
                       dtype='float32')
        warped = cv2.warpPerspective(image, cv2.getPerspectiveTransform(points, dst), (slot_width, slot_height))
        predictions.append(clf.predict([extract_lbp_features(warped)])[0])

    # Vykreslení predikce
    for slot_idx, slot in enumerate(slots_data):
        cv2.polylines(image, [np.array(slot['points'], dtype='int32')], True,
                      (0, 255, 0) if predictions[slot_idx] == 0 else (0, 0, 255), 2)

    cv2.imshow("Vyhodnocené parkoviště", image)
    cv2.waitKey(1000)
    cv2.destroyAllWindows()

    # Uložení výsledků
    with open(f'test_results_zao/result_{i}.txt', 'w') as f_out:
        f_out.write('\n'.join(map(str, predictions)))

    # Vyhodnocení přesnosti
    with open(f'test_texts_zao/test{i}.txt', 'r') as f_true:
        ground_truth = [int(line.strip()) for line in f_true.readlines()]

    correct_slots = sum(p == gt for p, gt in zip(predictions, ground_truth))
    correct_prediction += correct_slots
    total_predictions += len(predictions)
    print(f'Obrázek {i}: přesnost {correct_slots / len(predictions) * 100:.2f}%')

celkova_presnost = correct_prediction / total_predictions * 100
print(f'Celková přesnost: {celkova_presnost:.2f}%')