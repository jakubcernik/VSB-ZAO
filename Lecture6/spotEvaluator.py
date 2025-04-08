import cv2
import json
import numpy as np
import os
from skimage.feature import local_binary_pattern
from sklearn.svm import SVC

# Parametry pro LBP
radius = 1
n_points = 8 * radius

def extract_lbp_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, n_points, radius, method="uniform")
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)
    return hist


# Načtení a trénink SVM (stejné jako dříve)
# (Předpokládá se, že složky free/full existují)
def train_classifier():
    X_train, y_train = [], []
    for folder, label in [("free", 0), ("full", 1)]:
        for fname in os.listdir(folder):
            img_path = os.path.join(folder, fname)
            img = cv2.imread(img_path)
            if img is not None:
                X_train.append(extract_lbp_features(img))
                y_train.append(label)
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    clf = SVC(kernel='linear')
    clf.fit(X_train, y_train)
    return clf


# Načtení slotů z JSON
with open('coordinates.json', 'r') as f:
    slots_data = json.load(f)

slot_width, slot_height = 128, 256
clf = train_classifier()

os.makedirs('test_results_zao', exist_ok=True)
all_correct = 0
all_total = 0

for i in range(1, 25):
    # Načti obrázek
    img_file = f'test_images_zao/test{i}.jpg'
    image = cv2.imread(img_file)

    # Extrahuj a vyhodnoť
    predictions = []
    for slot in slots_data:
        points = np.array(slot['points'], dtype='float32')
        dst = np.array([
            [0, 0],
            [slot_width - 1, 0],
            [slot_width - 1, slot_height - 1],
            [0, slot_height - 1]
        ], dtype='float32')
        M = cv2.getPerspectiveTransform(points, dst)
        warped = cv2.warpPerspective(image, M, (slot_width, slot_height))
        features = extract_lbp_features(warped)
        pred = clf.predict([features])[0]
        predictions.append(pred)

    # Ulož výstup
    result_path = f'test_results_zao/result_{i}.txt'
    with open(result_path, 'w', encoding='utf-8') as f_out:
        for p in predictions:
            f_out.write(str(p) + '\n')

    # Porovnej s referenčním txt
    correct_file = f'test_texts_zao/test{i}.txt'
    with open(correct_file, 'r') as f_true:
        ground_truth = [int(line.strip()) for line in f_true.readlines()]

    # Spočítej úspěšnost
    total_slots = len(predictions)
    correct_slots = sum(1 for idx, val in enumerate(predictions) if val == ground_truth[idx])
    all_correct += correct_slots
    all_total += total_slots
    print(f'Obrázek {i}: přesnost {correct_slots / total_slots * 100:.2f}%')

celkova_presnost = all_correct / all_total * 100
print(f'Celková přesnost: {celkova_presnost:.2f}%')