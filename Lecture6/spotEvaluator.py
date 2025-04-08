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
        # Získáme body parkovacího místa jako pole
        body_mista = np.array(slot['points'], dtype='float32')

        # Definujeme cílové body obdélníkového tvaru
        cilove_body = np.array([
            [0, 0],                         # levý horní
            [slot_width - 1, 0],            # pravý horní
            [slot_width - 1, slot_height - 1], # pravý dolní
            [0, slot_height - 1]            # levý dolní
        ], dtype='float32')

        # Transformujeme místo do standardní velikosti
        transformace = cv2.getPerspectiveTransform(body_mista, cilove_body)
        normalizovane_misto = cv2.warpPerspective(image, transformace, (slot_width, slot_height))

        # Extrahujeme příznaky a predikujeme stav (0=volno, 1=obsazeno)
        priznaky = extract_lbp_features(normalizovane_misto)
        predikce = clf.predict([priznaky])[0]
        predictions.append(predikce)

    # Vykreslení predikce
    for slot_idx, slot in enumerate(slots_data):
        # Převést body slotu na čísla typu int32
        body_slotu = np.array(slot['points'], dtype='int32')

        # Určit barvu podle predikce (zelená = volno, červená = obsazeno)
        if predictions[slot_idx] == 0:
            barva = (0, 255, 0)  # zelená = volno
        else:
            barva = (0, 0, 255)  # červená = obsazeno

        # Vykreslit polygon kolem parkovacího místa
        cv2.polylines(image, [body_slotu], True, barva, 2)

    cv2.imshow("Vyhodnocene parkoviste", image)
    cv2.waitKey(1000)
    cv2.destroyAllWindows()

    # Uložení výsledků
    # Otevření souboru pro zápis výsledků
    output_file = open(f'test_results_zao/result_{i}.txt', 'w')
    # Zápis každé predikce na samostatný řádek
    for prediction in predictions:
        output_file.write(str(prediction) + '\n')
    output_file.close()

    # Vyhodnocení přesnosti
    with open(f'test_texts_zao/test{i}.txt', 'r') as f_true:
        ground_truth = []
        for line in f_true.readlines():
            ground_truth.append(int(line.strip()))

    correct_slots = np.sum(np.array(predictions) == np.array(ground_truth))
    correct_prediction += correct_slots
    total_predictions += len(predictions)
    print(f'Obrázek {i}: přesnost {correct_slots / len(predictions) * 100:.2f}%')

celkova_presnost = correct_prediction / total_predictions * 100
print(f'Celková přesnost: {celkova_presnost:.2f}%')