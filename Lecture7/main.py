import os
import cv2
import numpy as np
import dlib
from scipy.spatial import distance
from imutils import face_utils
from concurrent.futures import ProcessPoolExecutor
import time


def load_ground_truth(file_path):
    """Načte anotace mrknutí očí z textového souboru."""
    blink_ranges = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                if line.strip():  # Přeskočí prázdné řádky
                    start, end = map(int, line.strip().split())
                    blink_ranges.append((start, end))
        return blink_ranges
    except Exception as e:
        print(f"Chyba při načítání anotací: {e}")
        return []


def eye_aspect_ratio(eye):
    """Vypočítá Eye Aspect Ratio (EAR) pro detekci otevřenosti očí."""
    # Vzdálenosti mezi vertikálními body očí
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    # Vzdálenost mezi horizontálními body očí
    C = distance.euclidean(eye[0], eye[3])
    # Výpočet EAR
    ear = (A + B) / (2.0 * C)
    return ear


# Globální proměnné pro paralelní zpracování
detector = None
predictor = None
lStart, lEnd, rStart, rEnd = None, None, None, None
EAR_THRESHOLD = 0.18
blink_ranges = None


def initialize_globals():
    global detector, predictor, lStart, lEnd, rStart, rEnd, blink_ranges

    # Inicializace detektoru a prediktoru
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("./landmark_models/dlib_shape_predictor_68_face_landmarks.dat")

    # Definice indexů bodů pro levé a pravé oko
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    # Načtení anotací
    blink_ranges = load_ground_truth('./video1/anot1.txt')


def process_frame(args):
    """Zpracuje jeden snímek paralelně."""
    global detector, predictor, lStart, lEnd, rStart, rEnd, EAR_THRESHOLD, blink_ranges

    img_file, frame_num, skip_display = args

    # Určení ground truth stavu očí pro aktuální snímek
    gt_closed = False
    for start, end in blink_ranges:
        if start <= frame_num <= end:
            gt_closed = True
            break

    # Načtení snímku
    frame = cv2.imread(os.path.join('video1', img_file))

    # Zmenšení snímku pro rychlejší zpracování
    scale_factor = 0.75
    frame_small = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)
    gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)

    # Detekce obličejů
    faces = detector(gray, 0)

    # Inicializace predikovaného stavu a EAR hodnoty
    pred_closed = False
    ear_value = 0

    for face in faces:
        # Přepočet souřadnic pro původní velikost
        scaled_face = dlib.rectangle(
            int(face.left() / scale_factor),
            int(face.top() / scale_factor),
            int(face.right() / scale_factor),
            int(face.bottom() / scale_factor)
        )

        # Detekce landmark bodů na původním obrázku
        shape = predictor(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), scaled_face)
        shape = face_utils.shape_to_np(shape)

        # Extrakce bodů očí
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]

        # Výpočet EAR pro obě oči
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        # Průměrný EAR
        ear = (leftEAR + rightEAR) / 2.0
        ear_value = ear

        # Kontrola EAR pro detekci mrknutí
        if ear < EAR_THRESHOLD:
            pred_closed = True

        # Pouze pokud je vyžadováno zobrazení
        if not skip_display:
            # Vykreslení bodů očí
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

            if pred_closed:
                cv2.putText(frame, "MRKNUTÍ!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Zobrazení EAR hodnoty
            cv2.putText(frame, f"EAR: {ear:.2f}", (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Pouze pokud je vyžadováno zobrazení
    if not skip_display:
        # Zobrazení výsledku
        cv2.putText(frame, f"Frame: {frame_num}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"GT: {'Closed' if gt_closed else 'Open'}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Pred: {'Closed' if pred_closed else 'Open'}", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0 if pred_closed == gt_closed else 0), 2)

    return {
        'frame_num': frame_num,
        'gt_closed': gt_closed,
        'pred_closed': pred_closed,
        'ear': ear_value,
        'frame': frame if not skip_display else None
    }


def main():
    start_time = time.time()

    # Načtení anotací
    global blink_ranges
    blink_ranges = load_ground_truth('./video1/anot1.txt')
    print(f"Načteno {len(blink_ranges)} rozsahů mrknutí")

    # Inicializace detektoru a prediktoru
    global detector, predictor, lStart, lEnd, rStart, rEnd
    detector = dlib.get_frontal_face_detector()
    landmark_file = "./landmark_models/dlib_shape_predictor_68_face_landmarks.dat"
    if not os.path.exists(landmark_file):
        print(f"CHYBA: Soubor prediktoru neexistuje: {landmark_file}")
        return

    predictor = dlib.shape_predictor(landmark_file)

    # Definice indexů bodů pro levé a pravé oko
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    # Konstanty pro optimalizaci
    PROCESS_EVERY_N_FRAMES = 1  # Zpracovat každý N-tý snímek
    DISPLAY_FREQUENCY = 10  # Zobrazit každý N-tý snímek
    NUM_WORKERS = os.cpu_count()  # Počet paralelních procesů

    # Získání seznamu snímků
    image_files = sorted([f for f in os.listdir('./video1') if f.endswith('.jpg')],
                         key=lambda x: int(os.path.splitext(x)[0][3:]))

    print(f"Nalezeno {len(image_files)} snímků")

    # Příprava seznamu argumentů pro paralelní zpracování
    process_args = []
    for i, img_file in enumerate(image_files[::PROCESS_EVERY_N_FRAMES]):
        frame_num = int(os.path.splitext(img_file)[0][3:])
        # Zobrazit jen některé snímky
        skip_display = (i % DISPLAY_FREQUENCY != 0)
        process_args.append((img_file, frame_num, skip_display))

    # Paralelní zpracování snímků
    print(f"Spouštím paralelní zpracování s {NUM_WORKERS} procesy...")
    results = []

    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        # Inicializace globálních proměnných v každém procesu
        for _ in range(NUM_WORKERS):
            executor.submit(initialize_globals)

        # Zpracování snímků paralelně
        for result in executor.map(process_frame, process_args):
            results.append(result)

            # Zobrazení výsledku pouze pro vybrané snímky
            if result['frame'] is not None:
                cv2.imshow("Frame", result['frame'])
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC key
                    break

    cv2.destroyAllWindows()

    # Post-processing výsledků - vyhlazení pomocí stavového automatu
    smoothed_results = []
    state = 'open'  # Výchozí stav
    consec_frames = 0
    MIN_CONSEC_CLOSED = 2
    MIN_CONSEC_OPEN = 2

    for result in sorted(results, key=lambda x: x['frame_num']):
        ear = result['ear']
        raw_pred_closed = result['pred_closed']

        # Stavový automat pro vyhlazení predikcí
        if state == 'open' and raw_pred_closed:
            consec_frames += 1
            if consec_frames >= MIN_CONSEC_CLOSED:
                state = 'closed'
                consec_frames = 0
        elif state == 'closed' and not raw_pred_closed:
            consec_frames += 1
            if consec_frames >= MIN_CONSEC_OPEN:
                state = 'open'
                consec_frames = 0
        else:
            consec_frames = 0

        # Výsledný stav
        pred_closed = (state == 'closed')

        smoothed_results.append({
            'frame_num': result['frame_num'],
            'gt_closed': result['gt_closed'],
            'pred_closed': pred_closed
        })

    # Statistiky pro vyhodnocení
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    for result in smoothed_results:
        if result['pred_closed'] and result['gt_closed']:
            true_positives += 1
        elif result['pred_closed'] and not result['gt_closed']:
            false_positives += 1
        elif not result['pred_closed'] and result['gt_closed']:
            false_negatives += 1

    # Výpočet metrik
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    end_time = time.time()

    print(f"\nVýsledky detekce mrkání:")
    print(f"Přesnost (Precision): {precision:.4f}")
    print(f"Úplnost (Recall): {recall:.4f}")
    print(f"F1 skóre: {f1_score:.4f}")
    print(f"\nCelková doba zpracování: {end_time - start_time:.2f} sekund")
    print(f"Zpracováno {len(smoothed_results)} z {len(image_files)} snímků")


if __name__ == "__main__":
    main()