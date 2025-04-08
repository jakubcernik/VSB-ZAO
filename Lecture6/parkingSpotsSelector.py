import cv2
import json
import numpy as np

image_path = "./test_images_zao/test1.jpg"
output_json = "parking_slots_poly.json"

image = cv2.imread(image_path)
clone = image.copy()

all_slots = []
slot_index = 0
current_points = []

def order_points(pts):
    """
    Seřadí 4 body do pořadí:
    [levý horní, pravý horní, pravý dolní, levý dolní]
    """
    pts = np.array(pts, dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    top_left = pts[np.argmin(s)]
    bottom_right = pts[np.argmax(s)]
    top_right = pts[np.argmin(diff)]
    bottom_left = pts[np.argmax(diff)]

    return np.array([top_left, top_right, bottom_right, bottom_left], dtype="float32").tolist()

def click_event(event, x, y, flags, param):
    global current_points, slot_index

    if event == cv2.EVENT_LBUTTONDOWN:
        current_points.append((x, y))
        cv2.circle(clone, (x, y), 4, (0, 255, 0), -1)
        cv2.imshow("Klikni 4 body (rohy)", clone)

        if len(current_points) == 4:
            ordered = order_points(current_points)
            all_slots.append({
                "slot_id": slot_index,
                "points": ordered
            })
            print(f"Uloženo místo #{slot_index} – body: {ordered}")
            slot_index += 1
            current_points = []

            # zobraz číslo místa do středu
            cx = int(sum(p[0] for p in ordered) / 4)
            cy = int(sum(p[1] for p in ordered) / 4)
            cv2.putText(clone, str(slot_index - 1), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

cv2.imshow("Klikni 4 body (rohy)", clone)
cv2.setMouseCallback("Klikni 4 body (rohy)", click_event)

print("Klikni 4 rohy každého místa. Okno nezavírej, dokud neoznačíš všechna místa.")
print("Stiskni ESC pro ukončení.")

while True:
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break

cv2.destroyAllWindows()

# Ulož do souboru
with open(output_json, "w") as f:
    json.dump(all_slots, f, indent=2)

print(f"Hotovo! Uloženo {slot_index} míst do {output_json}")
