import cv2
import dlib
import numpy as np
from imutils import face_utils

src_img = cv2.imread('mask.jpg')
dst_img = cv2.imread('original.jpg')

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('dlib_shape_predictor_68_face_landmarks.dat')

def get_face_landmarks(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 1)
    if len(faces) == 0:
        raise Exception('Obličej nebyl nalezen.')
    shape = predictor(gray, faces[0])
    return face_utils.shape_to_np(shape), faces[0]

def get_eye_centers(landmarks):
    # Levé oko: body 36–41, pravé oko: 42–47
    left_eye_center = np.mean(landmarks[36:42], axis=0).astype("int")
    right_eye_center = np.mean(landmarks[42:48], axis=0).astype("int")
    return left_eye_center, right_eye_center

def align_face(img, landmarks):
    left_eye, right_eye = get_eye_centers(landmarks)
    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    angle = np.degrees(np.arctan2(dy, dx))
    eyes_center = (int((left_eye[0] + right_eye[0]) // 2), int((left_eye[1] + right_eye[1]) // 2))
    # srovnání oči horizontálně
    M = cv2.getRotationMatrix2D(tuple(eyes_center), angle, 1)
    aligned = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_CUBIC)   # afinní transformace
    landmarks_rot = np.hstack([landmarks, np.ones((landmarks.shape[0], 1))])    # homogenní souřadnice
    landmarks = (M @ landmarks_rot.T).T # zarovnané landmarky
    return aligned, landmarks.astype(int)


src_points, src_rect = get_face_landmarks(src_img)
dst_points, dst_rect = get_face_landmarks(dst_img)

src_img_aligned, src_points = align_face(src_img, src_points)
dst_img_aligned, dst_points = align_face(dst_img, dst_points)

src_rect = cv2.boundingRect(np.array(src_points)) # nejmenší obdélník
dst_rect = cv2.boundingRect(np.array(dst_points))

x, y, w, h = src_rect
src_face = src_img_aligned[y:y+h, x:x+w]
dst_face_w = dst_rect[2]
dst_face_h = dst_rect[3]

# Eliptická maska
face_mask = np.zeros((src_face.shape[0], src_face.shape[1]), dtype=np.uint8)
center = (src_face.shape[1]//2, src_face.shape[0]//2)
axes = (int(src_face.shape[1]*0.45), int(src_face.shape[0]*0.55))
cv2.ellipse(face_mask, center, axes, 0, 0, 360, 255, -1)
face_mask = cv2.GaussianBlur(face_mask, (31, 31), 0)

src_face_resized = cv2.resize(src_face, (dst_face_w, dst_face_h))
face_mask_resized = cv2.resize(face_mask, (dst_face_w, dst_face_h))

clone_center = (dst_rect[0] + dst_face_w // 2, dst_rect[1] + dst_face_h // 2)

# SeamlessClone
output = cv2.seamlessClone(
    src_face_resized, dst_img_aligned, face_mask_resized, clone_center, cv2.NORMAL_CLONE
)

cv2.imwrite('result.jpg', output)
print('Uloženo jako result.jpg')