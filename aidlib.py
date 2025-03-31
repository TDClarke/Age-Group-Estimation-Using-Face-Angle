import cv2
import numpy as np
import matplotlib.pyplot as plt
import dlib

def detect_face_angle(img):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    dets = detector(img, 1)
    if len(dets) == 0:
        print("No face detected")
        return None, None

    for d in dets:
        shape = predictor(img, d)
        left_eye = [np.mean([shape.part(p).x for p in [37,38,40,41]]), 
                    np.mean([shape.part(p).y for p in [37,38,40,41]])]
        right_eye = [np.mean([shape.part(p).x for p in [43,44,46,47]]), 
                     np.mean([shape.part(p).y for p in [43,44,46,47]])]
        lip = [np.mean([shape.part(p).x for p in [61,62,63,65,66,67]]), 
               np.mean([shape.part(p).y for p in [61,62,63,65,66,67]])]
        triangle = [tuple(left_eye), tuple(right_eye), tuple(lip)]

        if lip[0] == left_eye[0] or lip[0] == right_eye[0]:
            print("Error: Vertical alignment detected, undefined slope.")
            return None, triangle

        m1 = (lip[1] - left_eye[1]) / (lip[0] - left_eye[0])
        m2 = (lip[1] - right_eye[1]) / (lip[0] - right_eye[0])
        A = np.degrees(np.arctan(abs((m1 - m2) / (1 + m1 * m2))))
        
        return A, triangle

# Load image
img = cv2.imread("joe.jpg")
if img is None:
    raise ValueError("Error loading image. Check the file path.")

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

A, triangle = detect_face_angle(img)
if A is not None:
    plt.imshow(img)
    plt.axis('off')
    t = plt.Polygon(triangle, facecolor=(1,1,1,0), edgecolor='r')
    plt.gca().add_patch(t)
    plt.show()
    print(f'Angle: {A}')
