import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load YuNet face detector from OpenCV's ONNX model
YUNET_MODEL_PATH = "yunet.onnx"  # Make sure this file exists in the working directory

face_detector = cv2.FaceDetectorYN.create(
    model=YUNET_MODEL_PATH,  
    config="", 
    input_size=(960, 645),  # Adjust based on expected input size
    score_threshold=0.9,
    nms_threshold=0.3,
    top_k=5000
)

def detect_face_angle(image_path):
    try:
        # Load and preprocess image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Error: Unable to read the input image. Check the file path.")

        h, w, _ = image.shape
        face_detector.setInputSize((w, h))
        _, faces = face_detector.detect(image)

        if faces is None or len(faces) == 0:
            raise ValueError("Error: No face detected.")
        elif len(faces) > 1:
            raise ValueError("Error: More than one face detected.")

        # Extract landmarks
        x, y, w, h, conf, *landmarks = faces[0]
        landmarks = np.array(landmarks).reshape(-1, 2)
        
        # Get the landmark points directly (they are already in image coordinates)
        left_eye = landmarks[0]  # Left eye center
        right_eye = landmarks[1]  # Right eye center
        # Calculate mouth center as average of left and right mouth corners
        left_mouth = landmarks[3]  # Left mouth corner
        right_mouth = landmarks[4]  # Right mouth corner
        mouth = np.mean([left_mouth, right_mouth], axis=0)  # Mouth center

        # Extract coordinates
        C1, R1 = left_eye
        C2, _ = right_eye
        C3, R2 = mouth

        # Convert to integer values for visualization
        C1, R1, C2, R2, C3 = map(int, [C1, R1, C2, R2, C3])

        # Draw landmarks on image
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.scatter([C1, C2, C3], [R1, R1, R2], color='red', s=50)
        plt.plot([C1, C2, C3], [R1, R1, R2], 'g-')
        plt.show()

        # Calculate slopes
        m1 = (R2 - R1) / (C3 - C1) if C3 != C1 else np.inf
        m2 = (R2 - R1) / (C3 - C2) if C3 != C2 else np.inf

        # Compute face angle using arctan formula
        A = np.degrees(np.arctan(abs((m1 - m2) / (1 + m1 * m2))))

        # Display result
        if A < 44: print("< 18")
        elif 44 <= A <= 48: print("18 to 25")
        elif 49 <= A <= 54: print("26 to 35")
        elif 55 <= A <= 60: print("36 to 45")
        elif A > 60: print("> 45")
        else: print("None") # Handle invalid input if needed)

    except Exception as e:
        print(f"An error occurred: {e}")

detect_face_angle("kid1.jpg")