#Step 1: Detect the rectangular face area from input face image using Matlab in-built object function. If number of detected face is more than one, an error message will be displayed.
#Step 2: Crop the detected rectangular face area and detect the significance eye-region using Matlab in-built object function.
#Step 3: The cropped rectangular face image is histogram equalized and then converted into binary image.
#Step 4: The binary image of face is divided horizontally into two parts. Upper part that contains two eyes is denoted by UPART and lower part that contains mouth is denoted by LPART.
#Step 5: Divide UPART vertically into two parts. One part that contains right eye is denoted by REYE and other part that contains left eye is denoted by LEYE.
#Step 6: Find the row number R1 with minimum row sum of gray level in UPART. Find the column numbers C1 and C2 with minimum column sum of gray level in REYE and LEYE. So, (R1, C1) coordinate represents middle point of right eyeball and (R1, C2) coordinate represents middle point of left eyeball.
#Step 7: Find the row number R2 with minimum row sum of gray level in LPART. So, R2 row represents the mouth row.
#Step 8: Calculate the midpoint C3 of two eye ball. So, C3= (C1+ C2) /2 and the coordinate (R2, C3) is middle point of mouth.
#Step 9: Draw a triangle by three coordinate points left eyeball (R1, C1), right eyeball (R1, C2) & mouth point (R2, C3).
#Step 10: Calculate slope (m1) of triangle sides from mouth point (R2, C3) to right eyeball (R1, C1) and slope (m2) of triangle sides from mouth point (R2, C3) to left eyeball (R1, C2).
#Step 11: Find the face angle (A) using formula: A = tan-1( (m1- m2) / (1 + m1 * m2) )
#Step 12: Determine age group based on the face angle (A).


import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect_face_angle(image_path):
    try:
        # Step 1: Load image and detect face
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

        # Read the input image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Error: Unable to read the input image. Check the file path.")
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

        if len(faces) > 1:
            raise ValueError("Error: More than one face detected.")
        elif len(faces) == 0:
            raise ValueError("Error: No face detected.")

        # Extract the face region
        x, y, w, h = faces[0]
        face_image = gray[y:y+h, x:x+w]

        # Step 2: Detect eye region
        eyes = eye_cascade.detectMultiScale(face_image)
        if len(eyes) < 2:
            raise ValueError("Error: Less than two eyes detected in the face region.")

        # Step 3: Histogram equalization and binary conversion
        face_hist_eq = cv2.equalizeHist(face_image)
        _, binary_image = cv2.threshold(face_hist_eq, 128, 255, cv2.THRESH_BINARY)

        # Step 4: Divide face into UPART and LPART
        h, w = binary_image.shape
        UPART = binary_image[:h//2, :]
        LPART = binary_image[h//2:, :]

        # Step 5: Divide UPART into REYE and LEYE
        REYE = UPART[:, :w//2]
        LEYE = UPART[:, w//2:]

        # Step 6: Find R1, C1, and C2
        R1 = np.argmin(np.sum(UPART, axis=1))  # Row with minimum row sum in UPART
        if np.sum(UPART[R1, :]) == 0:
            raise ValueError("Error: Unable to determine the upper row for eyes.")

        C1 = np.argmin(np.sum(REYE, axis=0))  # Column with minimum column sum in REYE
        if np.sum(REYE[:, C1]) == 0:
            raise ValueError("Error: Unable to determine the column for the right eye.")

        C2 = np.argmin(np.sum(LEYE, axis=0)) + w//2  # Adjust C2 relative to full image
        if np.sum(LEYE[:, C2 - w//2]) == 0:
            raise ValueError("Error: Unable to determine the column for the left eye.")

        # Step 7: Find R2
        R2 = np.argmin(np.sum(LPART, axis=1)) + h//2  # Adjust R2 relative to full image
        if np.sum(LPART[R2 - h//2, :]) == 0:
            raise ValueError("Error: Unable to determine the row for the mouth.")

        # Step 8: Calculate midpoint C3
        C3 = (C1 + C2) // 2

        # Step 9: Draw triangle
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.scatter([x + C1, x + C2, x + C3], [y + R1, y + R1, y + R2], color='red', s=50)
        plt.plot([x + C1, x + C2, x + C3], [y + R1, y + R1, y + R2], 'g-')
        plt.show()

        # Step 10: Calculate slopes
        m1 = (R2 - R1) / (C3 - C1) if C3 != C1 else np.inf
        m2 = (R2 - R1) / (C3 - C2) if C3 != C2 else np.inf

        # Step 11: Calculate face angle
        A = np.degrees(np.arctan(abs((m1 - m2) / (1 + m1 * m2))))

        # Step 12: Determine age group based on the face angle (A)
        # Display result
        if A < 44: print("< 18")
        elif 44 <= A <= 48: print("18 to 25")
        elif 49 <= A <= 54: print("26 to 35")
        elif 55 <= A <= 60: print("36 to 45")
        elif A > 60: print("> 45")
        else: print("None") # Handle invalid input if needed)
            
    except Exception as e:
        print(f"An error occurred: {e}")

# Usage
detect_face_angle("kid1.jpg")