import cv2
import dlib
import numpy as np

# Load the face detector and facial landmarks predictor
#face_detector = dlib.get_frontal_face_detector()
face_detector2 = dlib.cnn_face_detection_model_v1('models/mmod_human_face_detector.dat')  # You need to download this file
landmark_predictor = dlib.shape_predictor('models/shape_predictor_68_face_landmarks.dat')  # You need to download this file


def rotate_bound(image, angle):
    print("rotate_bound")
    # grab the dimensions of the image and then determine the centre
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    affine = cv2.warpAffine(image, M, (nW, nH))
    # cv2.imshow('Rotated Image', affine)
    # cv2.waitKey(0)
    return affine


def detect_faces2(image, max_rotation_attempts=4):
    for _ in range(max_rotation_attempts):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_detector2(gray)
        for i, d in enumerate(faces):
            if (d.confidence > 1.0):
                print("Detection {}: Left: {} Top: {} Right: {} Bottom: {} Confidence: {}".format(
                    i, d.rect.left(), d.rect.top(), d.rect.right(), d.rect.bottom(), d.confidence))

                landmarks = landmark_predictor(gray, d.rect)
                left_eye = (landmarks.part(36).x, landmarks.part(36).y)
                right_eye = (landmarks.part(45).x, landmarks.part(45).y)
                cv2.circle(image, left_eye, 5, (0, 0, 255), -2)  # Left eye in red
                cv2.circle(image, right_eye, 5, (0, 0, 255), -2)  # Right eye in red

                cv2.imshow('Corrected Image', image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                return faces

        image = rotate_bound(image, 90)

    # If no faces are found after rotation attempts, return an empty list
    return []

# def detect_faces(image, max_rotation_attempts=4):
#     for _ in range(max_rotation_attempts):
#         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#         # Detect faces in the grayscale image
#         faces = face_detector(gray, 1)
#         print(faces)
#         # Check if any faces are detected
#         for face in faces:
#             print("Face detected at ({}, {}), with height: {} and width: {}".format(face.left(), face.top(), face.height(), face.width()))
#             landmarks = landmark_predictor(gray, face)
#             # Calculate the angle of the face
#             left_eye = (landmarks.part(36).x, landmarks.part(36).y)
#             right_eye = (landmarks.part(45).x, landmarks.part(45).y)
#             # Draw circles at the positions of the eyes
#             cv2.circle(image, left_eye, 5, (0, 0, 255), -1)  # Left eye in red
#             cv2.circle(image, right_eye, 5, (0, 0, 255), -1)  # Right eye in red
#
#         if len(faces):
#             cv2.imshow('Corrected Image', image)
#             cv2.waitKey(0)
#             cv2.destroyAllWindows()
#             return faces
#
#         image = rotate_bound(image, 90)
#
#     # If no faces are found after rotation attempts, return an empty list
#     return []


im_path = 'images/sources/AZ_180.jpg'

image = cv2.imread(im_path)

#detected_faces = detect_faces(image)

detected_faces = detect_faces2(image)


print(f"Found faces: {detected_faces}")
