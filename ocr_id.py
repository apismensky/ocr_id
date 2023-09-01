import pytesseract
# from PIL.Image import Transpose
from pytesseract import Output
import cv2
import os
import easyocr
import json
# from PIL import Image
import dlib
import numpy as np

source_folder = 'images/sources'
tesseract_folder = 'images/boxes_tesseract'
easy_folder = 'images/boxes_easy'
result_file = 'result.json'

# Load the EasyOCR model
reader = easyocr.Reader(['en'])
# Define the threshold value for angle detection by tesseract
threshold = 1.0  # Adjust this value as needed

# Load the face detector and facial landmarks predictor
face_detector = dlib.cnn_face_detection_model_v1('models/mmod_human_face_detector.dat')  # You need to download this file
#face_detector = dlib.get_frontal_face_detector()
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
    return affine


def detect_faces_and_rotate(image, max_rotation_attempts=4):
    for _ in range(max_rotation_attempts):
        # Convert the image to grayscale for face detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Detect faces in the grayscale image
        faces = face_detector(gray)
        # Check if any faces are detected
        for i, d in enumerate(faces):
            if (d.confidence > 1.0):
                print("Detection {}: Left: {} Top: {} Right: {} Bottom: {} Confidence: {}".format(
                    i, d.rect.left(), d.rect.top(), d.rect.right(), d.rect.bottom(), d.confidence))
                landmarks = landmark_predictor(gray, d.rect)
                # Calculate the angle of the face
                left_eye = (landmarks.part(36).x, landmarks.part(36).y)
                right_eye = (landmarks.part(45).x, landmarks.part(45).y)
                # Draw circles at the positions of the eyes
                cv2.circle(image, left_eye, 5, (255, 0, 0), -2)  # Left eye in red
                cv2.circle(image, right_eye, 5, (255, 0, 0), -2)  # Right eye in red
                return image

        image = rotate_bound(image, 90)
    return image


def tess(img_path):
    print("img_path: ", img_path)
    img = cv2.imread(img_path)
    d = pytesseract.image_to_data(img, output_type=Output.DICT, config='--psm 1')
    d['text'] = [t.strip() for t in d['text'] if t.strip()]
    txt = ' '.join(d['text'])

    print(f"tesseract: {txt}")
    n_boxes = len(d['level'])
    for i in range(n_boxes):
        (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 1)
    cv2.imwrite(os.path.join(tesseract_folder, os.path.basename(img_path)), img)
    return txt
    # can use tesseract to detect rotation angle
    # try:
    #     osd=pytesseract.image_to_osd(img, output_type=Output.DICT)
    # except Exception as e:
    #     print("error: ", e)
    #     return [txt, 0]
    # print(osd)
    #
    # # return rotation angle if the confidence is high do it can be used to fix the rotation in easyocr
    # if 'rotate' in osd and 'orientation_conf' in osd and osd['orientation_conf'] > threshold:
    #     return [txt, osd['rotate']]
    # else:
    #     return [txt, 0]


# def fix_rotation(img_path, angle):
#     img = Image.open(img_path)
#     new_img = img
#     if angle in range(80, 100):
#         new_img = img.transpose(Transpose.ROTATE_270)
#     elif angle in range(170, 190):
#         new_img = img.transpose(Transpose.ROTATE_180)
#     elif angle in range(260, 280):
#         new_img = img.transpose(Transpose.ROTATE_90)
#     new_path = os.path.join(source_folder, filename[:-4] + '_fixed.jpg')
#     new_img.convert('RGB').save(new_path)
#     return new_path


def easy(img_path):
    print("img_path: ", img_path)
    i = cv2.imread(img_path)
    i = detect_faces_and_rotate(i, 4)
    results = reader.readtext(i)
    texts = []
    for result in results:
        x1, y1, x2, y2 = result[0][0][0], result[0][0][1], result[0][2][0], result[0][2][1]
        cv2.rectangle(i, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 1)
        texts.append(result[1].strip())
        cv2.imwrite(os.path.join(easy_folder, os.path.basename(img_path)), i)
    result = ' '.join(texts)
    print(f"EasyOCR: {result}")
    return result


if not os.path.exists(easy_folder):
    os.makedirs(easy_folder)
if not os.path.exists(tesseract_folder):
    os.makedirs(tesseract_folder)

# Create a list to store the JSON data
json_data = []

for filename in os.listdir(source_folder):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        img_p = os.path.join(source_folder, filename)
        tess_result = tess(img_p)
        easyocr_text = easy(img_p)
        data = {
            "file": filename,
            "results": [
                {
                    "tesseract": {
                        "text": tess_result
                    }
                },
                {
                    "easyocr": {
                        "text": easyocr_text
                    }
                }
            ]
        }

        json_data.append(data)

with open(result_file, 'w') as f:
    json.dump(json_data, f, indent=4, sort_keys=True)
