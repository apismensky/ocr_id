import pytesseract
from pytesseract import Output
import cv2
import os
import json
from craft_text_detector import Craft
import time
import subprocess
# import dlib
import numpy as np

source_folder = 'images/sources'
tesseract_folder = 'images/boxes_tesseract'
result_file = 'result.json'
output_dir = 'images/boxes_craft'

# create a craft instance
craft = Craft(output_dir=output_dir, crop_type="poly", cuda=False)

# Load the face detector and facial landmarks predictor
#face_detector = dlib.cnn_face_detection_model_v1('models/mmod_human_face_detector.dat')  # You need to download this file
#face_detector = dlib.get_frontal_face_detector()
#landmark_predictor = dlib.shape_predictor('models/shape_predictor_68_face_landmarks.dat')  # You need to download this file


def rotate_bound(i, angle):
    # grab the dimensions of the image and then determine the centre
    (h, w) = i.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    m = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos = np.abs(m[0, 0])
    sin = np.abs(m[0, 1])

    # compute the new bounding dimensions of the image
    nw = int((h * sin) + (w * cos))
    nh = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    m[0, 2] += (nw / 2) - cX
    m[1, 2] += (nh / 2) - cY

    # perform the actual rotation and return the image
    affine = cv2.warpAffine(i, m, (nw, nh))
    return affine


# def detect_faces_and_rotate(image, max_rotation_attempts=4):
#     for _ in range(max_rotation_attempts):
#         # Convert the image to grayscale for face detection
#         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#         # Detect faces in the grayscale image
#         faces = face_detector(gray)
#         # Check if any faces are detected
#         for i, d in enumerate(faces):
#             if d.confidence > 1.0:
#                 print("Face Detection {}: Left: {} Top: {} Right: {} Bottom: {} Confidence: {}".format(
#                     i, d.rect.left(), d.rect.top(), d.rect.right(), d.rect.bottom(), d.confidence))
#                 landmarks = landmark_predictor(gray, d.rect)
#                 # Calculate the angle of the face
#                 left_eye = (landmarks.part(36).x, landmarks.part(36).y)
#                 right_eye = (landmarks.part(45).x, landmarks.part(45).y)
#                 # Draw circles at the positions of the eyes
#                 cv2.circle(image, left_eye, 5, (255, 0, 0), -2)  # Left eye in red
#                 cv2.circle(image, right_eye, 5, (255, 0, 0), -2)  # Right eye in red
#                 return image
#
#         image = rotate_bound(image, 90)
#     return image


def tess(img_path, cfg='--psm 1'):
    print("tesseract img_path: ", img_path)
    img = cv2.imread(img_path)
    d = pytesseract.image_to_data(img, output_type=Output.DICT, config=cfg)
    d['text'] = [t.strip() for t in d['text'] if t.strip()]
    txt = ' '.join(d['text'])

    print(f"tesseract: {txt}")
    n_boxes = len(d['level'])
    for i in range(n_boxes):
        (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 1)
    # cv2.putText(img, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    cv2.imwrite(os.path.join(tesseract_folder, os.path.basename(img_path)), img)
    return txt


def execute_tesseract(img_path, cfg='--psm 1'):
    command = f"tesseract {img_path} stdout {cfg}"
    return execute_external_command(command)


def execute_external_command(command):
    try:
        # Run the command and capture its output
        result = subprocess.check_output(command, shell=True, text=True)
        return result
    except subprocess.CalledProcessError as e:
        # Handle errors, such as non-zero exit status
        return f"Error: {e}"
    except Exception as e:
        # Handle other exceptions
        return f"An error occurred: {str(e)}"

if not os.path.exists(tesseract_folder):
    os.makedirs(tesseract_folder)

# Create a list to store the JSON data
json_data = []


def get_truth_file_name(fname):
    parts = fname.split("_")[0]
    if len(parts) == len(fname):
        parts = fname.split(".")[0]
    print("parts: ", parts)
    return parts


def get_truth(fname):
    truth_file_name = get_truth_file_name(fname)
    truth_path = os.path.join(source_folder, truth_file_name + ".txt")
    print("truth_path: ", truth_path)
    return read(truth_path)


def read(file_path):
    try:
        # Open the file in read mode
        with open(file_path, "r") as file:
            # Read the entire contents of the file into a string
            file_contents = file.read()

        # Print or process the file contents as needed
        return file_contents

    except FileNotFoundError:
        print(f"ERROR: The file '{file_path}' does not exist.")
    except Exception as e:
        print(f"ERROR: An error occurred: {str(e)}")
    return ""


def clean_text(text):
    if text is None:
        return ''
    return ' '.join(text.lower().strip().split())


def calculate_word_accuracy(detected_text, truth_text):
    detected_words = clean_text(detected_text).split()
    truth_words = clean_text(truth_text).split()
    total_words = len(truth_words)
    correct_words = sum(1 for dw in detected_words if dw in truth_words)
    if total_words == 0:
        return 0, 0, 0
    return correct_words, total_words, (correct_words / total_words) * 100


def calculate_dimensions(x_coordinates, y_coordinates):
    left = min(x_coordinates)
    top = min(y_coordinates)
    right = max(x_coordinates)
    bottom = max(y_coordinates)
    width = right - left
    height = bottom - top
    return left, top, width, height


def write_uzn_file(i, o):
    with open(i, "r") as infile, open(o, "w") as outfile:
        for line in infile:
            # Split the line into individual coordinate values
            coordinates = list(map(int, line.strip().split(',')))

            # Split the coordinates into x and y lists
            x_coordinates = coordinates[::2]
            y_coordinates = coordinates[1::2]

            # Calculate the dimensions
            l, t, w, h = calculate_dimensions(x_coordinates, y_coordinates)

            # Write the transformed data to the output file
            outfile.write(f'{l} {t} {w} {h} Text/Latin\n')


def get_number_from_filename(filename):
    try:
        number = int(filename.split("_")[1].split(".")[0])
        return number
    except (ValueError, IndexError):
        return float('inf')


total_truth_words = 0
total_tess_correct_words = 0
total_craft_correct_words = 0
#total_crop_correct_words = 0

for filename in os.listdir(source_folder):

    if filename == 'sign.jpg' or filename == 'dubai.jpg' or filename == 'madaba.jpg' or filename == 'german.jpg' or filename == 'monument.jpg':
    #if filename.endswith('az_90.jpg'):
    #if filename.endswith('.jpg') or filename.endswith('.png'):

        img_p = os.path.join(source_folder, filename)
        tess_result = tess(img_p)
        truth = get_truth(filename)
        tess_correct_words, truth_words, tess_word_accuracy = calculate_word_accuracy(tess_result, truth)
        print(f"tesseract word_accuracy: {tess_word_accuracy}")
        print(f"tesseract correct_words: {tess_correct_words}")
        print(f"truth_words: {truth_words}")

        total_truth_words += truth_words
        total_tess_correct_words += tess_correct_words

        # read the image
        img = cv2.imread(img_p)
        # detect the orientation

        # rotted_img = detect_faces_and_rotate(img)
        # rotated_image_path = img_p[:-4] + '_fixed.jpg'
        # cv2.imwrite(rotated_image_path, rotted_img)
        rotated_image_path = img_p

        # apply CRAFT
        start_time = time.time()
        craft.detect_text(rotated_image_path)
        print("CRAFT detect_text time: --- %s seconds ---" % (time.time() - start_time))

        # read the result of the bounding boxes detection and transform it into .uzn format
        # Split the file path by "/"
        parts = rotated_image_path.split("/")

        # Get the last part of the path (i.e., "ar_90_fixed.jpg")
        last_part = parts[-1]

        # Split the last part by "." and get the first part (i.e., "ar_90_fixed")
        result = last_part.split(".")[0]

        # Get the path of the .txt file
        input_file = f"{output_dir}/{result}_text_detection.txt"

        write_uzn_file(input_file, f"{source_folder}/{result}.uzn")

        # ready to run tesseract with CRAFT!
        craft_result = execute_tesseract(rotated_image_path, '--psm 4')
        print(f"CRAFT result: {craft_result}")
        craft_correct_words, truth_words, craft_word_accuracy = calculate_word_accuracy(craft_result, truth)
        print(f"craft word_accuracy: {craft_word_accuracy}")
        print(f"craft correct_words: {craft_correct_words}")
        print(f"truth_words: {truth_words}")

        total_craft_correct_words += craft_correct_words

        # crops_folder = f'{output_dir}/{result}_crops'

        # res = []
        # crops = os.listdir(crops_folder)
        # crops_sorted = sorted(crops, key=get_number_from_filename)
        #
        # for crop in crops_sorted:
        #     if crop.endswith('.jpg') or crop.endswith('.png'):
        #         ip = os.path.join(crops_folder, crop)
        #         tess_crop = tess(ip, cfg='--psm 7')
        #         print(f"{filename}: {tess_crop}")
        #         res.append(tess_crop)
        # crop_result = ' '.join(res)
        # print(f"CRAFT + tesseract result: {result}")
        #
        # crop_correct_words, truth_words, crop_word_accuracy = calculate_word_accuracy(crop_result, truth)
        # print(f"crop word_accuracy: {crop_word_accuracy}")
        # print(f"crop correct words: {crop_correct_words}")
        # print(f"truth_words: {truth_words}")
        #
        # total_crop_correct_words += crop_correct_words

        data = {
            "file": filename,
            "truth": truth,
            "truth_words": truth_words,
            "results": [
                {
                    "tesseract": {
                        "text": tess_result,
                        "word_accuracy": tess_word_accuracy,
                        "correct_words": tess_correct_words
                    }
                },
                {
                    "craft": {
                        "text": craft_result,
                        "word_accuracy": craft_word_accuracy,
                        "correct_words": craft_correct_words
                    }
                }
                # },
                # {
                #     "crop": {
                #         "text": crop_result,
                #         "word_accuracy": crop_word_accuracy,
                #         "correct_words": crop_correct_words
                #     }
                # }
            ]
        }

        json_data.append(data)

with open(result_file, 'w') as f:
    json.dump(json_data, f, indent=4, sort_keys=True)

print("total_truth_words: ", total_truth_words)
print("total tesseract correct words: ", total_tess_correct_words)
if total_truth_words > 0:
    print("total tesseract word accuracy: ", (total_tess_correct_words / total_truth_words) * 100)
print("total craft correct words: ", total_craft_correct_words)
if total_truth_words > 0:
    print("total craft word accuracy: ", (total_craft_correct_words / total_truth_words) * 100)
# print("total crop correct words: ", total_crop_correct_words)
# if total_truth_words > 0:
#     print("total crop word accuracy: ", (total_crop_correct_words / total_truth_words) * 100)
craft.unload_craftnet_model()
craft.unload_refinenet_model()