import pytesseract
from pytesseract import Output
import cv2
import os
import easyocr
import json
# from PIL import Image
#import dlib
import numpy as np
from craft_text_detector import Craft
import time

source_folder = 'images/sources'
tesseract_folder = 'images/boxes_tesseract'
result_file = 'result.json'
output_dir = 'images/boxes_craft'

# create a craft instance
craft = Craft(output_dir=output_dir, crop_type="poly", cuda=False)


# Load the EasyOCR model
#reader = easyocr.Reader(['en'])
# Define the threshold value for angle detection by tesseract
threshold = 1.0  # Adjust this value as needed


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


# def easy(img_path):
#     print("img_path: ", img_path)
#     i = cv2.imread(img_path)
#     i = detect_faces_and_rotate(i, 4)
#     results = reader.readtext(i)
#     texts = []
#     for result in results:
#         x1, y1, x2, y2 = result[0][0][0], result[0][0][1], result[0][2][0], result[0][2][1]
#         cv2.rectangle(i, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 1)
#         texts.append(result[1].strip())
#         cv2.imwrite(os.path.join(easy_folder, os.path.basename(img_path)), i)
#     result = ' '.join(texts)
#     print(f"EasyOCR: {result}")
#     return result



if not os.path.exists(tesseract_folder):
    os.makedirs(tesseract_folder)

# Create a list to store the JSON data
json_data = []


def get_truth_file_name(fname):
    parts = fname.split("_")[0]
    if len(parts) == len(fname):
        parts = fname.split(".")[0]

    return parts


# def detect_text_orientation(img):
#     try:
#         osd = pytesseract.image_to_osd(img, output_type=Output.DICT)
#     except Exception as e:
#         print("error: ", e)
#         return 0
#     print(f'OSD result: {osd}')
#     if osd['orientation_conf'] >= threshold:
#         return osd['orientation']
#     else:
#         return 0


def get_truth(fname):
    truth_file_name = get_truth_file_name(fname)
    truth_path = os.path.join(source_folder, truth_file_name + ".txt")
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
        print(f"The file '{file_path}' does not exist.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
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

total_truth_words = 0
total_tess_correct_words = 0
total_craft_correct_words = 0

for filename in os.listdir(source_folder):
    #if filename.endswith('ar_90.jpg'):
    if filename.endswith('.jpg') or filename.endswith('.png'):

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
        # angle = detect_text_orientation(img)

        # rotate the image if the angle is not 0
        rotated_image_path = img_p
        # if angle > 0:
        #     print(f"rotating image {img_p} angle: {angle}")
        #     rotated = rotate_bound(img, angle)
        #     rotated_image_path = img_p[:-4] + '_fixed.jpg'
        #     cv2.imwrite(rotated_image_path, rotated)
        #
        # print("rotated_image_path: ", rotated_image_path)

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
        craft_result = tess(rotated_image_path, '--psm 4')
        craft_correct_words, truth_words, craft_word_accuracy = calculate_word_accuracy(craft_result, truth)
        print(f"craft word_accuracy: {craft_word_accuracy}")
        print(f"craft correct_words: {craft_correct_words}")
        print(f"truth_words: {truth_words}")

        total_craft_correct_words += craft_correct_words

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
            ]
        }

        json_data.append(data)

with open(result_file, 'w') as f:
    json.dump(json_data, f, indent=4, sort_keys=True)

print("total_truth_words: ", total_truth_words)
print("total tesseract correct words: ", total_tess_correct_words)
print("total tesseract word accuracy: ", (total_tess_correct_words / total_truth_words) * 100)
print("total craft correct words: ", total_craft_correct_words)
print("total craft word accuracy: ", (total_craft_correct_words / total_truth_words) * 100)
craft.unload_craftnet_model()
craft.unload_refinenet_model()