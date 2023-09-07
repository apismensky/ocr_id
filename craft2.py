# import Craft class

from craft_text_detector import Craft
from pytesseract import Output
import pytesseract
import cv2
import os
import numpy as np
import time

# Input image
image_name = 'ar_90.jpg'
image_path = f'images/sources/{image_name}'
output_dir = 'outputs/'

# create a craft instance
craft = Craft(output_dir=output_dir, crop_type="poly", cuda=False)


def rotate_bound(img, angle):
    # grab the dimensions of the image and then determine the centre
    (h, w) = img.shape[:2]
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
    affine = cv2.warpAffine(img, m, (nw, nh))
    rotated_image_path = image_path[:-4] + '_fixed.jpg'
    cv2.imwrite(rotated_image_path, affine)
    return affine


def detect_text_orientation(img):
    try:
        osd = pytesseract.image_to_osd(img, output_type=Output.DICT)
    except Exception as e:
        print("error: ", e)
        return 0
    print(f'OSD result: {osd}')
    if osd['orientation_conf'] >= 1:
        return osd['orientation']
    else:
        return 0


def tess(path):
    img = cv2.imread(path)
    d = pytesseract.image_to_data(img, output_type=Output.DICT, config='--psm 4')
    txt = ' '.join(d['text'])
    return txt


def get_number_from_filename(filename):
    try:
        number = int(filename.split("_")[1].split(".")[0])
        return number
    except (ValueError, IndexError):
        return float('inf')


def calculate_dimensions(x_coordinates, y_coordinates):
    left = min(x_coordinates)
    top = min(y_coordinates)
    right = max(x_coordinates)
    bottom = max(y_coordinates)
    width = right - left
    height = bottom - top
    return left, top, width, height

def main():

    start_time = time.time()
    img = cv2.imread(image_path)
    print("cv2 imread: --- %s seconds ---" % (time.time() - start_time))

    #start_time = time.time()
    #angle = detect_text_orientation(img)
    #print("detect_text_orientation: --- %s seconds ---" % (time.time() - start_time))

    rotated_image_path = image_path
    #if angle > 0:
    #    print(f"angle: {angle}")
    #    start_time = time.time()
    #    rotate_bound(img, angle)
    #    rotated_image_path = image_path[:-4] + '_fixed.jpg'
    #    print("rotation: --- %s seconds ---" % (time.time() - start_time))

    print("rotated_image_path: ", rotated_image_path)

    start_time = time.time()
    craft.detect_text(rotated_image_path)
    print("CRAFT detect_text time: --- %s seconds ---" % (time.time() - start_time))

    input_file = f"{output_dir}ar_90_text_detection.txt"

    output_file = "images/sources/ar_90.uzn"

    with open(input_file, "r") as infile, open(output_file, "w") as outfile:
        for line in infile:
            # Split the line into individual coordinate values
            coordinates = list(map(int, line.strip().split(',')))

            # Split the coordinates into x and y lists
            x_coordinates = coordinates[::2]
            y_coordinates = coordinates[1::2]

            # Calculate the dimensions
            left, top, width, height = calculate_dimensions(x_coordinates, y_coordinates)

            # Write the transformed data to the output file
            outfile.write(f'{left} {top} {width} {height} Text/Latin\n')
    #start_time = time.time()
    #tess_result = tess(rotated_image_path)
    #print("tesseract time: --- %s seconds ---" % (time.time() - start_time))
    #print(f"tesseract: {tess_result}")


if __name__ == "__main__":
    main()

craft.unload_craftnet_model()
craft.unload_refinenet_model()