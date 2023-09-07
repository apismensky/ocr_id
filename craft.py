# import Craft class

from craft_text_detector import Craft
from pytesseract import Output
import pytesseract
import cv2
import os
import numpy as np

# Input image
image_path = 'images/sources/screen.png'
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
    d = pytesseract.image_to_data(img, output_type=Output.DICT, config='--psm 7')
    txt = ' '.join(d['text'])
    return txt


def get_number_from_filename(filename):
    try:
        number = int(filename.split("_")[1].split(".")[0])
        return number
    except (ValueError, IndexError):
        return float('inf')


def main():

    img = cv2.imread(image_path)
    angle = detect_text_orientation(img)

    rotated_image_path = image_path
    if angle > 0:
        print(f"angle: {angle}")
        rotate_bound(img, angle)
        rotated_image_path = image_path[:-4] + '_fixed.jpg'

    rotated_image_name = os.path.basename(rotated_image_path)
    #rotated_image_name_no_ext = os.path.splitext(rotated_image_name)[0]

    craft.detect_text(rotated_image_path)

    #crops_folder = f'{output_dir}{rotated_image_name_no_ext}_crops'

    #res = []
    #filenames = os.listdir(crops_folder)
    #filenames = sorted(filenames, key=get_number_from_filename)

    # for filename in filenames:
    #     if filename.endswith('.jpg') or filename.endswith('.png'):
    #         img_p = os.path.join(crops_folder, filename)
    #         tess_result = tess(img_p)
    #         print(f"{filename}: {tess_result}")
    #         res.append(tess_result)
    # result = ' '.join(res)
    # print(f"CRAFT + tesseract result: {result}")


if __name__ == "__main__":
    main()

craft.unload_craftnet_model()
craft.unload_refinenet_model()