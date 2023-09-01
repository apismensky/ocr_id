# script to create rotated images from source images
# store in the same folder as source images
# add extension to filename indicate rotation
import os
from PIL import Image

source_folder = 'images/sources'

for filename in os.listdir(source_folder):
    if filename.endswith('.jpg')  or filename.endswith('.png'):
        img = Image.open(os.path.join(source_folder, filename))
        img_90 = img.transpose(Image.ROTATE_90)
        img_180 = img.transpose(Image.ROTATE_180)
        img_270 = img.transpose(Image.ROTATE_270)
        img_90.convert('RGB').save(os.path.join(source_folder, filename[:-4] + '_90.jpg'))
        img_180.convert('RGB').save(os.path.join(source_folder, filename[:-4] + '_180.jpg'))
        img_270.convert('RGB').save(os.path.join(source_folder, filename[:-4] + '_270.jpg'))
