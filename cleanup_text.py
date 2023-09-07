import os
from PIL import Image

source_folder = 'images/sources'

new_line_delimiter = '\n'  # Change this to your desired delimiter


def clean():
    for filename in os.listdir(source_folder):
        if filename.endswith('.txt'):
            # read file and clean text
            file_path = os.path.join(source_folder, filename)
            with open(file_path, "r") as f:
                text = f.read()
                # Replace '\n' with the desired end-of-line delimiter
                text = text.replace('\\n', new_line_delimiter)

                with open(file_path, "w") as fw:
                    fw.write(text)


clean()
