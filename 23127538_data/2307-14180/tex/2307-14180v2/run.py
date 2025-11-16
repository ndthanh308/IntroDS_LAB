import os
from PIL import Image

def convert_png_to_jpg(root_dir):
    # Traverse the root directory, and list all subdirectories and files
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith('.png'):
                # Define full path for the original .png and new .jpg files
                png_path = os.path.join(subdir, file)
                jpg_path = os.path.join(subdir, file.rsplit('.', 1)[0] + '.jpg')

                # Open the .png image and convert it to .jpg
                with Image.open(png_path) as img:
                    rgb_img = img.convert('RGB')  # Ensure it's in RGB format
                    rgb_img.save(jpg_path, 'JPEG')
                
                print(f"Converted {png_path} to {jpg_path}")

# Set the root directory where you want to start the conversion
root_directory = './'
convert_png_to_jpg(root_directory)

