import cv2 as cv
import numpy as np
import os
from skimage.feature import hog
from skimage import io
from sklearn import preprocessing
import xml.etree.ElementTree as ET
from xml.dom import minidom

def load_images_from_folder(folder):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img = io.imread(os.path.join(folder, filename), as_gray=True)
        if img is not None:
            images.append(img)
            # Assuming filenames are like '0_rotation_0.png', '1_rotation_45.png', etc.
            label = int(filename.split('_')[0])
            labels.append(label)
    return images, labels

def extract_features(images):
    feature_list = []
    for image in images:
        # Using HOG features for each image
        features = hog(image, pixels_per_cell=(14, 14), cells_per_block=(1, 1), feature_vector=True)
        feature_list.append(features)
    return feature_list


# Function to insert newlines in the data string after every 'n' numbers
def insert_newlines(string, every=20):
    lines = []
    numbers = string.split()
    for i in range(0, len(numbers), every):
        line = ' '.join(numbers[i:i+every])
        lines.append(line)
    return '\n'.join(lines)



# Load images and labels
folder = "generated_numbers"  # Update this path
images, labels = load_images_from_folder(folder)

# Extract features from the images
features = extract_features(images)

# Standardize features
scaler = preprocessing.StandardScaler().fit(features)
features = scaler.transform(features)



# Prepare the data string with newlines for features
data_str = insert_newlines(' '.join(' '.join(str(num) for num in row) for row in features))

# Prepare the data string with newlines for labels
labels_str = insert_newlines(' '.join(str(lab) for lab in labels))


# Manually create XML file in the specified format
opencv_storage = ET.Element('opencv_storage')
hus_element = ET.SubElement(opencv_storage, 'hus', attrib={'type_id': 'opencv-matrix'})
ET.SubElement(hus_element, 'rows').text = str(len(features))
ET.SubElement(hus_element, 'cols').text = str(len(features[0]))
ET.SubElement(hus_element, 'dt').text = 'f'
ET.SubElement(hus_element, 'data').text = data_str

labels_element = ET.SubElement(opencv_storage, 'labels', attrib={'type_id': 'opencv-matrix'})
ET.SubElement(labels_element, 'rows').text = str(len(labels))
ET.SubElement(labels_element, 'cols').text = '1'
ET.SubElement(labels_element, 'dt').text = 'f'
ET.SubElement(labels_element, 'data').text = labels_str












# Convert to a string with pretty formatting
rough_string = ET.tostring(opencv_storage, 'utf-8')
reparsed = minidom.parseString(rough_string)
pretty_string = reparsed.toprettyxml(indent="  ")

# Write to XML file with declaration and pretty print
with open('..\OpenCV\Assets\OpenCV+Unity\Demo\OCR.Alphabet\custom_format.xml', 'w') as f:
    f.write(pretty_string)
