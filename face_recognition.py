#-------------------------------------------------------------------------------
# Name:        face_recognition.py
# Author:      Jerome Vonk
#
# Created:     04/05/2018
#
# Based on: https://www.superdatascience.com/opencv-face-recognition/
#-------------------------------------------------------------------------------

import cv2
import os
import numpy as np

# These will be our labels
subjects = ["Luiz IL da Silva", "Jerome Vonk", "Elvis Presley", "George HW Bush"]

# ----------------------------------------------------------------------------------
# Draw rectangle on image
# according to given (x, y) coordinates and given width and heigh
# ----------------------------------------------------------------------------------
def draw_rectangle(img, rect):
    '''Draw a rectangle on the image'''
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)

# ----------------------------------------------------------------------------------
# Write text on given image starting from (x, y) coordinates.
# ----------------------------------------------------------------------------------
def write_text(img, text, x, y):
    '''Draw text on the image'''
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 255), 2)

# ----------------------------------------------------------------------------------
# Predict who's in the image and draw a rectangle around the face
# plus the name of the subject
# ----------------------------------------------------------------------------------
def predict(test_img):
    '''Predict who's face is in the image'''

    # Make a copy (don't change original image)
    img = test_img.copy()

    # Detect face from the image
    face, rect = detect_face(img)

    # Predict the image using our face recognizer
    label = face_recognizer.predict(face)

    # Get the name that corresponds to the label
    label_text = subjects[label[0]]

    # Draw a rectangle around detected face
    draw_rectangle(img, rect)

    # Write predicted person's name
    write_text(img, label_text, rect[0] - 40, rect[1] - 5)

    return img

# ----------------------------------------------------------------------------------
# Detect face using OpenCV
# ----------------------------------------------------------------------------------
def detect_face(img):
    '''Detect face in an image'''

    # Convert the test image to gray scale (opencv face detector expects gray images)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Load OpenCV face detector (LBP is faster)
    face_cascade = cv2.CascadeClassifier('opencv-files/lbpcascade_frontalface.xml')

    # Detect multiscale images (some images may be closer to camera than others)
    # result is a list of faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5);

    # If not face detected, return None
    if  len(faces) == 0:
        return None, None

    # Assumingthat there will be only one face per image extract the face area
    (x, y, w, h) = faces[0]

    # Return the face image area and the face rectangle
    return gray[y:y + w, x:x + h], faces[0]

# ---------------------------------------------------------------------------------
# Read all training images from the training_data folder
# ---------------------------------------------------------------------------------
def prepare_training_data(training_path):
    '''Return list of faces(features) and list of targets'''

    # Get the directories in data folder
    dirs = os.listdir(training_path)

    # Hold all faces and labels
    faces  = []
    labels = []

    # Iterare through directories and read images within it
    for dir_name in dirs:

        # Extract label number from
        label = int(dir_name)

        # Path of directory containing images for current subject
        subject_dir_path = training_path + "/" + dir_name

        # Get a list of imagems inside the directory
        images = os.listdir(subject_dir_path)

        # For each image, detect the face and add to the list
        for image in images:

            # build image path
            image_path = subject_dir_path + "/" + image

            # Read image
            image = cv2.imread(image_path)

            # display an image window to show the image
            cv2.imshow("Training on image...", image)
            cv2.waitKey(25)

            # Detect face
            face, rect = detect_face(image)

            # Ignore pictures with no face detected
            if face is not None:

                # Add to list of faces
                faces.append(face)

                # Add to list of labels
                labels.append(label)
            else:
                print("No face detected for image {}/{}".format(subject_dir_path, image))

    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.destroyAllWindows()

    return faces, labels

# --------------------------------------------------------
# Prepare training data
# --------------------------------------------------------
print("Preparing data...")
faces, labels = prepare_training_data("training_data")
print("Data prepared")

# Print total faces and labels
print("Total faces: ", len(faces))
print("Total labels: ", len(labels))

# --------------------------------------------------------
# Train using the LBPH face recognizer
#
# Options are:
# - cv2.face.EigenFaceRecognizer_create()
# - cv2.face.FisherFaceRecognizer_create()
# - cv2.face.LBPHFaceRecognizer_create()

# For Eigenfaces and Fisherfaces recognizers,
# all input samples must be of equal size.
# --------------------------------------------------------

# Create the recognizer
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

# Train the recognizer
face_recognizer.train(faces, np.array(labels))

# --------------------------------------------------------
# Predict
# --------------------------------------------------------
print("Predicting images...")

for i in range(1, 5):
    # load test images
    test_img = cv2.imread("test_data/test{}.jpg".format(i))

    # perform a prediction
    predicted_img = predict(test_img)

    # display image
    cv2.imshow('Test {}'.format(i), predicted_img)


cv2.waitKey(0)
cv2.destroyAllWindows()
print("Prediction complete")
