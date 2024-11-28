# Import necessary libraries
import cv2
import os
import numpy as np
import face_recognition
import numpy as np
import json
from datetime import datetime
import pandas as pd


# Initialize the camera
cap = cv2.VideoCapture(0)

# Capture the image
while True:
    ret, frame = cap.read()
    cv2.imshow("frame", frame)
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

# Load the image
img = cv2.imread("image.jpg")

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to the image
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# Apply adaptive thresholding to the image
thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

# Apply morphological operations to the image
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
dilate = cv2.dilate(thresh, kernel, iterations=3)
erode = cv2.erode(dilate, kernel, iterations=3)

# Display the image
cv2.imshow("image", erode)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Initialize the camera
cap = cv2.VideoCapture(0)

# Create a directory for storing the dataset
if not os.path.exists("dataset"):
    os.makedirs("dataset")

# Create a dataset
count = 0
while True:
    ret, frame = cap.read()
    cv2.imshow("frame", frame)
    if cv2.waitKey(1) == ord('q'):
        break
    if cv2.waitKey(1) == ord('s'):
        count += 1
        filename = "dataset/user_" + str(count) + ".jpg"
        cv2.imwrite(filename, frame)
cap.release()
cv2.destroyAllWindows()

# Create a directory for storing the embeddings
if not os.path.exists("embeddings"):
    os.makedirs("embeddings")

# Load the images
image_paths = [os.path.join("dataset", f) for f in os.listdir("dataset")]
images = []
for image_path in image_paths:
    image = face_recognition.load_image_file(image_path)
    images.append(image)

# Compute the face embeddings
embeddings = []
for image in images:
    face_locations = face_recognition.face_locations(image)
    face_encodings = face_recognition.face_encodings(image, face_locations)
    if len(face_encodings) == 1:
        embeddings.append(face_encodings[0])

# Save the embeddings
np.savetxt("embeddings/embeddings.txt", embeddings)

# Load the embeddings
embeddings = np.loadtxt("embeddings/embeddings.txt")
# Load the attendance log
if not os.path.exists("attendance.json"):
    with open("attendance.json", "w") as f:
        json.dump({}, f)
with open("attendance.json", "r") as f:
    attendance = json.load(f)
# Recognize the faces and mark attendance
while True:
    ret, frame = cap.read()
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(embeddings, face_encoding)
        if True in matches:
            index = matches.index(True)
            name = "user_" + str(index+1)
            if name not in attendance:
                attendance[name] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                with open("attendance.json", "w") as f:
                    json.dump(attendance, f)
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.imshow("frame", frame)
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

# Load the attendance log from the JSON file
with open('attendance.json', 'r') as f:
    attendance_log = json.load(f)

# Convert the attendance log to a Pandas dataframe
df = pd.DataFrame(attendance_log)

# Write the attendance data to an Excel file
writer = pd.ExcelWriter('attendance_log.xlsx')
df.to_excel(writer, index=False)
writer.save()


