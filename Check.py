import cv2
import numpy as np
import os
import time
import pandas as pd

# Define the duration to run the face detection window (in seconds)
RUN_DURATION = 60  # 1 minute

size = 4
datasets = r"C:\Users\KIIT\OneDrive - kiit.ac.in\Desktop\6TH\minor\datasets"
print('Training...')

(images, labels, names, id) = ([], [], {}, 0)

# Add paths to Haar cascade classifier files
haarcascade_default_path = r"C:\Users\KIIT\OneDrive - kiit.ac.in\Desktop\6TH\minor\haarcascade_frontalface_default.xml"
haarcascade_alt_path = r"C:\Users\KIIT\OneDrive - kiit.ac.in\Desktop\6TH\minor\haarcascade_frontalface_alt.xml"
haarcascade_alt2_path = r"C:\Users\KIIT\OneDrive - kiit.ac.in\Desktop\6TH\minor\haarcascade_frontalface_alt2.xml"

# Load Haar cascade classifiers
face_cascade_default = cv2.CascadeClassifier(haarcascade_default_path)
face_cascade_alt = cv2.CascadeClassifier(haarcascade_alt_path)
face_cascade_alt2 = cv2.CascadeClassifier(haarcascade_alt2_path)

# Load dataset images
for (subdirs, dirs, files) in os.walk(datasets):
    for subdir in dirs:
        names[id] = subdir
        subjectpath = os.path.join(datasets, subdir)
        for filename in os.listdir(subjectpath):
            path = os.path.join(subjectpath, filename)
            label = id
            img = cv2.imread(path, 0)
            # Resize the image to ensure all images have the same dimensions
            img = cv2.resize(img, (130, 100))  # Use specific values or define width and height earlier
            images.append(img)
            labels.append(int(label))
        id += 1

(images, labels) = [np.array(lis) for lis in [images, labels]]

# Train the model
model = cv2.face.LBPHFaceRecognizer_create()
model.train(images, labels)

webcam = cv2.VideoCapture(0)
start_time = time.time()

while time.time() - start_time < RUN_DURATION:
    (_, im) = webcam.read()
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    
    # Detect faces using cascade classifiers
    faces_default = face_cascade_default.detectMultiScale(gray, 1.3, 5)
    faces_alt = face_cascade_alt.detectMultiScale(gray, 1.3, 5)
    faces_alt2 = face_cascade_alt2.detectMultiScale(gray, 1.3, 5)
    
    # Combine all detected faces
    faces_list = [faces_default, faces_alt, faces_alt2]
    faces = np.concatenate([face for face in faces_list if len(face) > 0])

    if len(faces) > 0:
        # Get only the first detected face
        (x, y, w, h) = faces[0]
        cv2.rectangle(im, (x, y), (x + w, y + h), (255, 255, 0), 2)
        face = gray[y:y + h, x:x + w]
        face_resize = cv2.resize(face, (130, 100))  # Use specific values or define width and height earlier
        prediction = model.predict(face_resize)
        
        cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 3)
        if prediction[1] < 800:
            cv2.putText(im, '%s - %.0f' % (names[prediction[0]], prediction[1]), (x - 10, y - 10),
                        cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 2)
            print(names[prediction[0]])
        else:
            cv2.putText(im, "Unknown", (x - 10, y - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
    
    cv2.imshow('OpenCV', im)
    key = cv2.waitKey(10)
    if key == ord("c"):
        break

webcam.release()
cv2.destroyAllWindows()
df = pd.DataFrame(output_data, columns=['Timestamp', 'Name', 'Confidence'])

# Save DataFrame to Excel file
output_file = 'face_recognition_output.xlsx'
df.to_excel(output_file, index=False)
print(f"Output saved to '{output_file}'")
