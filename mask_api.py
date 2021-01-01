from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2

prototxtPath = "C:\\Users/DELL/PycharmProjects/Exam/face_detector/deploy.prototxt"
weightsPath = "C:\\Users/DELL/PycharmProjects/Exam/face_detector/res10_300x300_ssd_iter_140000.caffemodel"
net = cv2.dnn.readNet(prototxtPath, weightsPath)
print("[INFO] loading face mask detector model...")
model = load_model("C:\\Users/DELL/PycharmProjects/Exam/face_detector/mask_detector.model")

image = cv2.imread("C:\\Users/DELL/PycharmProjects/Exam/face_detector/text.jpg")
# image.copy()
(h, w) = image.shape[:2]
blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))
print("[INFO] computing face detections...")
net.setInput(blob)
detections = net.forward()

for i in range(0, detections.shape[2]):
    # extract the confidence (i.e., probability) associated with
    confidence = detections[0, 0, i, 2]
    # filter out weak detections by ensuring the confidence is
    # greater than the minimum confidence
    if confidence > 0.5:
        # compute the (x, y)-coordinates of the bounding box for
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
        # ensure the bounding boxes fall within the dimensions of
        (startX, startY) = (max(0, startX), max(0, startY))
        (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
        face = image[startY:endY, startX:endX]
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = cv2.resize(face, (224, 224))
        face = img_to_array(face)
        face = preprocess_input(face)
        face = np.expand_dims(face, axis=0)
        # pass the face through the model to determine if the face
        (mask, withoutMask) = model.predict(face)[0]
        # determine the class label and color we'll use to draw
        # the bounding box and text
        mask_list = []
        withoutmask_list = []
        if mask > withoutMask:
            label = "Mask"
            mask_list.append(label)

        else:
            label = "No Mask"
            withoutmask_list.append(label)
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
        # include the probability in the label
        label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
        # display the label and bounding box rectangle on the output
        # frame
        cv2.putText(image, label, (startX, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)

cv2.imshow("data.jpg", image)
print("Mask Conunting is:",len(mask_list))
print("Without Mask Conunting is:",len(withoutmask_list))
cv2.imwrite("data.jpg", image)
# cv2.imshow("Output.png", image)
cv2.waitKey(0)

