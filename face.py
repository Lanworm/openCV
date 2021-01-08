import cv2
import numpy as np
from PIL import ImageGrab
import face_recognition

imgElon = face_recognition.load_image_file('ImagesBasic/me1.jpg')
imgElon = cv2.cvtColor(imgElon, cv2.COLOR_BGR2RGB)
# faceCascade = cv2.CascadeClassifier("Resources/haarcascade_frontalface_default.xml")
# cv2.namedWindow("settings")
# cv2.createTrackbar('scale', 'settings', 1, 10, lambda x: x)

while True:
    # img = np.array(ImageGrab.grab(bbox=(0, 32, 800, 600)))
    # scale = cv2.getTrackbarPos('scale', 'settings')
    # scale = 1.1+scale * 0.1
    # imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #
    # faces = faceCascade.detectMultiScale(imgGray, scale, 4)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #
    # for (x, y, w, h) in faces:
    #     cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    #
    # cv2.imshow("Result", img)

    imgTest = np.array(ImageGrab.grab(bbox=(0, 32, 800, 600)))
    imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)

    faceLoc = face_recognition.face_locations(imgElon)[0]
    encodeElon = face_recognition.face_encodings(imgElon)[0]
    cv2.rectangle(imgElon, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (255, 0, 255), 2)

    faceLocTest = face_recognition.face_locations(imgTest)[0]
    encodeTest = face_recognition.face_encodings(imgTest)[0]
    cv2.rectangle(imgTest, (faceLocTest[3], faceLocTest[0]), (faceLocTest[1], faceLocTest[2]), (255, 0, 255), 2)

    results = face_recognition.compare_faces([encodeElon], encodeTest)
    faceDis = face_recognition.face_distance([encodeElon], encodeTest)
    print(results, faceDis)
    cv2.putText(imgTest, f'{results} {round(faceDis[0], 2)}', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow('Elon Test', imgTest)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
