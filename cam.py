import cv2 as cv


cap = cv.VideoCapture(0,cv.CAP_DSHOW) #Change API settings
flag = cap.isOpened()

index = 1
while (flag):
    ret, frame = cap.read()
    frame = cv.flip(frame, 1) # Flip horizontally
    cv.imshow("Capture_Paizhao", frame)
    k = cv.waitKey(1) & 0xFF
    if k == ord('s'): # Press the s key to enter the save picture operation below
        cv.imwrite(r"F:\PyCharm\Camera calibration\Aruco_Identify\0" + str(index) + ".jpg", frame)
        print(cap.get(3))
        print(cap.get(4))
        print("save" + str(index) + ".jpg successfuly!")
        print("-------------------------")
        index += 1
    elif k == ord('q'): # Press the q key to exit the program
        break
cap.release()
cv.destroyAllWindows()