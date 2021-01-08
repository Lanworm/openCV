import numpy as np
# import pytesseract
from PIL import ImageGrab
import cv2
import time

from text_detection import text_d


def screen_record():
    fps = 0
    real = 0
    last_time = time.time()
    cv2.namedWindow("settings", cv2.WINDOW_NORMAL)  # создаем окно настроек

    # создаем 6 бегунков для настройки начального и конечного цвета фильтра
    cv2.createTrackbar('h1', 'settings', 0, 255, lambda x: x)
    cv2.createTrackbar('s1', 'settings', 167, 255, lambda x: x)
    cv2.createTrackbar('v1', 'settings', 128, 255, lambda x: x)
    cv2.createTrackbar('h2', 'settings', 5, 255, lambda x: x)
    cv2.createTrackbar('s2', 'settings', 255, 255, lambda x: x)
    cv2.createTrackbar('v2', 'settings', 218, 255, lambda x: x)
    cv2.createTrackbar('view', 'settings', 0, 5, lambda x: x)
    cv2.createTrackbar('convert_h', 'settings', 30, 225, lambda x: x)
    cv2.createTrackbar('convert_c', 'settings', 0, 225, lambda x: x)
    cv2.createTrackbar('convert_v', 'settings', 255, 225, lambda x: x)
    cv2.createTrackbar('threshold_h', 'settings', 75, 225, lambda x: x)
    cv2.createTrackbar('threshold_c', 'settings', 225, 225, lambda x: x)
    cv2.createTrackbar('threshold_v', 'settings', 200, 225, lambda x: x)
    cv2.createTrackbar('w_limit', 'settings', 50, 225, lambda x: x)
    cv2.createTrackbar('h_limit', 'settings', 50, 225, lambda x: x)
    while (True):
        printscreen = np.array(ImageGrab.grab(bbox=(0, 32, 800, 1080)))

        # считываем значения бегунков
        h1 = cv2.getTrackbarPos('h1', 'settings')
        s1 = cv2.getTrackbarPos('s1', 'settings')
        v1 = cv2.getTrackbarPos('v1', 'settings')
        h2 = cv2.getTrackbarPos('h2', 'settings')
        s2 = cv2.getTrackbarPos('s2', 'settings')
        v2 = cv2.getTrackbarPos('v2', 'settings')
        view = cv2.getTrackbarPos('view', 'settings')
        convert_h = cv2.getTrackbarPos('convert_h', 'settings')
        convert_c = cv2.getTrackbarPos('convert_c', 'settings')
        convert_v = cv2.getTrackbarPos('convert_v', 'settings')
        threshold_h = cv2.getTrackbarPos('threshold_h', 'settings')
        threshold_c = cv2.getTrackbarPos('threshold_c', 'settings')
        threshold_v = cv2.getTrackbarPos('threshold_v', 'settings')
        w_limit = cv2.getTrackbarPos('w_limit', 'settings')
        h_limit = cv2.getTrackbarPos('h_limit', 'settings')

        # формируем начальный и конечный цвет фильтра
        h_min = np.array((h1, s1, v1), np.uint8)
        h_max = np.array((h2, s2, v2), np.uint8)


        printscreen = cv2.cvtColor(printscreen, cv2.COLOR_BGR2RGB)
        hsv_img = cv2.cvtColor(printscreen, cv2.COLOR_BGR2HSV)

        green_low = h_min
        green_high = h_max
        curr_mask = cv2.inRange(hsv_img, green_low, green_high)
        hsv_img[curr_mask == 0] = ([convert_h, convert_c, convert_v])

        RGB_again = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2RGB)
        gray = cv2.cvtColor(RGB_again, cv2.COLOR_RGB2GRAY)

        ret, threshold = cv2.threshold(gray, threshold_h, threshold_c, threshold_v)

        contours, hierarchy = cv2.findContours(curr_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # cv2.drawContours(printscreen, contours, -1, (0, 0, 255), 3)

        def get_w(c):
            x1, y1, w1, h1 = cv2.boundingRect(c)
            return w1 > w_limit and h1 < h_limit

        contours = [item for item in contours if get_w(item)]

        total_contours = len(contours)
        i = 0
        while (i < total_contours):
            x, y, w, h = cv2.boundingRect(contours[i])
            cv2.rectangle(printscreen, (x, y), (x + w, y + h), (0, 255, 0), 1)
            cv2.putText(printscreen, '{0:.2f}%'.format(100/74*w), (x, y + 20), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)
            i += 1

        views = [printscreen, hsv_img, curr_mask, gray, threshold, text_d(threshold)]

        # if time.time() - last_time > 10:
        #     last_time = time.time()
        #     text = pytesseract.image_to_string(curr_mask, lang="rus")
        #     print(text)


        cv2.imshow('window', views[view])
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break


screen_record()
