import cv2
import utlis
import requests
import numpy as np;


def empty(a):
    pass


###################################
cv2.namedWindow("parameters")
cv2.resizeWindow("parameters", 640, 240)
cv2.createTrackbar("Threshold1", "parameters", 23, 255, empty)
cv2.createTrackbar("Threshold2", "parameters", 20, 255, empty)
cv2.createTrackbar("Area", "parameters", 5000, 30000, empty)
###################################
url = 'http://192.168.0.104:8080/shot.jpg'


def stackImages(scale, imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]),
                                                None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank] * rows
        hor_con = [imageBlank] * rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None, scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(imgArray)
        ver = hor
    return ver


def getContours(img, imgContour):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(imgContour, contours, -1, (255, 0, 255), 7)
    for cnts in contours:
        area = cv2.contourArea(cnts)
        minArea = cv2.getTrackbarPos("Area", "parameters")

        if area > minArea:
            cv2.drawContours(imgContour, cnts, -1, (255, 0, 255), 7)
            peri = cv2.arcLength(cnts, True)
            approx = cv2.approxPolyDP(cnts, 0.02 * peri, True)
            x, y, w, h = cv2.boundingRect(approx)
            cv2.rectangle(imgContour, (x, y), (x + w, y + h), (0, 255, 0), 5)
            cv2.putText(imgContour, "Point: " + str(len(approx)), (x + w - 20, y + 20), cv2.FONT_HERSHEY_SIMPLEX, .7,
                        (0, 255, 0), 2)
            cv2.putText(imgContour, "Area: " + str(int(area)), (x - 20, y + 20), cv2.FONT_HERSHEY_SIMPLEX, .7,
                        (0, 255, 0), 2)


while True:
    raw_date = requests.get(url, verify=False)
    image_array = np.array(bytearray(raw_date.content), dtype=np.uint8)
    img = cv2.imdecode(image_array, -1)
    imgContour = img.copy()
    imgBlur = cv2.GaussianBlur(img, (7, 7), 1)
    imgGray = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2GRAY)
    threshold1 = cv2.getTrackbarPos("Threshold1", "parameters")
    threshold2 = cv2.getTrackbarPos("Threshold1", "parameters")
    imgCanny = cv2.Canny(imgGray, threshold1, threshold2)
    kernel = np.ones((8, 8))
    kernel2 = np.ones((5, 5))
    imgDil = cv2.dilate(imgCanny, kernel, iterations=1)
    imgErode = cv2.erode(imgDil, kernel2, iterations=1)
    getContours(imgErode, imgContour)
    imgStack = stackImages(0.8, ([img, imgCanny, imgGray],
                                 [imgDil, imgContour, imgErode]))

    cv2.imshow('Original', imgStack)
    cv2.waitKey(1)
