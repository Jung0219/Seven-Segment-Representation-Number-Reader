import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread(
    r"C:\Users\OWNER\Downloads\python\OpenCV\images\sevensegmentrepresentation\clock3.jpg")
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
thresh = cv.threshold(gray, 210, 255, cv.THRESH_BINARY_INV)[1]

contours = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)[0]
largest = sorted(contours, key=cv.contourArea, reverse=True)[0]

rect = cv.minAreaRect(largest)
box = cv.boxPoints(rect)
box = np.intp(box)
boxPoints = np.array(box, dtype=np.float32)

# (x, y) coordinates
topLeft = box[0]
topRight = box[1]
botRight = box[2]
botLeft = box[3]

x1 = int(np.sqrt((topRight[0] - topLeft[0]) **
         2 + (topRight[1] - topLeft[1]) ** 2))
x2 = int(np.sqrt((botRight[0] - botLeft[0]) **
         2 + (topRight[1] - topLeft[1]) ** 2))
y1 = int(np.sqrt((topRight[0] - botRight[0]) **
         2 + (topRight[1] - botRight[1]) ** 2))
y2 = int(np.sqrt((topLeft[0] - botLeft[0]) **
         2 + (topLeft[1] - botLeft[1]) ** 2))

w = max(x1, x2)
h = max(y1, y2)

# perspective transform
dst = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
M = cv.getPerspectiveTransform(boxPoints, dst)
warped = cv.warpPerspective(thresh, M, (w, h))
warpedImg = cv.warpPerspective(img, M, (w, h))

# fill out the gaps
k = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
opened = cv.morphologyEx(warped, cv.MORPH_OPEN, k, iterations=2)
opened = np.invert(opened)

contours = cv.findContours(opened, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)[0]
digits = []

# finding digits
for contour in contours:
    x, y, w, h = cv.boundingRect(contour)
    if w > 15 and w < 300 and h > 92 and h < 130:
        digits.append(contour)

# sorting them from left to right
boxes = [cv.boundingRect(c) for c in digits]
(digits, _) = zip(*sorted(zip(digits, boxes), key=lambda c: c[1][0]))

numberMap = {
    (1, 1, 1, 0, 1, 1, 1): 0,
    (0, 0, 1, 0, 0, 1, 0): 1,
    (1, 0, 1, 1, 1, 0, 1): 2,
    (1, 0, 1, 1, 0, 1, 1): 3,
    (0, 1, 1, 1, 0, 1, 0): 4,
    (1, 1, 0, 1, 0, 1, 1): 5,
    (1, 1, 0, 1, 1, 1, 1): 6,
    (1, 1, 1, 0, 0, 1, 0): 7,
    (1, 1, 1, 1, 1, 1, 1): 8,
    (1, 1, 1, 0, 1, 1, 0): 9,
}

# for loop over digits
for digit in digits:
    x, y, w, h = cv.boundingRect(digit)

    # make an exception for 1 (narrower boundingrect)
    if h // w > 4:
        cv.rectangle(warpedImg, (x, y), (x + w, y + h), (0, 255, 0), 1)
        cv.putText(warpedImg, "{}".format(1), (x, y-10),
                   cv.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.5, (0, 255, 0), 1)
        continue

    thickness = int(h * 0.15)
    roi = opened[y:y+h, x:x+w]

    # calculate segment positions
    segments = [
        ((0, 0), (w, thickness)),  # top
        ((0, 0), (thickness, int((h - thickness) * 0.5))),  # top left
        ((w - thickness, 0), (w, int((h - thickness) * 0.5))),  # top right
        ((0, int((h - thickness) * 0.5)), (w, int((h + thickness) * 0.5))),  # middle
        ((0, int((h + thickness) * 0.5)), (thickness, h)),  # bottom left
        ((w - thickness, int((h + thickness) * 0.5)), (w, h)),  # bottom right
        ((0, h - thickness), (w, h))  # bottom
    ]

    display = ()

    # check whether segment is on
    for (xA, yA), (xB, yB) in segments:
        segmentROI = roi[yA:yB, xA:xB]
        total = cv.countNonZero(segmentROI)
        area = (xB - xA) * (yB - yA)

        # more than half of the area is white
        if total > area * 0.5:
            display += (1,)
        else:
            display += (0,)

    cv.rectangle(warpedImg, (x, y), (x + w, y + h), (0, 255, 0), 1)
    cv.putText(warpedImg, "{}".format(
        numberMap[display]), (x, y - 10), cv.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.5, (0, 255, 0), 1)


cv.imwrite("result3.jpg", warpedImg)

plt.figure(figsize=[15, 15])
plt.imshow(warpedImg)
plt.waitforbuttonpress()
plt.close("all")
