import cv2
import numpy as np


cap = cv2.VideoCapture("night1.mp4")


fgbg = cv2.createBackgroundSubtractorMOG2()

contour_area=[]
while True:
    ret, frame = cap.read()
    if ret:
    #====================== switch and filter ================
        col_switch = cv2.cvtColor(frame, 70)
        lower = np.array([0,0,0])
        upper = np.array([40,10,255])   
        mask = cv2.inRange(col_switch, lower, upper)
        res = cv2.bitwise_and(col_switch,col_switch, mask= mask)
        #======================== get foreground mask=====================
        fgmask = fgbg.apply(res)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        # Dilate to merge adjacent blobs
        d_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        dilation = cv2.dilate(fgmask, d_kernel, iterations = 4)
        dilation[dilation < 255] = 0
        contours, hierarchy= cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        try:
            hierarchy = hierarchy[0]
        except:
            hierarchy = []

        for contour, hier in zip(contours, hierarchy):
            areas = [cv2.contourArea(c) for c in contours]
            for a in areas:
                if a > 10000 and a < 20000:
            #max_index1 = np.argmax(areas1)
            #cnt1 = contours[max_index1]
        
                    (x1, y1, w1, h1) = cv2.boundingRect(contour)
                    cx1 = int((w1 / 2) + x1)
                    cy1 = int((h1 / 2) + y1)

                    if w1 > 100 and h1 > 100:
                        cv2.rectangle(frame, (x1 - 10, y1 - 10), (x1 + w1, y1 + h1), (255, 255, 0), 2)
                        cv2.circle(dilation, (cx1, cy1), 5, (0, 255, 0), -1)
                        cv2.circle(frame, (cx1, cy1), 3, (0, 0, 255), 1)
        cv2.imshow("frame",frame)
        cv2.imshow("night",dilation)
        k = cv2.waitKey(1) & 0xff
        if k == ord('q'):
            break

        
cap.release()
cv2.destroyAllWindows()