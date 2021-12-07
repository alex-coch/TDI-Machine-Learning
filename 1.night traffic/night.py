
import numpy as np
import timeit
import cv2
import datetime
import time 
import math


from line import get_points



global start,endtime
distance = 0.003

start = datetime.datetime.now()
endtime = datetime.datetime.now()

cap =cv2.VideoCapture("night1.mp4")

flag = True
# #Prompt user to draw 2 lines
_, img = cap.read()


line1,line2 = get_points(img)


# #for line 1
l1_x1,l1_x2,l1_y1,l1_y2 = line1[0][0],line1[1][0],line1[0][1],line1[1][1]

# #for line2
l2_x1,l2_x2,l2_y1,l2_y2 = line2[0][0],line2[1][0],line2[0][1],line2[1][1]

# #last check point for reference of centroid tracking
# '''find dist between frst 2 lines '''
starttrack = l1_y1
midtrack = l2_y1
lasttrack = int((midtrack-starttrack))
if lasttrack < 100 :
    lasttrack = (int(midtrack-starttrack)*3)+l2_y1
else:
    lasttrack = (int(midtrack-starttrack)*2)+l2_y1

print(starttrack)
print(lasttrack)


if cap.isOpened() == False:
  print("Video Not available")

while(cap.isOpened()):
  start = timeit.default_timer()
  ret, frame = cap.read()
  # framespersecond= int(cap.get(cv2.CAP_PROP_FPS))
  # print(framespersecond)
  if ret == True:
    h,w,_=(frame.shape)
    y=0
    x=0
    

    #roi = frame[int(y+h/2):int(y+h),x: int(x+w)]
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    gray = cv2.GaussianBlur(gray, (41, 41), 0)
   
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray)

    #to find speed 
    cx,cy = maxLoc
    print("cy",cy)
    if cy > starttrack:
      if flag is True and cy < midtrack:
        starttime= datetime.datetime.now()
        flag = False
      if cy> midtrack and cy< lasttrack:
        endtime = datetime.datetime.now()
        print('endtime', endtime)
        timedelta = (endtime - starttime).total_seconds()
        print(timedelta)
        speed = (distance/timedelta)
        print(speed)
    cv2.circle(frame, maxLoc, 10, (255, 0, 255), -1)
    cv2.line(frame, (l1_x1, l1_y1), (l1_x2, l1_y2), (0, 255, 0), 1)
    cv2.line(frame, (l2_x1, l2_y1), (l2_x2, l2_y2), (0, 0, 255), 1)
    cv2.line(frame, (l1_x1, int((lasttrack))), (l1_x2, int((lasttrack))),(0, 0, 0), 1)
    cv2.imshow("Robust", frame)
    
    if cv2.waitKey(25) & 0xFF == ord('q'):
      break

  # Break the loop
  else: 
    break
end = timeit.default_timer()
print(end)
cap.release()

# Closes all the frames
cv2.destroyAllWindows()