import numpy as np
import cv2
import timeit
import datetime
import os
import time
from line import get_points
import pickle
import time
import math


#for SERVER SIDE
import json                    
#import requests
#convert img to JSON object
import base64
import pickle


#API endpoint
api = 'https://tdispeeddetection.free.beeceptor.com/success'



speed_limit = int(input('Enter The Speed Limit: '))
distance =int(input('Enter distance between 2 lines in Meters: '))





def show_angle(distance):
    if distance !=0:
        show_direction = cv2.imread("PromptAngleinfo.JPG")
        cv2.imshow("Angle Help",show_direction)
        k = cv2.waitKey(1) & 0xff
        cv2.waitKey(100)
        Angle = int(input("Enter apporximate Angle with road :")) 

        return Angle
#Prompts user with demo image for choosing right angle.
show_angle(distance)



# Initialize the video
cap = cv2.VideoCapture('night1.mp4')
fps = cap.get(cv2.CAP_PROP_FPS)

lane_1_1 = []
lane_1_2 = []

#collect mask 

road_cropped = "regions.p"
with open(road_cropped,'rb')as f:
    mask_list =pickle.load(f)
    print(mask_list[0])
    print(mask_list[1])

#getting mask
mask1 = cv2.imread('m1.jpeg')
mask1 = cv2.cvtColor(mask1, cv2.COLOR_BGR2GRAY)
ret1, thresh_MASK_1 = cv2.threshold(mask1, 127, 255, cv2.THRESH_BINARY_INV)
mask2 = cv2.imread('m2.jpeg')
mask2 = cv2.cvtColor(mask2, cv2.COLOR_BGR2GRAY)
ret2, thresh_MASK_2 = cv2.threshold(mask2, 127, 255, cv2.THRESH_BINARY_INV)

# Create the background subtraction object
method = 1

if method == 0:
    bgSubtractor = cv2.bgsegm.createBackgroundSubtractorMOG()
elif method == 1:
    bgSubtractor = cv2.createBackgroundSubtractorMOG2()
else:
    bgSubtractor = cv2.bgsegm.createBackgroundSubtractorGMG()

# Create the kernel that will be used to remove the noise in the foreground mask
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
kernel_di = np.ones((5, 1), np.uint8)

# define variables
cnt = 0
cnt1 = 0
flag = True
flag1 = True
3


#store X,Y  for lne 1
X1=[]
X2=[]

Y1=[]
Y2=[]

#store X_,Y_ for lane2

X1_=[]
X2_=[]

Y1_=[]
Y2_=[]

# Prompt user to draw 2 lines
_, img = cap.read()

line1, line2 = get_points(img)


#for line 1
l1_x1,l1_x2,l1_y1,l1_y2 = line1[0][0],line1[1][0],line1[0][1],line1[1][1]

#for line2
l2_x1,l2_x2,l2_y1,l2_y2 = line2[0][0],line2[1][0],line2[0][1],line2[1][1]

mid_pt = int((l2_y2 - l1_y1)/2)
print("mid_pt",mid_pt)
print(l1_y1+mid_pt)




pixel_distance = (l2_y2-l1_y1) #pixels travel in Y axis by centroid (Theoritically displacement)

print("pixel_distance",pixel_distance)



area_s=[]
# Play until the user decides to stop
#for sending data to server
# def send(img):
#     retval, buffer = cv2.imencode(".jpg", img)
#     img = base64.b64encode(buffer).decode('utf-8')
#     data = json.dumps({"image1": img, "id" : "2345AB"})
#     response = requests.post(api, data=data, timeout=5, headers = {'Content-type': 'application/json', 'Accept': 'text/plain'})
#     try:
#        data = response.json()     
#        print(data)                
#     except requests.exceptions.RequestException:
#        print(response.text)





API_ENDPOINT = api # replace this endpoint with your own


######################################################CAL FRAME seen for vehicle
#no of frame vechicle appears in lane1
frame_cntr =0

#no of frames vehicle appears in lane2

frame_cntr2=0




while True:
    start = timeit.default_timer()
    ret, frame = cap.read()
    frame_og = frame
    l, a, b = cv2.split(frame)
    clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(1, 1))
    frame = clahe.apply(l)
    # cv2.line(frame_og, (300, 513), (1900, 513), (0, 255, 0), 2)
    # cv2.line(frame_og, (300, 482), (1900, 482), (0, 0, 255), 2)
    cv2.line(frame_og, (l1_x1, l1_y1), (l1_x2, l1_y2), (0, 255, 0), 2)
    cv2.line(frame_og, (l2_x1, l2_y1), (l2_x2, l2_y2), (0, 0, 255), 2)

    if ret == True:
        foregroundMask = bgSubtractor.apply(frame)
        foregroundMask = cv2.morphologyEx(foregroundMask, cv2.MORPH_OPEN, kernel)
        foregroundMask = cv2.erode(foregroundMask, kernel, iterations=3)
        foregroundMask = cv2.morphologyEx(foregroundMask, cv2.MORPH_CLOSE, kernel,iterations=6)
        foregroundMask = cv2.dilate(foregroundMask, kernel_di, iterations=7)
        foregroundMask = cv2.medianBlur(foregroundMask,5)
        thresh = cv2.threshold(foregroundMask, 25, 255, cv2.THRESH_BINARY)[1]
        thresh1 = np.bitwise_and(thresh, thresh_MASK_1)
        thresh2 = np.bitwise_and(thresh, thresh_MASK_2)
        contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        try:
            hierarchy = hierarchy[0]
        except:
            hierarchy = []

        for contour, hier in zip(contours, hierarchy):
            areas = [cv2.contourArea(c) for c in contours]
            cv2.drawContours(frame_og, contours=contours,contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
            max_index = np.argmax(areas)
            # start_time= 0

            cnt = contours[max_index]
            (x, y, w, h) = cv2.boundingRect(cnt)
            area_=(w*h)
            area_s.append(area_) 
            
            cx = int((w / 2) + x)
            cy = int((h / 2) + y)
            

            if w > 10 and h >10 and h < 80:
                
                cv2.rectangle(frame_og, (x - 10, y - 10), (x + w, y + h), (0, 255, 0), 2)
                #cv2.line(frame_og,(x,y+h),(x+w,y+h+10),(255,0,0),3)
                cv2.circle(frame_og, (cx, cy), 10, (0, 0, 255), -1)
    
        if cy > l1_y1 and w > 70 and h > 100:       #l1_y1 is first Y from top.
            
            if flag is True and cy < (l1_y1+mid_pt):
                start_time = datetime.datetime.now() 
                flag = False
            if flag is False and cy > (l1_y1 + mid_pt) and cy < (l2_y2) :
                later = datetime.datetime.now()
                seconds_Lane1 = (later - start_time).total_seconds()
                #seconds = (later - start_time).total_seconds()
                # numb_frames1= int(30/seconds_Lane1)
                # print("num of frames object appeared between 2 lines for lane 1 BEFORE IF",numb_frames1)
                # print("TOTAL TIME",seconds_Lane1)
                # speed_Lane1 = (distance/numb_frames1)*30*3.6
                # Angle = 60
                # Angle = math.radians(Angle)
                # Angle = math.cos(Angle)
                # speed_Lane1 = speed_Lane1*Angle
                # print("speed_Lane1",speed_Lane1)
               
                if seconds_Lane1 <= 0.2:
                    print("diff 0")
                else:
                    if flag is False:
                        numb_frames1= int(30/seconds_Lane1)
                        print("num of frames object appeared between 2 lines for lane 1 AFTER IF",numb_frames1)
                        print("TOTAL TIME",seconds_Lane1)
                        speed_Lane1 = (distance/numb_frames1)*30*3.6
                        speed = ((distance) / (36.6 *(seconds_Lane1))) * 3600 * 90
                        Angle = 60
                        Angle = math.radians(Angle)
                        Angle = math.cos(Angle)
                        speed_Lane1 = speed_Lane1/Angle
                        print("speed_Lane1",speed_Lane1)
                        print("SPEED",speed)
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        cv2.putText(frame_og, str(int(speed_Lane1)), (x, y), font, 2, (255, 255, 255), 8, cv2.LINE_AA)
                        cv2.putText(frame, str(int(speed_Lane1)), (x, y), font, 2, (255, 255, 255), 8, cv2.LINE_AA)
                       
                        if int(speed_Lane1) > speed_limit and w > 70 and h > 100 :
                            roi = frame[y-50:y + h, x:x + w]
                            cv2.imshow("Lane_1", roi)
                            lane_1_1.append(roi)

                            #send(roi)                                                  #Send images to SERVER ++++++++++++++++API
                            # write_name = 'corners_found' + str(cnt1) + '.jpg'
                            # cv2.imwrite(write_name, roi)
                            # cv2.imwrite(os.path.join(path, 'carimage_l2_' + str(cnt1)) + '.jpg', roi)
                            cnt += 1
                    flag = True
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(frame, str(int(speed_Lane1)), (x, y), font, 2, (255, 255, 255), 8, cv2.LINE_AA)


                  

        contours1, hierarchy1= cv2.findContours(thresh2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        try:
            hierarchy1 = hierarchy1[0]
        except:
            hierarchy1 = []
        
        for contour1, hier1 in zip(contours1, hierarchy1):
            areas1 = [cv2.contourArea(c) for c in contours1]
            max_index1 = np.argmax(areas1)
            cnt1 = contours1[max_index1]
            (x1, y1, w1, h1) = cv2.boundingRect(cnt1)
            cx1 = int((w1 / 2) + x1)
            cy1 = int((h1 / 2) + y1)
            
            if w1 > 10 and h1 > 10:
                cv2.rectangle(frame_og, (x1 - 10, y1 - 10), (x1 + w1, y1 + h1), (255, 255, 0), 2)
                cv2.circle(frame_og, (cx1, cy1), 5, (0, 255, 0), -1)
        
        if cy1 > l1_y1 and w1 > 70 and h1 > 100:
            

            if flag1 is True and cy1 < (l1_y1+mid_pt):
                print("CY++++++++++++++++++++++",cy)
                start_time_Lane2 = datetime.datetime.now()  #START TIME LANE 2
                flag1 = False
            if cy1 > (l1_y1+mid_pt) and cy1 < (l2_y2) :
                later_Lane2 = datetime.datetime.now()       # END TIME LANE 2
                frame_cntr2 += 1     
                seconds_Lane2 = (later_Lane2 - start_time_Lane2).total_seconds()

                if seconds_Lane2 <= 0.2:
                   print("diff1 0")
                else:
                    print("seconds_Lane2: " + str(seconds_Lane2))
                    if flag1 is False:
                        time_per_frame = (1/30)
                        numb_frames_Lane2= (seconds_Lane2/time_per_frame)
                        print("num of frames object appeared between 2 lines for lane 2",numb_frames_Lane2)
                        speed_Lane2 = (distance/numb_frames_Lane2)*30*3.6
               
                        print("SPEED NEW",speed_Lane2)
                        print("DIST",distance)
                        print("Total Seconds",seconds_Lane2)
                       
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        cv2.putText(frame_og, str(int(speed_Lane2)), (x1, y1), font, 2, (255, 255, 255), 3, cv2.LINE_AA)
                        cv2.putText(frame, str(int(speed_Lane2)), (x1, y1), font, 2, (255, 255, 255), 8, cv2.LINE_AA)
                          # if not os.path.exists(path):
                          #     os.makedirs(path)
                        if int(speed_Lane2) > speed_limit and cy1 <= (l2_y2+120) and w1 > 70 and h1 > 100:
                           roi = frame[y1-50:y1 + h1, x1:x1 + w1]
                           cv2.imshow("Lane_2", roi)
                           lane_1_2.append(roi)
                            #send(roi)
                            #cv2.imwrite(os.path.join('Offenders/', 'carimage_l2_' + str(cnt1)) + '.jpg', roi)
                           cnt1 += 1
                        flag1 = True
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        cv2.putText(frame_og, str(int(speed_Lane2)), (x1, y1), font, 2, (255, 255, 255), 3, cv2.LINE_AA)
        
        cv2.imshow('backgroundsubtraction', frame_og)
        stop = timeit.default_timer()
        time = stop-start
        print('One_frame = ',time)
        
        k = cv2.waitKey(1) & 0xff
        if k == ord('q'):
            break
    else:
        break

#send images to server for processing and ANPR (via robust_pipeline.py)
    #     u,v =0,0
    #     for la in lane_1_1:
    # #cv2.imwrite('Offenders/lane1/'+'Lane'+str(v)+'.jpeg',la)
    #        send(la)
    #        v+=1
    #     for li in lane_1_2:
    # #cv2.imwrite('Offenders/lane2/'+str(v)+'.jpeg',li)
    #        send(li)
    #        u+=1





  
                
cap.release()
cv2.destroyAllWindows()

