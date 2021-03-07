#Based on Zed code - Person Fall detection using raspberry pi camera and opencv lib. Link: https://www.youtube.com/watch?v=eXMYZedp0Uo
import datetime
from cv2 import cv2
import time

#Slowdown variables
slowDown = 0.1
allSlowDown = 0

#Video Variables
video_name = "Fall 14.mp4"
vidName = cv2.VideoCapture(0)

#Image processing variables
fgbg = cv2.createBackgroundSubtractorKNN()#Edit

#Fall variables
fall_frames = 0
fall_count = 0          #Times the program detected a "fall"
fall_check = True     #Check if fall has been entered to system   

status = "Empty"
run = True

#File Check
try:
    file = open('FallTimmings.txt','r')
    file.close()
except:
    print("File not found!")
    file = open('FallTimmings.txt','w')
    file.close()

while(run):
    ret, frames = vidName.read()
    
    try:
        status = "Empty"

        #Convert each frames to gray scale and subtract the background
        gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
        gray = cv2.blur(gray,(5,5))

        #Edges
        edges = cv2.Canny(frames,110,110)
        
        #fgmask processing
        fgmask = fgbg.apply(gray)
        fgmask = cv2.erode(fgmask,(3,3))
        fgmask = cv2.dilate(fgmask,(5,5))

        #Find contours
        contours, _ = cv2.findContours(fgmask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # List to hold all areas
            areas = []

            for contour in contours:
                ar = cv2.contourArea(contour)
                areas.append(ar)
            
            max_area = max(areas, default = 1)
            max_area_index = areas.index(max_area)

            #Contours
            cnt = contours[max_area_index]
            x, y, w, h = cv2.boundingRect(cnt)
            

            #Weed out specks and shakes
            if max_area > 1000:
                if  w > h:
                    fall_frames += 1

                    if fall_frames > 8:
                        if not fall_check:
                            #Save The Falls
                            fall_count += 1
                            file = open('FallTimmings.txt','a')
                            file.write("{}\n".format(datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p")))
                            file.close()
                            file = open('Fall.txt','w')
                            file.write("Fall: {}\n".format(fall_count))
                            file.close()
                            fall_check = True
                        status = "Fell"
                        cv2.rectangle(frames,(x,y),(x+w,y+h),(0,0,255),2)
                        time.sleep(slowDown)
                    else:
                        #Ensure there is an rectangle on the contour
                        cv2.rectangle(frames,(x,y),(x+w,y+h),(0,255,0),2)

                if h >= w:
                    status = "Occupied"
                    fall_frames = 0
                    fall_check = False
                    cv2.rectangle(frames,(x,y),(x+w,y+h),(0,255,0),2)

        #Text written no video
        cv2.putText(frames, "Room Status: {0}".format(status), (10, 20),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(frames, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),(10, frames.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        #Show each frame
        cv2.imshow('video', frames)
        cv2.imshow('FGmask', fgmask)
        cv2.imshow('Edges', edges)
        time.sleep(allSlowDown)
        
        #Stop code using "q"
        if cv2.waitKey(1) & 0xFF == ord('q'):
            run = False
            break
    except Exception as e:
        break

cv2.destroyAllWindows()