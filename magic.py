import cv2
import time
import numpy as np

#video codecs compress video data&encode into format that can later be decoded & played back/edited
fourcc = cv2.VideoWriter_fourcc(*"XVID")
outputfile = cv2.VideoWriter("output.avi" , fourcc , 20.0 , (640 , 680))
cap = cv2.VideoCapture(0)
time.sleep(2)
bg = 0

#Capturing background for 60 frames
for i in range(60):
    ret,bg = cap.read()
bg = np.flip(bg,axis=1)

while(cap.isOpened()):
    ret,img = cap.read()
    if not ret :
        break
    img = np.flip(img,axis=1)
    hsv = cv2.cvtColor(img ,cv2.COLOR_BGR2HSV)
    lower_red = np.array([0,120,50])
    upper_red = np.array([10,255,255])
    mask1 = cv2.inRange(hsv,lower_red , upper_red)

    lower_red = np.array([170,120,70])
    upper_red = np.array([180,255,255])
    mask2 = cv2.inRange(hsv,lower_red , upper_red)
    mask1 = mask1 + mask2

    mask1 = cv2.morphologyEx(mask1 , cv2.MORPH_OPEN , np.ones( (3,3) , np.uint8 )) 
    mask2 = cv2.morphologyEx(mask1 , cv2.MORPH_DILATE , np.ones( (3,3) , np.uint8 ))

    #Selecting only the part that does not have mask one and saving in mask 2 - segment out red color from the image
    mask2 = cv2.bitwise_not(mask1) 

    #Keeping only the part of the images without the red color 
    res1 = cv2.bitwise_and(img,img,mask = mask2)
    
    #Keeping only the part of the images with the red color
    res2 = cv2.bitwise_and(bg,bg , mask = mask1)
    
    #function helps in transition of img to another. In order to blend this image, we can add weights & define the transparency and translucency of the images.
    final_output = cv2.addWeighted(res1 , 1 , res2 , 1 , 0)
    outputfile.write(final_output)
    cv2.imshow("magic " , final_output)
    cv2.waitKey(1)

cap.release()
outputfile.release()
cv2.destroyAllWindows()


      
   