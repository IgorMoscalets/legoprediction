import cv2
import numpy as np

lowerBound=np.array([0,150,40])
upperBound=np.array([179,255,255])

cam= cv2.VideoCapture(0)
kernelOpen=np.ones((5,5))
kernelClose=np.ones((20,20))

while True:
	ret, img=cam.read()
	#img=cv2.resize(img,(340,220))

	#convert BGR to HSV
	imgHSV= cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
	# create the Mask
	mask=cv2.inRange(imgHSV,lowerBound,upperBound)
	#morphology
	maskOpen=cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernelOpen)
	maskClose=cv2.morphologyEx(maskOpen,cv2.MORPH_CLOSE,kernelClose)

	maskFinal=maskClose
	b,conts,h=cv2.findContours(maskFinal.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

	cv2.drawContours(img,conts,-1,(255,0,0),3) # for blue contour drawing
	for i in range(len(conts)):
		x,y,w,h=cv2.boundingRect(conts[i])
		y = int(y-w/1.5)
		x = x-w/2
		#print x,y,w,h
		cv2.rectangle(img,(x,y),(x+w*2,y+w*2),(0,0,255), 2)
		if y+(w*2) > 200 and x+(w*2) > 200 and x > 1 and y > 1:
			crop_img = img[y:y+(w*2),x:x+(w*2)]
			cv2.imshow("cropped", crop_img)
			
	cv2.imshow("cam",img)
	cv2.resizeWindow("cam",1000,1000)
	cv2.waitKey(10)