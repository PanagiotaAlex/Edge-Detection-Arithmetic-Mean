# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 18:36:30 2021

@author: Panay
"""
#Import Libraries
import numpy as np
import cv2 #opencv
import skvideo
import skvideo.io
import skvideo.datasets

# Add zero padding         
def pad_image(A, size):
    pad = (size-1)/2 #stride is 1
    pad = int(pad)
    size = pad
    Apad = np.pad(A, ((pad,pad), (pad,pad),(pad,pad)), 'constant', constant_values = (0,0))
    return(Apad)

#Arithmetic Mean
def mean(A,param):
    
    if param == 'same':
        pad = 0
        A = A
        
    elif param == 'pad':
        pad = 3
        A = pad_image(A,pad)
    #Create an empty matrix    
    M = np.zeros_like(A) 
    
    
    for z in range(A.shape[0]):     
        if z>A.shape[0]-3:
            break 
        for y in range(A.shape[1]):
            if y>A.shape[1]-3:
                break  
            for x in range(A.shape[2]):
                if x>A.shape[2]-3:
                    break
                       
                M[z, y, x] = (np.sum(A[z-1:z+1,y-1:y+1,x-1:x+1]))*(1/27)

                            
 
    return M

#Edge Detection    
def mySobel(A):
    
    filter_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    filter_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
     

    #Create an empty matrix    
    C = np.zeros_like(A) 
    
    for z in range(A.shape[0]):     

        for y in range(A.shape[1]):#-2
      
            for x in range(A.shape[2]):#-2
   
                try:    
                     
                        vert_start = y
                        vert_end = y + 3
                        horiz_start = x 
                        horiz_end = x + 3
                        A_x =  np.sum(np.multiply(filter_x,A[z,  vert_start:vert_end, horiz_start:horiz_end]))
                        A_y =  np.sum(np.multiply(filter_y,A[z,  vert_start:vert_end, horiz_start:horiz_end]))
                        C[z, y+1, x+1] = np.sqrt(A_x**2 + A_y**2)
                except:
                    break
 
 
    return C
    
def display(video):

    cap = cv2.VideoCapture(video)
    
    # Check if camera opened successfully
    if (cap.isOpened()== False): 
      print("Error opening video stream or file")
    
    # Read until video is completed
    while(cap.isOpened()):
      # Capture frame-by-frame
      ret, frame = cap.read()
      if ret == True:
    
        # Display the resulting frame
        cv2.imshow('Frame',frame)
    
        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
          break
    
      # Break the loop
      else: 
        break
    
    # When everything done, release the video capture object
    cap.release()
    
    # Closes all the frames
    cv2.destroyAllWindows()
    

def main():
	#Read the video using skvideo
    video = skvideo.io.vread('car.mp4') 
    print(video.shape)
    #Convert to grayscale
    grayframe = np.zeros_like(video[...,0])
    for i in range(video.shape[0]):
        grayframe[i]=cv2.cvtColor(video[i],cv2.COLOR_RGB2GRAY)            
    print(grayframe.shape)
    
    
    ar_mean = mean(grayframe,'same')
    #Save the video    
    skvideo.io.vwrite("output_1.mp4", ar_mean)
    display('output_1.mp4')

    sobel = mySobel(grayframe) 
    #Save the video 
    skvideo.io.vwrite("output_2.mp4", sobel)
    display('output_2.mp4')


    
if __name__ == "__main__":
    main()