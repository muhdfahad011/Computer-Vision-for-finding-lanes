import cv2
import numpy as np

def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1 * (3/5))
    #both of our lines will start from 700 and will go 3/5 of y HORIZONTALLY
    x1 = int((y1 - intercept) / slope) #just rearrange the y=mx+c in terms of x
    x2 = int((y2 - intercept) / slope) 

    return np.array([x1, y1, x2, y2])




def average_slope_intercept(image, lines):
    left_fit = [] #contains the coordinates of the averaged lines on the left 
    right_fit = [] #contains the coordinates of the averaged lines on the right
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4) #converts the coordinates of lines as [x1,y1,x2,y2]
        parameters = np.polyfit((x1,x2), (y1, y2), 1) #np.polyfit() fits the first degree polynomial to x and y points abd returns the coefficients which describe m and c 
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0: # if the slope in negative we'll apend it to the left  else on the right
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))

    left_fit_average = np.average(left_fit, axis = 0) # returns the average m and c on the left   
    right_fit_average = np.average(right_fit, axis = 0) ## returns the average m and c on the right

    left_line = make_coordinates(image, left_fit_average)
    right_line = make_coordinates(image, right_fit_average)
    return np.array([left_line, right_line])




def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) # "cvtColor" converts the image from one color space to another
    blur = cv2.GaussianBlur(gray, (5,5), 0) # (5,5) is the dimention of the kernel matrix used for smoothening the image
    canny = cv2.Canny(blur, 50, 150)
    return canny


def display_lines(image, lines):# image is the image on which the lines will be imposed on 
    line_image = np.zeros_like(image) #line image will have the same pixels as the image although line_image  has the 0 intensity pixels   
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 =  line.reshape(4)
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0,0), 10)# this specifies on what coordinates on the line_image we want the lines to appear
        return line_image       
            



def region_of_interest(image):
    height = image.shape[0] #this correspons to the the row of the image, we're only traversing only along the height that is y-axis which corresponds to the height of image
    polygons = np.array([
        [(200, height), (1100, height), (500, 250)] # this corresponds to an array of a singlr polygon
         ])
    mask = np.zeros_like(image) #mask is where our triangle will be on 
    # np.zeros_like(image) creates 0 array of same  dimentions as  the dimentions of image
    cv2.fillPoly(mask, polygons, 255) # fills the mask with the triangle, triangle having the intensit of 255
    masked_image = cv2.bitwise_and(mask, image)
    return masked_image




cap = cv2.VideoCapture('test2.mp4')
while(cap.isOpened()):
    _, frame = cap.read()
    canny_image = canny(frame)
    cropped_image = region_of_interest(canny_image)  
    lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100,  minLineLength=40, maxLineGap=5)
    averaged_lines = average_slope_intercept(frame, lines)
    line_image = display_lines(frame, averaged_lines)
    combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    cv2.imshow('result', combo_image)
    if cv2.waitKey(1) & 0xFF == ord(' '):
        break
cap.release()
cv2.destroyAllWindows()

