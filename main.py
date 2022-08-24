from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image 
import cv2

raw_image = cv2.imread('/home/farooq/kamil/Qasim_Whitefly/13.1.1.png')
#cropped_raw_image = cv2.imread('/home/farooq/kamil/Qasim_Whitefly/cropped_13.1.1.png')

#raw_image = cv2.imread('/home/farooq/kamil/Qasim_Whitefly/51.1.1_slightcrop.png')

#raw_image = cv2.imread('/home/farooq/kamil/Qasim_Whitefly/56.3.1_slightcrop.png')

#raw_image = cv2.imread('/home/farooq/kamil/Qasim_Whitefly/57.3.1_slightcrop.png')

rgb_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)

cv2.imshow('raw', raw_image)
cv2.waitKey(0)


cv2.imshow('rgb', rgb_image)
cv2.waitKey(0)

#cv2.destroyAllWindows()



hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
h, s, v = cv2.split(hsv_image)

cv2.imshow('hue', h) 
cv2.waitKey(0)


#ycc_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2YCrCb)
#y, cr, cb = cv2.split(ycc_image)

#cv2.imshow('y', y) 
#cv2.waitKey(0)

#cv2.imshow('cr', cr) 
#cv2.waitKey(0)

#cv2.imshow('cb', cb) 
#cv2.waitKey(0)


#increasing brightness and contrast
new_image = np.zeros(h.shape, h.dtype)
for beta in range(25,26): #25-13.1
    for alpha in range(4,5): #4-13.1
        for y in range(h.shape[0]):
            for x in range(h.shape[1]):
                #for c in range(h.shape[2]):
                new_image[y,x] = np.clip(alpha*h[y,x] + beta, 0, 255)
        name = 'New_Image_alpha'+str(alpha)+'_beta_'+str(beta)
        cv2.imshow(name, new_image)
        cv2.waitKey(0)

######### TRYING
new_image = np.zeros(cb.shape, cb.dtype)
for beta in range(1,2): #25
    for alpha in range(2,3): #4
        for y in range(cb.shape[0]):
            for x in range(cb.shape[1]):
                #for c in range(h.shape[2]):
                new_image[y,x] = np.clip(alpha*cb[y,x] + beta, 0, 255)
        name = 'CB_New_Image_alpha'+str(alpha)+'_beta_'+str(beta)
        cv2.imshow(name, new_image)
        cv2.waitKey(0)

##################


cv2.imshow(name, new_image)
cv2.waitKey(0)


thresh_binary = 190#138
ret, binary = cv2.threshold(new_image, thresh_binary, 255, cv2.THRESH_BINARY) #190-13.1, 160-56.3.1
cv2.imshow('binary_' + str(thresh_binary), binary)
cv2.waitKey(0)


# To detect object contours, we want a black background and a white 
# foreground, so we invert the image (i.e. 255 - pixel value)
with_contour_image = raw_image.copy()
#newest_image = raw_image.copy()
inverted_binary = ~binary
cv2.imshow('Inverted binary image', inverted_binary)
cv2.waitKey(0) # Wait for keypress to continue


# Find the contours on the inverted binary image, and store them in a list
# Contours are drawn around white blobs.
# hierarchy variable contains info on the relationship between the contours
contours, hierarchy = cv2.findContours(inverted_binary,
  cv2.RETR_TREE,
  cv2.CHAIN_APPROX_SIMPLE)#cv2.RETR_EXTERNAL)#cv2.CHAIN_APPROX_SIMPLE)-13.1.1
     
# Draw the contours (in red) on the original image and display the result
# Input color code is in BGR (blue, green, red) format
# -1 means to draw all contours
with_contours = cv2.drawContours(with_contour_image, contours, -1,(255,0,255),3)
cv2.imshow('Detected contours_' + str(thresh_binary), with_contours)
cv2.waitKey(0)

#cv2.imshow('CB_D' + str(thresh_binary)etected contours', with_contours)
#cv2.waitKey(0)


cv2.destroyAllWindows()
 
# Show the total number of contours that were detected
print('Total number of contours detected: ' + str(len(contours)))


cv2.destroyAllWindows()


################ SAVING ##############

cv2.imwrite('raw_image_13.1.1.png', raw_image)
cv2.imwrite('rgb_image_13.1.1.png', rgb_image)
cv2.imwrite('hue_image_13.1.1.png', h)
cv2.imwrite('bright_contrast_hue_image_13.1.1.png', new_image)
cv2.imwrite('binary_image_13.1.1.png', binary)
cv2.imwrite('inverted_binary_image_13.1.1.png', inverted_binary)
cv2.imwrite('identified_whitefly_eggs_image_13.1.1.png', with_contours)

######### Separate approach: Trying to ignore contours that don't correspond to whitefly eggs through area thresholding ###########
area_list = []
for i in range(len(contours)):
    area_list.append(cv2.contourArea(contours[i])) 

area_list.sort()
idx = area_list.count(0)

area_list[80:90]
#idx = area_list.index(50)

area_list[351]

idx = 0
mean = np.mean(area_list[idx:])


temp = list(np.float_(area_list[350:]))
scaled_temp = my_new_list = [i * (1/mean) for i in temp]

sum(scaled_temp)

