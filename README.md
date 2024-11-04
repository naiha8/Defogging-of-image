# Defogging-of-image
Remove the fog from images and make them clear and sharp
import cv2         #using cv2
import numpy as np
from google.colab.patches import cv2_imshow #for coding in colab
import matplotlib.pyplot as plt         #use for ploting

img1 = cv2.imread('/content/flickr1.bmp',3)  #taking image 1 path(read image)
cv2_imshow(img1) #(desplaying image)

img2 = cv2.imread('/content/flickr2.bmp',3)  #taking image 2 path(read image)
cv2_imshow(img2) 

img3 = cv2.imread('/content/flickr3.bmp',3)  #taking image 3 path(read image)
cv2_imshow(img3) 

# size of image 1
img1.shape
# size of image 2
img2.shape
# size of image 3
img3.shape

#histogram of orignal images
#for image 1
hist,bins = np.histogram(img1.flatten(),255,[0,255])

plt.hist(img1.flatten(),255,[0,255], color = 'g')
plt.xlim([0,255])
plt.legend(('histogram'), loc = 'upper left')
plt.show()

#for image 2
hist,bins = np.histogram(img2.flatten(),255,[0,255])

plt.hist(img2.flatten(),255,[0,255], color = 'b')
plt.xlim([0,255])
plt.legend(('histogram'), loc = 'upper left')
plt.show()

#for Image 3
hist,bins = np.histogram(img3.flatten(),255,[0,255])

plt.hist(img3.flatten(),255,[0,255], color = 'm')
plt.xlim([0,255])
plt.legend(('histogram'), loc = 'upper left')
plt.show()
#images in HSV
hsv1 = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV) #in img1 hue saturation value
cv2_imshow(hsv1)
cv2.waitKey(0)

hsv2 = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV) #in img2 hue saturation value
cv2_imshow(hsv2)
cv2.waitKey(0)

hsv3 = cv2.cvtColor(img3, cv2.COLOR_BGR2HSV) #in img3 hue saturation value
cv2_imshow(hsv3)
cv2.waitKey(0)

#histogram equilizer
#hsv1
R,G,B=cv2.split(hsv1)
output_R=cv2.equalizeHist(R)
output_G=cv2.equalizeHist(G)
output_B=cv2.equalizeHist(B)
hist_eq11=cv2.merge((output_R,output_G,output_B))
cv2_imshow(hist_eq11)
plt.hist(hist_eq11.ravel(), 256, [0, 256])
plt.show()

im_rgb = cv2.cvtColor(hist_eq11, cv2.COLOR_HSV2BGR)
cv2_imshow(im_rgb)
cv2.waitkey(0)

#from skimage import exposure
#image = img

#gamma_transformation
#gamma_corrected = exposure.adjust_gamma(image,2)

#log
#log_corrected_n = exposure.adjust_log(image, gain=1, inv=False)

#eq_im_g = exposure.equalize_hist(img)
#eq_im_l = exposure.equalize_adapthist(img)

#gamma_corrected_n = exposure.adjust_gamma(eq_im_l,1.2)
#con_img = exposure.rescale_intensity(eq_im_l)
#log_corrected_n = exposure.adjust_log(eq_im_l,gain=0.8, inv=False)

#outcome is darker for gamma >1

#cv2_imshow(img)
#cv2_imshow(gamma_corrected) #gamma before eq
#cv2_imshow(log_corrected_n) #log before eq
#cv2_imshow(eq_im_g) #eq_imag_hist
#cv2_imshow(eq_im_l) #eq_imag_adaphist
#cv2_imshow(gamma_corrected_n) #gamma after eq
#cv2_imshow(con_img) #con imag after eq
#cv2_imshow(log_corrected_n) #log imag after eq

#cv2.waitKey(0)
#cv2.destroyAllWindows

# Trying 3 gamma values on img1: 
for gamma in [1.5,1.7,1.9,]: 
      
      #gamma>1the intensity of pixels decreases i.e. the image becomes darker.

    # Apply gamma correction. 
    gamma_corrected1 = np.array(255*(img1 / 255) ** gamma, dtype = 'uint8') 
  
    # Save edited images. 
    cv2_imshow(gamma_corrected1) 
    
# Trying 3 gamma values on img2: 
for gamma in [1.6,1.7,1.8]: 
      
    # Apply gamma correction. 
    gamma_corrected2 = np.array(255*(img2 / 255) ** gamma, dtype = 'uint8') 
  
    # Save edited images. 
    cv2_imshow(gamma_corrected2)

 # Trying 3 gamma values on img3: 
for gamma in [1.5,1.6,1.7]: 
      
    # Apply gamma correction. 
    gamma_corrected3 = np.array(255*(img3 / 255) ** gamma, dtype = 'uint8') 
  
    # Save edited images. 
    cv2_imshow(gamma_corrected3)
    #histogram of last gamma value image
#img1
hist,bins = np.histogram(gamma_corrected1.flatten(),255,[0,255])
plt.hist(gamma_corrected1.flatten(),255,[0,255], color = 'g')
plt.xlim([0,255])
plt.legend(('histogram'), loc = 'upper left')
plt.show()

#img2
hist,bins = np.histogram(gamma_corrected2.flatten(),255,[0,255])
plt.hist(gamma_corrected2.flatten(),255,[0,255], color = 'b')
plt.xlim([0,255])
plt.legend(('histogram'), loc = 'upper left')
plt.show()

#img3
hist,bins = np.histogram(gamma_corrected3.flatten(),255,[0,255])
plt.hist(gamma_corrected3.flatten(),255,[0,255], color = 'm')
plt.xlim([0,255])
plt.legend(('histogram'), loc = 'upper left')
plt.show()
#shading correction on img 1
#-------------Converting image to LAB Color model----------------------------------- 
lab1= cv2.cvtColor(img1, cv2.COLOR_BGR2LAB)   #lab is some sort of colours
cv2_imshow(lab1)

#-----Splitting the LAB image to different channels-------------------------
l1, a1, b1 = cv2.split(lab1)
cv2_imshow( l1) #l_channel    #luminance channel    
cv2_imshow( a1) #a_channel
cv2_imshow( b1) #b_channel

#transformation on l1
#log transformation
c = 255/(np.log(1 + np.max(l1))) 
log_transformedl = c * np.log(1 + l1) #the dark pixels in an image are expanded as compare to the higher pixel values.
  
# Specify the data type. 
log_transformedll = np.array(log_transformedl, dtype = np.uint8) 
  
# Save the output. 
cv2_imshow(log_transformedll)
cv2.waitKey(0)

#-----Applying CLAHE to L-channel-------------------------------------------
clahe1 = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
                                               #clahe is limited adaptive histogram equilizer (adaptive histogram equilizer use for improve contrast)
cl = clahe1.apply(log_transformedll)
cv2_imshow( cl)


#-----Merge the CLAHE enhanced L-channel with the a and b channel-----------
limg1 = cv2.merge((cl,a1,b1))
cv2_imshow( limg1)

#-----Converting image from LAB Color model to RGB model--------------------
final1 = cv2.cvtColor(limg1, cv2.COLOR_LAB2BGR)
cv2_imshow( final1)


# Trying 3 gamma values on shedding correction img1: 
for gamma in [2.8,3,3.3]: 
      
    # Apply gamma correction. 
    gamma_corrected111 = np.array(255*(final1 / 255) ** gamma, dtype = 'uint8') 
  
    # Save edited images. 
    cv2_imshow(gamma_corrected111)

    #biletral filter
bil = cv2.bilateralFilter(gamma_corrected111,9,75,50)
cv2_imshow(bil)
#is highly effective at noise removal while preserving edges.

![download](https://github.com/user-attachments/assets/f86329a3-c5d4-4b3c-a4ff-a8716bcfd1a7)
![download (1)](https://github.com/user-attachments/assets/b138cb06-0c23-4cb3-a8fe-736596823ced)
![download (2)](https://github.com/user-attachments/assets/5597fdf2-9450-475f-b860-fa2d3a6b8026)
![download (3)](https://github.com/user-attachments/assets/f6256b05-6ecd-4f18-a00c-941b4834111a)
![download (4)](https://github.com/user-attachments/assets/6362f958-837e-467b-9f82-e1e25a93e6bd)
![download (5)](https://github.com/user-attachments/assets/426c4c00-2c5b-48f7-aafa-596b7f4eae74)
![download (6)](https://github.com/user-attachments/assets/54edd434-f497-4b6d-a9fa-c1223efe53fa)
