import os

import numpy as np
import cv2
from PIL import Image

class Lines:
    def __init__(self, image_path):
        self.image_path = image_path
        self.img = Image.open(image_path)

    def rgb_to_grayscale_luminosity(self,image):
        """
        Convert an RGB image to grayscale using luminosity method.

        Parameters:
            image: Input RGB image (numpy array or equivalent).

        Returns:
            grayscale_image: Grayscale image (numpy array or equivalent).
        """
        # Get the dimensions of the input image
        img_array = np.array(image)
        height, width, channels = img_array.shape

        # Create an empty array for the grayscale image
        grayscale_image = np.zeros((height, width), dtype=np.uint8)

        # Iterate over each pixel in the image
        for y in range(height):
            for x in range(width):
                # Extract the RGB values of the pixel
                r, g, b = img_array[y, x]

                # Calculate luminosity
                # luminosity = 0.21 * r + 0.72 * g + 0.07 * b
                luminosity = 0.299 * r + 0.587 * g + 0.114 * b
                # Set the grayscale pixel value
                grayscale_image[y, x] = int(luminosity)

        grayscale_image = Image.fromarray(grayscale_image)
        return grayscale_image


    def adaptive_thresholdMean(self,image, block_size, c):
        """
        Apply Adaptive Mean Thresholding to the input grayscale image.

        Parameters:
            image: Input grayscale image (PIL Image object).
            block_size: Size of the local neighborhood for threshold calculation.
            c: Constant subtracted from the mean to obtain the threshold value.

        Returns:
            thresholded_image: Thresholded image (PIL Image object).
        """
        # Convert PIL Image to numpy array
        grayscale_img = self.rgb_to_grayscale_luminosity(image)
        img_array = np.array(grayscale_img)

        # Get image dimensions
        height, width = img_array.shape

        # Create an empty array for the output
        thresholded_image = np.zeros((height, width))

        # Apply adaptive thresholding
        for i in range(height):
            for j in range(width):
                # Calculate block boundaries
                x_min = max(0, i - block_size // 2)
                y_min = max(0, j - block_size // 2)
                x_max = min(height - 1, i + block_size // 2)
                y_max = min(width - 1, j + block_size // 2)
                
                # Extract block from image
                block = img_array[x_min:x_max+1, y_min:y_max+1]
                
                # Calculate threshold
                thresh = np.mean(block) - c
                
                # Apply thresholding
                if img_array[i, j] >= thresh:
                    thresholded_image[i, j] = 255
                else:
                    thresholded_image[i, j] = 0

        # Convert thresholded numpy array back to PIL Image
        thresholded_image = Image.fromarray(thresholded_image)

        return thresholded_image

    def negate(self,image):
        '''
        White background, black text -> black background, white text
        '''
        img_arr = np.array(image)
        # img_arr[img_arr == 0] = np.uint8(255)
        height, width = img_arr.shape
        for y in range(height):
            for x in range(width):
                img_arr[y,x] = (img_arr[y,x]==0).astype(np.uint8)*255
        return Image.fromarray(img_arr)
def get_lines(image_path):
    lines = Lines(image_path)
    thresh_img = lines.adaptive_thresholdMean(lines.img,50,30)
    thresh_img = lines.negate(thresh_img)
    # Assuming thresh_img is a PIL.Image.Image object
    thresh_img = np.asarray(thresh_img)

    # dilation
    kernel = np.ones((5,100),np.uint8)
    dilated_img = cv2.dilate(thresh_img,kernel,iterations = 1)

    # Convert the dilated image to a suitable format for contour detection
    dilated_img_uint8 = dilated_img.astype(np.uint8)

    # Find contours
    contours, hierarchy = cv2.findContours(dilated_img_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours by y-coordinate
    sorted_contours_lines = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[1])

    # img2 = np.array(img.copy())
    line_boxes = []
    for ctr in sorted_contours_lines:
        
        if cv2.contourArea(ctr)<1000:
            continue

        x,y,w,h = cv2.boundingRect(ctr)
        line_boxes.append((x,y,w,h))
    #     cv2.rectangle(img2, (x,y), (x+w,y+h),(40, 100, 250) ,1)
    return line_boxes

# image_path = 'mal2.jpg'
# img = Image.open(image_path)

# thresh_img = adaptive_thresholdMean(img, 50, 30)
# thresh_img = negate(thresh_img)

# # Assuming thresh_img is a PIL.Image.Image object
# thresh_img = np.asarray(thresh_img)

# # dilation
# kernel = np.ones((5,100),np.uint8)
# dilated_img = cv2.dilate(thresh_img,kernel,iterations = 1)


# # Convert the dilated image to a suitable format for contour detection
# dilated_img_uint8 = dilated_img.astype(np.uint8)

# # Find contours
# contours, hierarchy = cv2.findContours(dilated_img_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# # Sort contours by y-coordinate
# sorted_contours_lines = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[1])

# img2 = np.array(img.copy())

# for ctr in sorted_contours_lines:
    
#     if cv2.contourArea(ctr)<1000:
#         continue

#     x,y,w,h = cv2.boundingRect(ctr)
#     cv2.rectangle(img2, (x,y), (x+w,y+h),(40, 100, 250) ,1)
