# **Distance Transform:**
The distance transform is calculated based on euclidean distance assuming 8-pixels neighbors.
The algorithm performs a forward scan calculating euclidean distances then a backward scan for repairing distances to be the maximum distance from the nearest contour's pixel.

## Inner Distance Transform: 
The background of the product image is black and the pixels inside the components have the maximum euclidean distance from the nearest contour's pixel.

## Outer Distance Transform:
The foreground of the product image is black and the pixels outside the components have the maximum euclidean distance from the nearest contour's pixel.

## Signed Distance Transform: 
Pixels outside the components have the maximum positive euclidean distance from the nearest contour's pixel and pixels inside the components have the minimum negative euclidean distance from the nearest contour's pixel.

### Process:
1. Grayscaling
2. Binarization
3. Components threshold applying
4. Contours calculating with cv::findContours build-in function.
5. Distance transform calculation using our custom function.
6. Normalization using cv::normalize build-in function.
7. Show original and product images.

### Note:
For Inner distance transform we calculate it based on the binary image of the input image.
For Outer distance transform we calculate is based on the inversed of the binary image of the input image.
For the signed distance transform, we caculate both inner and outer distance maps and the signed distance transform is the difference between the outer and inner distance maps.

# Gaussian Filtering 
I calculate the filter kernel by iterating from 0 to desired kernel size and I substitute the index f(i - (kernel size) // 2, j - (kernel size) // 2), such that:
$$ f(x,y)= {1 \over 2\pi\sigma} e^{- (x^2 + y^2) \over 2 \sigma^2} $$

Then we have to normalize the filter kernel and make sure the sum of its elements equals to 1.\
I apply the filter by convolving the image with the kernel filter.\
At the end we got our blurred/gaussian filtered image.

# Edge Detection
First, I perform Smoothing using Gaussian filter with kernel size equals to 3 and sigma equals to 1.0.\
Second, I pop a pre-prepared sobel Y kernel, I normalise it and I derive the sobel X from it by transpose it (My program supports till 9x9 sobels).\
Third, I convolve the image with sobel Y and X, so I get the following results Gx and Gy, then I calculate the magnitude with the following formula:$$G = \sqrt{Gx^2 + Gy^2}$$\
Fourth, I normalize the pixels to [0 - 255].\
Finaly, I perform a threshold with a given thresh.
# Selection Filtering
## MIN Filter: 
I have implemented the minimum filter in way that it select the minimum intensity in the neighborhood based on the given filter size.
## MAX Filter:
I have implemented the maximum filter in way that it select the maximum intensity in the neighborhood based on the given filter size.
## MEDIAN Filter: 
I have implemented the median filter in way that it select the median intensity in the neighborhood based on the given filter size, I find the median by pushing all the intensities of the neighbourhood into a buffer then I sort the buffer and I select the element in the middle (length//2).\
Note: Sorting the buffer when k is the filter size which is not that big:
 $$O(k^2 log(k^2))$$
