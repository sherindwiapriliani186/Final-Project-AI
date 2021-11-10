import cv2

#membaca image
image = cv2.imread("Daun sirih/001.jpg")
print(image)

#menampilkan image
cv2.imshow('Daun sirih/001', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
print(image.shape)

#mengakses nilai di pixel x=100, y=20
(b, g, r) = image[20, 100]
print("blue = ", b)
print("green = ", g)
print("red = ", r)

#crop image
im_crop = image[100:660, 100:380]

cv2.imshow('crop',im_crop)
cv2.waitKey(0)
cv2.destroyAllWindows()
print(im_crop.shape)

#mengcopy image
cp_image = image.copy()
print(cp_image.shape)

#mangubah nilai pixel
cp_image[300:350, 170:300] = (255, 255, 255)

cv2.imshow('cp_image',cp_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

#resize image (ignore aspect ration)
im_resized = cv2.resize(image, (400,400))

cv2.imshow('im_resized',im_resized)
cv2.waitKey(0)
cv2.destroyAllWindowns()

#resize image (mempertahankan aspect ratio)
r = 400/image.shape[1]
dim = (400,int(image.shape[0]*r))
im_resized = cv2.resized(image, dim)

cv2.imshow('im_resized',im_resized)
cv2.waitKey(0)
cv2.destroyAllWindows()

#rotating an image
(h, w) = image.shape[:2]
center = (w/2, h/2)

M = cv2.getRotationMatrix2D(center, 180, 1.0)
rotated = cv2.warpAffine(image, M (w,h))

cv2.imshow('rotate',rotated)
cv2.waitKey(0)
cv2.destroyAllWindows()

#menyimpan image
cv2.imwrite("rotate.png", rotated)

#adjust image contrast
import numpy as np
im_adjusted = cv2.addWeighted(im_resized, 1.5, np.zeros(im_resized.shape, im_resized.dtype), 0, -100)

cv2.imshow('Original Image',im_resized)
cv2.imshow('Adjusted Image',im_resized)
cv2.waitKey(0)
cv2.destroyAllWindows()

#delect edges
im_edges = cv2.Canny(im_resized, 100, 200)

cv2.imshow('Original Image',im_resized)
cv2.imshow('Detected Image',im_edges)
cv2.waitKey(0)
cv2.destroyAllWindows()

#convert image to grayscale
im_gray = cv2.cvtColor(im_resized, cv2.COLOR_BGR2GRAY)

cv2.imshow('Original Image',im_resized)
cv2.imshow('Grayscale Image',im_gray)
cv2.waitKey(0)
cv2.destroyAllWindows()

#get all files in a folder
import glob

imdir = 'lidah buaya/elephant_contrast/'
ext = ['jpg'] #add image formats here

files = []
[files.extend(glob.glob(imdir + '*.' + e)) for e in ext]

image = [cv2.imread(file) for file in files]

#adjust contrast to all of them nd save to different location
i = 1
for img in image:
    im_adjusted = cv2.addWeighted(img, 1.5, np.zeros(img.shape, img.dtype), 0, -100)
    im_name = "lidah buaya/elephant_contrast/" + str(i) + ".jpg"
    cv2.imwrite(im_name, im_adjusted)
    i+=1