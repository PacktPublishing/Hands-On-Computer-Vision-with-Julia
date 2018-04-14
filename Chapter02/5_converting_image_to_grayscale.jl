# convert an image to grayscale
using Images
img = load("sample-images/cats-3061372_640.jpg");
img_gray = Gray.(img)
imshow(img_gray)

# create 3 channel RGB from a grayscale
img_gray_rgb = RGB.(Gray.(img_gray))
imshow(img_gray_rgb)

