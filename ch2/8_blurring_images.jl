using Images, ImageView
img = load("sample-images/cats-3061372_640.jpg");
img_area_range = 320:640

img_area = img[:, img_area_range]
img_area = imfilter(img_area, Kernel.gaussian(5))
img[:, img_area_range] = img_area

imshow(img)