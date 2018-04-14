using Images, ImageView
img = load("sample-images/cats-3061372_640.jpg");
img_area_range = 320:640

img_area = img[:, img_area_range]
img_area = imfilter(img_area, Kernel.gaussian(5))
img[:, img_area_range] = img_area

imshow(img)

# create a border of a fixed style based on a content from an image
using Images, ImageView

border_size = 50
gaussian_kernel_value = 10

img = load("sample-images/cats-3061372_640.jpg");

# add borders
img = padarray(img, Pad(:reflect, border_size, 0))
img = parent(img) # reset the indices after using padarray

# apply blurring to top border
img_area_top = 1:border_size
img[img_area_top, :] = imfilter(img[img_area_top, :], Kernel.gaussian(gaussian_kernel_value))

# apply blurring to bottom border
img_area_bottom = size(img, 1)-border_size:size(img, 1)
img[img_area_bottom, :] = imfilter(img[img_area_bottom, :], Kernel.gaussian(gaussian_kernel_value))

imshow(img)