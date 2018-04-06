# using colorview function to convert an array to image
using Images, ImageView
random_img_array = rand(3, 8, 8); # channel, height, width
img = colorview(RGB, random_img_array);
imshow(img)

# changing order of dimensions prior to conversion
using Images, ImageView
random_img_array = rand(40, 100, 3); # height, width, channel
img_perm = permuteddimsview(random_img_array, (3, 1, 2))
img = colorview(RGB, Float16.(img_perm))
imshow(img)

# the alternative way of writing the same code is as follows:
using Images, ImageView
random_img_array = rand(40, 100, 3); # height, width, channel
img_perm = permutedims(random_img_array, [3, 1, 2])
img = colorview(RGB, img_perm)
imshow(img)