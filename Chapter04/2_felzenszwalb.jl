using Images, ImageView, ImageSegmentation

# convert image segments to mean color value
segment_to_image(segments) = map(i->segment_mean(segments, i), labels_map(segments))

# one object
img = load("sample-images/cat-3352842_640.jpg");
imshow(img);

# find segments
segments_1 = felzenszwalb(img, 75)
segments_2 = felzenszwalb(img, 75, 150)
segments_3 = felzenszwalb(img, 75, 350)


# define helper values
color_black = RGB4(0.,0.,0.)
img_width = size(img, 2)

# create a new image which will be a stack
# of 3 different results of felzenszwalb algorithm
new_img = fill(color_black, size(img) .* (1, 3))
new_img[:, 1:img_width] = segment_to_image(segments_1)
new_img[:, img_width+1:img_width*2] = segment_to_image(segments_2)
new_img[:, img_width*2+1:img_width*3] = segment_to_image(segments_3)
new_img[:, img_width] = new_img[:, img_width*2] = RGB4(0.,0.,0.)

# preview the result
imshow(map(i->segment_mean(segments,i), labels_map(segments)))
# save("cat-felzenszwalb-123.jpg", restrict(new_img))

# multiple objects
img = load("sample-images/bird-3183441_640.jpg");
#imshow(img);

# find segments
segments_1 = felzenszwalb(img, 10)
segments_2 = felzenszwalb(img, 30, 50)
segments_3 = felzenszwalb(img, 35, 300)

# define helper values
color_black = RGB4(0.,0.,0.)
img_width = size(img, 2)

# create a new image which will be a stack
# of 3 different results of felzenszwalb algorithm
new_img = fill(color_black, size(img) .* (1, 3))
new_img[:, 1:img_width] = segment_to_image(segments_1)
new_img[:, img_width+1:img_width*2] = segment_to_image(segments_2)
new_img[:, img_width*2+1:img_width*3] = segment_to_image(segments_3)
new_img[:, img_width] = new_img[:, img_width*2] = RGB4(0.,0.,0.)

imshow(new_img)
save("bird-felzenszwalb-123.jpg", restrict(new_img))
