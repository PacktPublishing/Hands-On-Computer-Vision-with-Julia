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
img_width = size(img, 2)

new_img = hcat(
    segment_to_image(segments_1),
    segment_to_image(segments_2),
    segment_to_image(segments_3)
)

new_img[:, img_width] = new_img[:, img_width*2] = colorant"black"

imshow(new_img)

# multiple objects
img = load("sample-images/bird-3183441_640.jpg");
#imshow(img);

# find segments
segments_1 = felzenszwalb(img, 10)
segments_2 = felzenszwalb(img, 30, 50)
segments_3 = felzenszwalb(img, 35, 300)

# create a new image which will be a stack
# of 3 different results of felzenszwalb algorithm
img_width = size(img, 2)

new_img = hcat(
    segment_to_image(segments_1),
    segment_to_image(segments_2),
    segment_to_image(segments_3)
)

new_img[:, img_width] = new_img[:, img_width*2] = colorant"black"

imshow(new_img)
