using Images, ImageView, ImageSegmentation

# convert image segments to mean color value
segment_to_image(segments) = map(i->segment_mean(segments, i), labels_map(segments))

# one object
img = load("sample-images/cat-3352842_640.jpg");
imshow(img);

# find segments
segments_1 = fast_scanning(img, 0.05)
segments_2 = fast_scanning(img, 0.15)
segments_3 = fast_scanning(img, 0.2)

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

# prune segments
deletion_rule = i -> (segment_pixel_count(segments_3,i) < 750)
replacement_rule = (i,j) -> (-segment_pixel_count(segments_3, j))
segments_n = prune_segments(segments_3, deletion_rule, replacement_rule)

# preview the result
segment_to_image(segments_n)
# save("cat-felzenszwalb-123.jpg", restrict(new_img))

# multiple objects
img = load("sample-images/bird-3183441_640.jpg");
#imshow(img);

# find segments
segments_1 = fast_scanning(img, 0.05)
segments_2 = fast_scanning(img, 0.15)
segments_3 = fast_scanning(img, 0.2)

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

deletion_rule = i -> (segment_pixel_count(segments_2,i) < 750)
replacement_rule = (i,j) -> (-segment_pixel_count(segments_2, j))
segments_n = prune_segments(segments_2, deletion_rule, replacement_rule)

# preview the result

imshow(new_img)