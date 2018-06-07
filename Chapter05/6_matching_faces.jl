using Images, ImageFeatures, ImageDraw, ImageView

img1 = Gray.(load("sample-images/beautiful-1274051_640_100_1.jpg"))
img2 = Gray.(load("sample-images/beautiful-1274056_640_100_2.jpg"))

keypoints_1 = Keypoints(fastcorners(img1, 12, 0.25));
keypoints_2 = Keypoints(fastcorners(img2, 12, 0.25));

freak_params = FREAK()
desc_1, ret_keypoints_1 = create_descriptor(img1, keypoints_1, freak_params);
desc_2, ret_keypoints_2 = create_descriptor(img2, keypoints_2, freak_params);
matches = match_keypoints(ret_keypoints_1, ret_keypoints_2, desc_1, desc_2, 0.2)

grid = Gray.(hcat(img1, img2))
offset = CartesianIndex(0, size(img1, 2))
map(m -> draw!(grid, LineSegment(m[1], m[2] + offset)), matches)
grid

imshow(grid)