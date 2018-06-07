using Images, ImageFeatures

img1 = Gray.(load("cat-3417184_640.jpg"))
img2 = Gray.(load("cat-3417184_640_watermarked.jpg"))

keypoints_1 = Keypoints(fastcorners(img1, 12, 0.5));
keypoints_2 = Keypoints(fastcorners(img2, 12, 0.5));

size(keypoints_1)
size(keypoints_2)

brief_params = BRIEF()
desc_1, ret_features_1 = create_descriptor(img1, keypoints_1, brief_params);
desc_2, ret_features_2 = create_descriptor(img2, keypoints_2, brief_params);
matches = match_keypoints(ret_features_1, ret_features_2, desc_1, desc_2, 0.5)

grid = Gray.(hcat(img1, img2))
offset = CartesianIndex(0, size(img1, 2))
map(m -> draw!(grid, LineSegment(m[1], m[2] + offset)), matches)
grid


img3 = Gray.(load("cat-3418815_640.jpg"))
