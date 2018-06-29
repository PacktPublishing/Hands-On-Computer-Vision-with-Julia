using Images, ImageFeatures, ImageDraw

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

# Creating panorama

using Images, ImageFeatures
img = load("sample-images/cat-3418815_640.jpg")

img_width = size(img, 2)
img_left_width = 400
img_right_width = 340

img_left = view(img, :, 1:img_left_width)
img_left_gray = Gray.(img_left)
img_right = view(img, :, (img_width - img_right_width):img_width)
img_right_gray = Gray.(img_right)

keypoints_1 = Keypoints(fastcorners(img_left_gray, 12, 0.3));
keypoints_2 = Keypoints(fastcorners(img_right_gray, 12, 0.3));

brief_params = BRIEF()
desc_1, ret_features_1 = create_descriptor(img_left_gray, keypoints_1, brief_params);
desc_2, ret_features_2 = create_descriptor(img_right_gray, keypoints_2, brief_params);
matches = match_keypoints(ret_features_1, ret_features_2, desc_1, desc_2, 0.15)

grid = hcat(img_left, img_right)
offset = CartesianIndex(0, size(img_left_gray, 2))
map(m -> draw!(grid, LineSegment(m[1], m[2] + offset)), matches)
imshow(grid)


offset_x = median(map(m -> (img_left_width - m[1][2]) + m[2][2], matches))
offset_x_half = Int(trunc(offset_x / 2))
img_output = hcat(
    img_left[:, 1:(img_left_width-diff_on_x_half)],
    img_right[:, diff_on_x_half:img_right_width]
)

imshow(hcat(restrict(img), restrict(img_output)))