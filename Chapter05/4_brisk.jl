using Images, ImageFeatures, CoordinateTransformations, ImageDraw

img1 = Gray.(load("cat-3417184_640.jpg"))
img2 = Gray.(load("cat-3417184_640_watermarked.jpg"))

# rotate img2 around the center and resize it
rot = recenter(RotMatrix(5pi/6), [size(img2)...] .÷ 2)  # a rotation around the center
tform = rot ∘ Translation(-50, -40)
img2 = warp(img2, tform, indices(img2))
img2 = imresize(img2, Int.(trunc.(size(img2) .* 0.7)))

features_1 = Features(fastcorners(img1, 12, 0.35));
features_2 = Features(fastcorners(img2, 12, 0.35));

brisk_params = BRISK()
desc_1, ret_features_1 = create_descriptor(img1, features_1, brisk_params);
desc_2, ret_features_2 = create_descriptor(img2, features_2, brisk_params);
matches = match_keypoints(Keypoints(ret_features_1), Keypoints(ret_features_2), desc_1, desc_2, 0.2)

# create
img3 = zeros(Gray, size(img1))
img3[1:size(img2, 1), 1:size(img2, 2)] = img2

grid = hcat(img1, img3)
offset = CartesianIndex(0, size(img1, 2))
map(m -> draw!(grid, LineSegment(m[1], m[2] + offset)), matches)

imshow(grid)
