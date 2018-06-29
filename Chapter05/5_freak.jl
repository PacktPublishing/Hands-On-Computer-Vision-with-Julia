using Images, ImageFeatures, CoordinateTransformations, ImageDraw

img1 = Gray.(load("cat-3417184_640.jpg"))
img2 = Gray.(load("cat-3417184_640_watermarked.jpg"))

# rotate img2 around the center and resize it
rot = recenter(RotMatrix(5pi/6), [size(img2)...] .÷ 2) 
tform = rot ∘ Translation(-50, -40)
img2 = warp(img2, tform, indices(img2))
img2 = imresize(img2, Int.(trunc.(size(img2) .* 0.7)))

keypoints_1 = Keypoints(fastcorners(img1, 12, 0.35));
keypoints_2 = Keypoints(fastcorners(img2, 12, 0.35));

freak_params = FREAK()
desc_1, ret_keypoints_1 = create_descriptor(img1, keypoints_1, freak_params);
desc_2, ret_keypoints_2 = create_descriptor(img2, keypoints_2, freak_params);
matches = match_keypoints(ret_keypoints_1, ret_keypoints_2, desc_1, desc_2, 0.2)

# create
img3 = zeros(size(img1, 1), size(img2, 2))
img3[1:size(img2, 1), 1:size(img2, 2)] = img2

grid = Gray.(hcat(img1, img3))
offset = CartesianIndex(0, size(img1, 2))
map(m -> draw!(grid, LineSegment(m[1], m[2] + offset)), matches)

imshow(grid)