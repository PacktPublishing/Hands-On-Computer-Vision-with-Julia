using Images, ImageFeatures, CoordinateTransformations

img1 = Gray.(load("cat-3417184_640.jpg"))
img2 = Gray.(load("cat-3417184_640_watermarked.jpg"))

# rotation img2 around the center
rot = recenter(RotMatrix(5pi/6), [size(img2)...] .÷ 2)  # a rotation around the center
tform = rot ∘ Translation(-50, -40)
img2 = warp(img2, tform, indices(img2))

orb_params = ORB(num_keypoints = 1000)

desc_1, ret_keypoints_1 = create_descriptor(img1, orb_params)
desc_2, ret_keypoints_2 = create_descriptor(img2, orb_params)

matches = match_keypoints(ret_keypoints_1, ret_keypoints_2, desc_1, desc_2, 0.2)

grid = Gray.(hcat(img1, img2))
offset = CartesianIndex(0, size(img1, 2))
map(m -> draw!(grid, LineSegment(m[1], m[2] + offset)), matches)
grid

imshow(grid)