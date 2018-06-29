using Images, ImageFeatures, ImageDraw, ImageMorphology

#### Example 1

img = Gray.(load("sample-images/cat-3417184_640.jpg"))
img_f = Float16.(restrict(img))

new_img = Gray.(hcat(
    Float32.(img_f) .* (~fastcorners(img_f, 12, 0.15)),
    Float32.(img_f) .* (~fastcorners(img_f, 12, 0.25)),
    Float32.(img_f) .* (~fastcorners(img_f, 12, 0.35))
))

new_img = Gray.(hcat(
    Float32.(img_f) .* (~fastcorners(img_f, 12, 0.15)),
    Float32.(img_f) .* (~imcorner(img_f, method=harris)),
    Float32.(img_f) .* (~imcorner(img_f, method=shi_tomasi))
))

imshow(new_img)

### Example 2

img = Gray.(load("sample-images/newspaper-37782_640.png"))
img_f = Float16.(img)

new_img = Gray.(hcat(
    img_f,
    Float16.(img_f) .* (~fastcorners(img_f, 12, 0.15))
))

imshow(new_img)


### Example 3
img = Gray.(restrict(load("sample-images/cat-3417184_640.jpg")))

img = restrict(load("sample-images/board-157165_640.png"))
img_f = Float16.(Gray.(img))

img_harris = copy(img)
img_harris[dilate(imcorner(img_f, method=harris)) .> 0.01] = colorant"yellow"

img_shi = copy(img)
img_shi[dilate(imcorner(img_f, method=shi_tomasi)) .> 0.01] = colorant"yellow"

img_rosenfield = copy(img)
img_rosenfield[dilate(imcorner(img_f, method=kitchen_rosenfeld)) .> 0.01] = colorant"yellow"

img_fast = copy(img)
img_fast[dilate(fastcorners(img_f, 12, 0.05)) .> 0.01] = colorant"yellow"

new_img = vcat(
    hcat(
        img_harris,
        img_shi),
    hcat(
        img_rosenfield,
        img_fast)
)

new_img[Int(size(new_img, 1) / 2), :] = colorant"yellow"
new_img[:, Int(size(new_img, 2) / 2)] = colorant"yellow"

imshow(new_img)


### Example 4
img = Gray.(restrict(load("sample-images/cat-3417184_640.jpg")))
img_f = Float16.(Gray.(img))

img_harris = copy(img)
img_harris[dilate(imcorner(img_f, method=harris)) .> 0.01] = colorant"yellow"

img_shi = copy(img)
img_shi[dilate(imcorner(img_f, method=shi_tomasi)) .> 0.01] = colorant"yellow"

img_rosenfield = copy(img)
img_rosenfield[dilate(imcorner(img_f, method=kitchen_rosenfeld)) .> 0.01] = colorant"yellow"

img_fast = copy(img)
img_fast[dilate(fastcorners(img_f, 12, 0.05)) .> 0.01] = colorant"yellow"

new_img = vcat(
    hcat(
        img_harris,
        img_shi),
    hcat(
        img_rosenfield,
        img_fast)
)

imshow(new_img)

### Example 5

img = Gray.(restrict(load("sample-images/cat-3417184_640.jpg")))
img_f = Float16.(Gray.(img))

img_harris = copy(img)
img_harris[dilate(imcorner(img_f, Percentile(95), method=harris)) .> 0.01] = colorant"yellow"

img_shi = copy(img)
img_shi[dilate(imcorner(img_f, Percentile(95), method=shi_tomasi)) .> 0.01] = colorant"yellow"

img_rosenfield = copy(img)
img_rosenfield[dilate(imcorner(img_f, Percentile(95), method=kitchen_rosenfeld)) .> 0.01] = colorant"yellow"

img_fast = copy(img)
img_fast[dilate(fastcorners(img_f, 12, 0.05)) .> 0.01] = colorant"yellow"

new_img = vcat(
    hcat(
        img_harris,
        img_shi),
    hcat(
        img_rosenfield,
        img_fast)
)

imshow(new_img)

### Performance comparison

img = restrict(load("sample-images/cat-3417184_640.jpg"))
img_f = Float16.(Gray.(img))

@btime fastcorners(img_f, 12, 0.15);
@btime fastcorners(img_f, 12, 0.05);
@btime imcorner(img_f, method=harris);
@btime imcorner(img_f, Percentile(95), method=harris);
@btime imcorner(img_f, method=shi_tomasi);
@btime imcorner(img_f, method=kitchen_rosenfeld);
