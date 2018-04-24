using Images, ImageView, ImageMorphology

# examples on using tophat and bothat
geom_img = load("sample-images/geometrical-figures-and-noise.jpg");
geom_img_gray = Gray.(geom_img);
geom_img_th = tophat(geom_img_gray)
geom_img_bh = bothat(geom_img_gray)

geom_img_new = zeros(ColorTypes.Gray{FixedPointNumbers.Normed{UInt8,8}}, size(geom_img_gray) .* (1, 2));
geom_img_new_center = Int(size(geom_img_gray, 2))

geom_img_new[:, 1:geom_img_new_center] = geom_img_th
geom_img_new[:, geom_img_new_center:end - 1] = geom_img_bh
geom_img_new[:, geom_img_new_center] = 1

imshow(Gray.(geom_img_new))

# adjusting image contrast
minmax = scaleminmax(Gray, 0, 1)

img = load("sample-images/fabio_gray_512.png");
img_gray = Gray.(img);
img_new = minmax.(img_gray + tophat(img_gray) - bothat(img_gray))

img_center = Int(size(img_gray, 2) / 2)
img_gray[:, img_center:end] = img_new[:, img_center:end]
img_gray[:, img_center] = 0
imshow(img_gray)

img = load("sample-images/busan-night-scene.jpg");
img_gray = Gray.(img);
img_new = minmax.(img_gray + tophat(img_gray) * 0.5 - bothat(img_gray) * 0.5)

img_center = Int(size(img_gray, 2) / 4)
img_gray[:, img_center:end] = img_new[:, img_center:end]
img_gray[:, img_center] = 0
imshow(img_gray)