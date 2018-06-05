using Images, ImageView, ImageMorphology

geom_img = load("sample-images/geometrical-figures-and-noise.jpg");
geom_img_binary = Gray.(Gray.(geom_img) .> 0.5);
geom_img_binary_e = erode(geom_img_binary)
imshow(geom_img_binary_e)
geom_img_binary_e = erode(erode(geom_img_binary_e))
imshow(geom_img_binary_e)
geom_img_binary_e = erode(erode(geom_img_binary_e))
imshow(geom_img_binary_e)

carplate_img = load("sample-images/caribbean-2726429_640.jpg")
carplate_img_binary = Gray.(Gray.(carplate_img) .< 0.5);
carplate_img_binary_e = erode(carplate_img_binary)
imshow(geom_img_binary_e)
carplate_img_binary_e = erode(erode(carplate_img_binary_e))
imshow(geom_img_binary_e)

geom_img = load("sample-images/geometrical-figures-and-noise.jpg");
geom_img_binary = Gray.(Gray.(geom_img) .> 0.5);
geom_img_binary_d = dilate(geom_img_binary)
imshow(geom_img_binary_d)
geom_img_binary_d = dilate(dilate(geom_img_binary_d))

carplate_img = load("sample-images/caribbean-2726429_640.jpg")
carplate_img_binary = Gray.(Gray.(carplate_img) .< 0.5)
carplate_img_binary_d = dilate(carplate_img_binary)
imshow(carplate_img_binary_d)