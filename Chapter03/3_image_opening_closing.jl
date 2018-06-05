using Images, ImageView, ImageMorphology

geom_img = load("sample-images/geometrical-figures-and-noise.jpg");
geom_img_binary = Gray.(Gray.(geom_img) .> 0.5);

geom_img_binary_o = opening(geom_img_binary)
imshow(geom_img_binary_e)

geom_img_binary_c = closing(geom_img_binary)
imshow(geom_img_binary_c)