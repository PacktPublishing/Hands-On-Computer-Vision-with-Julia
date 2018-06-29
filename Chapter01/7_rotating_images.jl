using Images, CoordinateTransformations, ImageView

img = load("sample-images/cats-3061372_640.jpg");
tfm = LinearMap(RotMatrix(-pi/3)) # rotate by -60 degrees 
img = warp(img, tfm)
imshow(img)
