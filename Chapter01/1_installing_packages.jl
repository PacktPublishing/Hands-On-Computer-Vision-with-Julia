# installation
using Pkg
Pkg.add("Images")
Pkg.add("ImageMetadata")
Pkg.add("ImageView")
Pkg.add("TestImages")
Pkg.update()

# validation
using Images, ImageMetadata, TestImages, ImageView