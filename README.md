## $5 Tech Unlocked 2021!
[Buy and download this Book for only $5 on PacktPub.com](https://www.packtpub.com/product/hands-on-computer-vision-with-julia/9781788998796)
-----
*If you have read this book, please leave a review on [Amazon.com](https://www.amazon.com/gp/product/1788998790).     Potential readers can then use your unbiased opinion to help them make purchase decisions. Thank you. The $5 campaign         runs from __December 15th 2020__ to __January 13th 2021.__*

# Hands-On Computer Vision with Julia

<a href="https://www.packtpub.com/application-development/hands-computer-vision-julia?utm_source=github&utm_medium=repository&utm_campaign=9781788998796"><img src="https://d1ldz4te4covpm.cloudfront.net/sites/default/files/imagecache/ppv4_main_book_cover/10308_cover.png" alt="Hands-On Computer Vision with Julia" height="256px" align="right"></a>

This is the code repository for [Hands-On Computer Vision with Julia](https://www.packtpub.com/application-development/hands-computer-vision-julia?utm_source=github&utm_medium=repository&utm_campaign=9781788998796), published by Packt.

**Build complex applications with advanced Julia packages for image processing, neural networks, and Artificial Intelligence**

## What is this book about?
Hands-On Computer Vision with Julia is a thorough guide for developers who want to get started with building computer vision applications using Julia. Julia is well suited to image processing because it’s easy to use and lets you write easy-to-compile and efficient machine code.

This book covers the following exciting features: 
* Analyze image metadata and identify critical data using JuliaImages
* Apply filters and improve image quality and color schemes
* Extract 2D features for image comparison using JuliaFeatures
* Cluster and classify images with KNN/SVM machine learning algorithms
* Recognize text in an image using the Tesseract library

If you feel this book is for you, get your [copy](https://www.amazon.com/dp/1788998790) today!

<a href="https://www.packtpub.com/?utm_source=github&utm_medium=banner&utm_campaign=GitHubBanner"><img src="https://raw.githubusercontent.com/PacktPublishing/GitHub/master/GitHub.png" 
alt="https://www.packtpub.com/" border="5" /></a>


## Instructions and Navigations
All of the code is organized into folders. For example, Chapter02.

The code will look like the following:
 ```
 using Images
 sample_image_path = "sample-images/cats-3061372_640.jpg";
 sample_image = nothing
 if isfile(sample_image_path)
  sample_image = load(sample_image_path);
 else
  info("ERROR: Image not found!")
 end
```
**Following is what you need for this book:**
Hands-On Computer Vision with Julia is for Julia developers who are interested in learning how to perform image processing and want to explore the field of computer vision. Basic knowledge of Julia will help you understand the concepts more effectively.

With the following software and hardware list you can run all code files present in the book (Chapter 1-9).

### Software and Hardware List

| Chapter  | Software required                   | OS required                        |
| -------- | ------------------------------------| -----------------------------------|
| 1 - 9        | Julia 0.6.3                     | Windows 7+, macOS 10.8+, Linux 2.6.18+, Ubuntu Linux 16.04 |
| 7        | MXNet 1.2.0         | Windows 7+, macOS 10.8+, Linux 2.6.18+, Ubuntu Linux 16.04 |
| 9        | Open CV 3.2.0            | macOS 10.8+, Linux 2.6.18++ GUI, Ubuntu Linux 16.04+(desktop version) |



We also provide a PDF file that has color images of the screenshots/diagrams used in this book. [Click here to download it](https://www.packtpub.com/sites/default/files/downloads/HandsOnComputerVisionwithJulia_ColorImages.pdf).

### Related products <Paste books from the Other books you may enjoy section>
* Julia High Performance [[Packt]](https://www.packtpub.com/application-development/julia-high-performance?utm_source=github&utm_medium=repository&utm_campaign=9781785880919) [[Amazon]](https://www.amazon.com/dp/1785880918)

* Julia for Data Science [[Packt]](https://www.packtpub.com/big-data-and-business-intelligence/julia-data-science?utm_source=github&utm_medium=repository&utm_campaign=9781785289699) [[Amazon]](https://www.amazon.com/dp/1785289691)

## Get to Know the Author
**Dmitrijs Cudihins**
is a skilled data scientist, machine learning engineer, and software developer with more than eight years' commercial experience. His started off as a web developer, but later switched to data science and computer vision. He has been a senior data scientist for the last three years, providing consultancy services for a state-owned enterprise. There, he uses Julia to automate communication with citizens, applying different CV techniques and scanned image processing.


### Suggestions and Feedback
[Click here](https://docs.google.com/forms/d/e/1FAIpQLSdy7dATC6QmEL81FIUuymZ0Wy9vH1jHkvpY57OiMeKGqib_Ow/viewform) if you have any feedback or suggestions.
