# 

<a href="https://www.packtpub.com/application-development/hands-computer-vision-julia?utm_source=github&utm_medium=repository&utm_campaign="><img src="" alt="" height="256px" align="right"></a>

This is the code repository for Hands-On-Computer-Vision-with-Julia, published by Packt.

****

## What is this book about?

This book covers the following exciting features:


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
| 1-9      | Julia 0.6.3                     | Windows 7+, macOS 10.8+, Linux 2.6.18+, Ubuntu Linux 16.04 |
| 7       | MXNet 1.2.0            | Windows 7+, macOS 10.8+, Linux 2.6.18+, Ubuntu Linux 16.04 |
| 8        | Open CV 3.2.0            | macOS 10.8+, Linux 2.6.18++ GUI, Ubuntu Linux 16.04+ (desktop version) |


We also provide a PDF file that has color images of the screenshots/diagrams used in this book. [Click here to download it](https://www.packtpub.com/sites/default/files/downloads/HandsOnComputerVisionwithJulia_ColorImages.pdf).

### Related products

*  Julia High Performance [[Packt]](https://www.packtpub.com/application-development/julia-high-performance?utm_source=github&utm_medium=repository&utm_campaign=9781788998796)

* Julia for Data Science [[Packt]](https://www.packtpub.com/big-data-and-business-intelligence/julia-data-science?utm_source=github&utm_medium=repository&utm_campaign=9781785289699 )



## Get to Know the Author
**Dmitrijs Cudihins**
is a skilled data scientist, machine learning engineer, and software developer with more than eight years' commercial experience. His started off as a web developer, but later switched to data science and computer vision. He has been a senior data scientist for the last three years, providing consultancy services for a state-owned enterprise. There, he uses Julia to automate communication with citizens, applying different CV techniques and scanned image processing.

