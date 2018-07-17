
# windows users are required to specify full path to the exe file unless they configured PATH variable
TESSERACT_PATH = "tesseract" 
image_path = joinpath("data", "book-covers", "test", "007184418X.jpg")

text_str = readstring(`$TESSERACT_PATH $image_path stdout`)
text_array = readlines(`$TESSERACT_PATH $image_path stdout`)

println(text_str)
println(text_array)