# Convert to MAT function compatible with latest version of Julia
# (c) original by https://github.com/kvmanohar22/OpenCV.jl
# (c) improvements by Dmitrijs Cudihins

function convertToMat(image)
    img = permuteddimsview(channelview(image), (2,3,1))
    cd = Base.size(channelview(image))[1] > 3 ? 1 : 3
    _rows = Base.size(image, 1)
    _cols = Base.size(image, 2)

    mat = nothing

    if (cd < 3); mat = Mat(_rows, _cols, CV_32FC1);
    elseif (cd == 3); mat = Mat(_rows, _cols, CV_32FC3); end 

    if (cd < 3)   # grayscale or binary
        for j = 1:_rows     # index row first (Mat is row-major order)
            for k =1:_cols  # index column second
                # slow algorithm  - will try to use pointer method (C++)!
                pixset(mat, j-1, k-1, Float64(img[j,k,1].i))
            end
        end
    end

    #println(size(img), _rows, _cols)

   if (cd == 3)   # color (RGB) image
        for j = 1:_rows     # index row first (Mat is row-major order)
            for k =1:_cols  # index column second
                #colorvec = tostdvec([float(img[j,k,1].i),float(img[j,k,2].i),float(img[j,k,3].i)])
                colorvec = tostdvec(Array(Float64.(img[j, k, :])))
                pixset(mat, j-1, k-1, colorvec)   # -1 to have 0-indexing per C++
            end
        end
    end

    return(mat)
end

