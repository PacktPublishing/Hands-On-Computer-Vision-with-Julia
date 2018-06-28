using Images, MXNet

### LOADING THE MODEL

const MODEL_NAME = "weights/inception-v3/Inception-7"
const MODEL_CLASS_NAMES = "weights/inception-v3/synset.txt"

nnet = mx.load_checkpoint(MODEL_NAME, 1, mx.FeedForward);
synset = readlines(MODEL_CLASS_NAMES);

### PREPARING THE INPUT

img = imresize(load("sample-images/bird-3183441_640.jpg"), (299, 299));
img = permutedims(Float16.(channelview(img)), (3, 2, 1));

img[:, :, :] *= 256.0;
img[:, :, :] -= 128.;
img[:, :, :] /= 128.;

img = reshape(img, 299, 299, 3, 1);

mx_data = mx.zeros((299, 299, 3, 1));
mx_data[1:1] = img;
data_provider = mx.ArrayDataProvider(:data => mx_data);

### PREDICTING

@time pred = mx.predict(nnet, data_provider)
mxval, mxindx = findmax(pred[:, 1]);
println(mxval, " ", mxindx, " ", synset[mxindx])

