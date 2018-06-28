using Images, MXNet

### LOADING THE MODEL

const MODEL_NAME = "weights/mobilenet-v2/mobilenet_v2"
const MODEL_CLASS_NAMES = "weights/mobilenet-v2/synset.txt"

nnet = mx.load_checkpoint(MODEL_NAME, 0, mx.FeedForward);
synset = readlines(MODEL_CLASS_NAMES);


img = imresize(load("sample-images/bird-3183441_640.jpg"), (224, 224));
img = permutedims(Float16.(channelview(img)), (3, 2, 1));

img[:, :, :] *= 256.0;
img[:, :, :] -= 128.;
img[:, :, :] /= 128.;

img = reshape(img, 224, 224, 3, 1);

mx_data = mx.zeros((224, 224, 3, 1));
mx_data[1:1] = img;
data_provider = mx.ArrayDataProvider(:data => mx_data);

### PREDICTING

@time pred = mx.predict(nnet, data_provider);
mxval, mxindx = findmax(pred[:, 1]);
println(mxval, " ", mxindx, " ", synset[mxindx])