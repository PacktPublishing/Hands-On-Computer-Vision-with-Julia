using Images, MXNet

### LOADING THE MODEL
const MODEL_NAME = "weights/mobilenet-v2/mobilenet_v2"
const MODEL_CLASS_NAMES = "weights/mobilenet-v2/synset.txt"

nnet = mx.load_checkpoint(MODEL_NAME, 0, mx.FeedForward; context = mx.gpu());
synset = readlines(MODEL_CLASS_NAMES);

### SEARCH FOR A LAYER OF INTERESTS
layers = mx.get_internals(nnet.arch);
layers_flatten = nothing
layers_to_remove = Symbol[]

# We iterate over all layers until we find the one matching our requirements
# and remove the ones to follow after
for i = 1:2000

    layer = layers[i];
    layer_name = mx.get_name(layer)
    
    if layers_flatten == nothing && layer_name == :pool6
        layers_flatten = layer
    elseif layers_flatten != nothing
        push!(layers_to_remove, layer_name)
        if layer_name in [:softmax, :label, :prob] break end
    end
end

# UPDATE NETWORK STRUCTURE WITH NEW
nnet.arch = @mx.chain layers_flatten => Flatten()

### REMOVE SOFTMAX OUTPUT AND FULLY CONNECTED LAYER
map(x -> delete!(nnet.arg_params, x), layers_to_remove);
map(x -> delete!(nnet.aux_params, x), layers_to_remove);

mx.save_checkpoint(nnet, "weights/mobilenet-v2/MobiletNet-FE", mx.OptimizationState(1, 0, 0, 0))

### PREPARING THE INPUT

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

@time pred = mx.predict(nnet, data_provider)