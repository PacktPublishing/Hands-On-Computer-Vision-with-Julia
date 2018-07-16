using Images, MXNet

### LOADING THE MODEL
const MODEL_NAME = "weights/inception-v3/Inception-7"

nnet = mx.load_checkpoint(MODEL_NAME, 1, mx.FeedForward);

layers = mx.get_internals(nnet.arch);
layers_flatten = nothing
layers_to_remove = Symbol[]

for i = 1:2000

    layer = layers[i];
    layer_name = mx.get_name(layer)
    
    if layers_flatten == nothing && layer_name == :flatten
        layers_flatten = layer
    elseif layers_flatten != nothing
        push!(layers_to_remove, layer_name)
        if layer_name in [:softmax, :label] break end
    end
end

nnet.arch = layers_flatten
map(x -> delete!(nnet.arg_params, x), layers_to_remove);
map(x -> delete!(nnet.aux_params, x), layers_to_remove);

mx.save_checkpoint(nnet, "weights/inception-v3/InceptionV3-FE", mx.OptimizationState(1, 0, 0, 0))