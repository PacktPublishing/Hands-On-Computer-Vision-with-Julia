using Images, MLDatasets, MXNet, ImageView

train_x, train_y = CIFAR10.traindata();
test_x,  test_y  = CIFAR10.testdata();


# preview the images from train_x dataset
preview_img = zeros((0..., size(train_x, 1, 3)...));

for i = 1:10
    preview_img = vcat(preview_img, train_x[:, :, :, i])
end

imshow(colorview(RGB, permutedims(preview_img, (3, 2, 1))))

### PREPARING THE DATASET

train_x = train_x ./ 1;
test_x = test_x ./ 1;

train_length = 40000
validation_length = 10000

train_data_array  = mx.zeros((size(train_x, 1, 2, 3)..., train_length...));
train_label_array = mx.zeros(train_length);

validation_data_array  = mx.zeros((size(train_x, 1, 2, 3)..., validation_length...));
validation_label_array = mx.zeros(validation_length);

test_data_array  = mx.zeros((size(train_x, 1, 2, 3)..., size(test_x, 4)...));
test_label_array = mx.zeros(size(test_x, 4));

for idx = 1:train_length
    train_data_array[idx:idx] = reshape(train_x[:, :, :, idx], (size(train_x, 1, 2,3 )..., 1...))
    train_label_array[idx:idx] = train_y[idx]
end

for idx = 1:validation_length
    validation_data_array[idx:idx] = reshape(train_x[:, :, :, train_length + idx], (size(train_x, 1, 2,3 )..., 1...))
    validation_label_array[idx:idx] = train_y[train_length + idx]
end

for idx = 1:size(test_x, 4)
    test_data_array[idx:idx] = reshape(test_x[:, :, :, idx], (size(test_x, 1, 2,3 )..., 1...))
    test_label_array[idx:idx] = test_y[idx]
end

train_data_provider = mx.ArrayDataProvider(:data => train_data_array, :label => train_label_array, batch_size = 100, shuffle = true);
validation_data_provider = mx.ArrayDataProvider(:data => validation_data_array, :label => validation_label_array, batch_size = 100, shuffle = true);
test_data_provider = mx.ArrayDataProvider(:data => test_data_array, :label => test_label_array, batch_size = 100);

### Simple 1-layer NN

arch = @mx.chain mx.Variable(:data) =>
  mx.Flatten() =>
  mx.FullyConnected(num_hidden=128) =>
  mx.Activation(name=:relu1, act_type=:relu) =>
  mx.FullyConnected(num_hidden=10) =>
  mx.SoftmaxOutput(mx.Variable(:label))

nnet = mx.FeedForward(arch, context = mx.cpu())
mx.fit(nnet, mx.ADAM(), train_data_provider, eval_data = validation_data_provider, n_epoch = 50, initializer = mx.XavierInitializer());
 
### Simple 2-layer NN

arch = @mx.chain mx.Variable(:data) =>
  mx.FullyConnected(num_hidden=128) =>
  mx.Activation(act_type=:relu) =>
  mx.FullyConnected(num_hidden=64) =>
  mx.Activation(act_type=:relu) =>
  mx.FullyConnected(num_hidden=10) =>
  mx.Activation(act_type=:relu) =>
  mx.SoftmaxOutput(mx.Variable(:label))

nnet = mx.FeedForward(arch, context = mx.cpu())
mx.fit(nnet, mx.ADAM(), train_data_provider, eval_data = validation_data_provider, n_epoch = 50, initializer = mx.XavierInitializer());

## Convolution Neural Network

arch = @mx.chain mx.Variable(:data) =>
        mx.Convolution(kernel=(3, 3), num_filter=32) =>
        mx.Activation(act_type=:relu) =>
        mx.Dropout(p = 0.25) => 
        mx.Pooling( kernel=(2, 2), pool_type=:max) =>
        mx.Flatten() =>
        mx.FullyConnected(num_hidden=256) =>
        mx.Activation(act_type=:relu) =>
        mx.FullyConnected(num_hidden=10) =>
        mx.SoftmaxOutput(mx.Variable(:label))

nnet = mx.FeedForward(arch, context = mx.cpu())
mx.fit(nnet, mx.ADAM(), train_data_provider, eval_data = test_data_provider, n_epoch = 50, initializer = mx.XavierInitializer());