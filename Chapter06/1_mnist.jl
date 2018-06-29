using Images, ImageView, MLDatasets

train_x, train_y = MNIST.traindata();
test_x,  test_y  = MNIST.testdata();

# preview the images from train_x dataset
preview_img = zeros(size(train_x, 1), 0)

for i = 1:10
    preview_img = hcat(preview_img, train_x[:, :, i])
end

imshow(Gray.(preview_img))


# creating first neural network
using MXNet

train_length = 50000
validation_length = 10000

train_data_array  = mx.zeros((size(train_x, 1, 2)..., train_length...));
train_label_array = mx.zeros(train_length);

validation_data_array  = mx.zeros((size(train_x, 1, 2)..., validation_length...));
validation_label_array = mx.zeros(validation_length);

# The number of records we send to the training should be at least number of outcome
for idx = 1:train_length
    train_data_array[idx:idx] = reshape(train_x[:, :, idx], (size(train_x, 1, 2)..., 1...))
    train_label_array[idx:idx] = train_y[idx]
end

for idx = 1:validation_length
    validation_data_array[idx:idx] = reshape(train_x[:, :, train_length + idx], (size(train_x, 1, 2)..., 1...))
    validation_label_array[idx:idx] = train_y[train_length + idx]
end

arch = @mx.chain mx.Variable(:data) =>
  mx.FullyConnected(num_hidden=64) =>
  mx.FullyConnected(num_hidden=10) =>
  mx.SoftmaxOutput(mx.Variable(:label))

nnet = mx.FeedForward(arch, context = mx.cpu())

train_data_provider = mx.ArrayDataProvider(:data => train_data_array, :label => train_label_array, batch_size = 1000);
validation_data_provider = mx.ArrayDataProvider(:data => validation_data_array, :label => validation_label_array, batch_size = 1000);

mx.fit(nnet, mx.ADAM(), train_data_provider, eval_data = validation_data_provider, n_epoch = 50, callbacks = [mx.speedometer()]);
 
# more complicated network
arch = @mx.chain mx.Variable(:data) =>
  mx.FullyConnected(num_hidden=128) =>
  mx.Activation(act_type=:relu) =>
  mx.FullyConnected(num_hidden=64) =>
  mx.Activation(act_type=:relu) =>
  mx.FullyConnected(num_hidden=10) =>
  mx.SoftmaxOutput(mx.Variable(:label))

nnet = mx.FeedForward(arch, context = mx.cpu())

train_data_provider = mx.ArrayDataProvider(:data => train_data_array, :label => train_label_array, batch_size = 1000);
validation_data_provider = mx.ArrayDataProvider(:data => validation_data_array, :label => validation_label_array, batch_size = 1000);

mx.fit(nnet, mx.ADAM(), train_data_provider, eval_data = validation_data_provider, n_epoch = 30, callbacks = [mx.speedometer()]);

# predicting
data_array  = mx.zeros((size(test_x, 1, 2)..., 10));
mx.copy!(data_array, test_x[:, :, 1:10]);

data_provider = mx.ArrayDataProvider(:data => data_array);
results = round(mx.predict(nnet, data_provider; verbosity = 0), 2)

preview_img = zeros(size(test_x, 1), 0)

for i = 1:10
    preview_img = hcat(preview_img, test_x[:, :, i])
end

map(x -> findmax(results[:, x])[2] - 1, 1:10)
imshow(preview_img)