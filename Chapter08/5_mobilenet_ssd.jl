using Images, MXNet

nnet = mx.load_checkpoint("weights/mobilenet-ssd-512/deploy_ssd_300", 0, mx.FeedForward; context = mx.cpu());

img = load("sample-images/women-1209678_640.jpg");
img = Float16.(permuteddimsview(channelview(imresize(img, (512, 512))), (3, 2, 1)));

mx_data = mx.zeros((512, 512, 3, 1));
mx_data[1:1] = reshape(img, (512, 512, 3, 1));
mx_data[2:2] = reshape(img, (512, 512, 3, 1));
mx_data[3:3] = reshape(img, (512, 512, 3, 1));
mx_data[4:4] = reshape(img, (512, 512, 3, 1));
mx_data[5:5] = reshape(img, (512, 512, 3, 1));

data_provider = mx.ArrayDataProvider(:data => mx_data);
a = mx.predict(nnet, data_provider)
