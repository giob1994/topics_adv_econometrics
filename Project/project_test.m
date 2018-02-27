%
%   TAE - Final Project
%
%   Giovanni Ballarin, Stefanie Bertele
%

clear;
close all;
figj = 1;

train_size = 5000;
test_size  = 1000;

% Import MNIST dataset:

% X_mnist = csvread('digits_data.csv');
% y_mnist = csvread('digits_labels.csv');

mnist_data   = transpose(loadMNISTImages('mnist-train-images-idx3-ubyte'));
mnist_labels = loadMNISTLabels('mnist-train-labels-idx1-ubyte');

X_mnist = mnist_data(1:train_size, :);
y_mnist = mnist_labels(1:train_size, :);

n_mnist = size(X_mnist, 1);
p_mnist = size(X_mnist, 2);

X_mnist_test = mnist_data(train_size+1:train_size+test_size, :);
y_mnist_test = mnist_labels(train_size+1:train_size+test_size, :);

% Import Fashion MNIST dataset:

fashion_data   = transpose(loadMNISTImages('fashion-mnist-images-idx3-ubyte'));
hashion_labels = loadMNISTLabels('fashion-mnist-labels-idx1-ubyte');

X_fashion = fashion_data(1:train_size, :);
y_fashion = hashion_labels(1:train_size, :);

n_fashion = size(X_fashion, 1);
p_fashion = size(X_fashion, 2);

X_fashion_test = fashion_data(train_size+1:train_size+test_size, :);
y_fashion_test = hashion_labels(train_size+1:train_size+test_size, :);

% Import notMNIST dataset:

notMnist_data   = transpose(loadMNISTImages('notmnist-images-idx3-ubyte'));
notMnist_labels = loadMNISTLabels('notmnist-labels-idx1-ubyte');

X_notMnist = notMnist_data(1:train_size, :);
y_notMnist = notMnist_labels(1:train_size, :);

n_notMnist = size(X_notMnist, 1);
p_notMnist = size(X_notMnist, 2);

X_notMnist_test = notMnist_data(train_size+1:train_size+test_size, :);
y_notMnist_test = notMnist_labels(train_size+1:train_size+test_size, :);

clear mnist_data mnist_labels fashion_data hashion_labels ...
        notMnist_data notMnist_labels;
 
% We are using display_network from the autoencoder code
% displayData(X_fashion(1:100, :));
% disp(y_fashion(1:10));

num_labels = 10;

Y_mnist      = zeros(n_mnist, num_labels);
Y_mnist_test = zeros(test_size, num_labels);
for j = 1:train_size+test_size
    if j <= train_size
        Y_mnist(j, y_mnist(j)+1) = 1;
    else
        Y_mnist_test(j, y_mnist_test(j-train_size)+1) = 1;
    end
end

Y_notMnist      = zeros(n_notMnist, num_labels);
Y_notMnist_test = zeros(test_size, num_labels);
for j = 1:train_size+test_size
    if j <= train_size
        Y_notMnist(j, y_notMnist(j)+1) = 1;
    else
        Y_notMnist_test(j, y_notMnist_test(j-train_size)+1) = 1;
    end
end

Y_fashion      = zeros(n_fashion, num_labels);
Y_fashion_test = zeros(test_size, num_labels);
for j = 1:train_size+test_size
    if j <= train_size
        Y_fashion(j, y_fashion(j)+1) = 1;
    else
        Y_fashion_test(j, y_fashion_test(j-train_size)+1) = 1;
    end
end

% Y_fashion = zeros(n_fashion, num_labels);
% for j = 1:train_size+test_size
%     Y_fashion(j, y_fashion(j)+1) = 1;
% end
% 
% Y_notMnist = zeros(n_notMnist, num_labels);
% for j = 1:train_size+test_size
%     Y_notMnist(j, y_notMnist(j)+1) = 1;
% end

%% ANN + Test Training Algorithm:

train_alg = { 'trainrp', 'trainscg', ...
                'traincgb', 'traincgf', 'traincgp', 'trainoss', 'traingdx', ...
                'traingdm', 'traingd'};
            
NN_mnist_error_alg      = zeros(length(train_alg),1);
NN_mnist_error_alg_test = zeros(length(train_alg),1);
NN_mnist_time_alg = zeros(length(train_alg),1);
            
for k = 1:length(train_alg)
    
    net = feedforwardnet(25, char(train_alg(k)));
    tic
    net = train(net, X_mnist',  Y_mnist', [], [], [], 'useParallel');
    NN_mnist_time_alg(k) = toc;
    
    [~, NN_class] = max(net(X_mnist')', [], 2);
    NN_mnist_error_alg(k) = sum(NN_class ~= (y_mnist+1))/train_size;
    
    [~, NN_class_test] = max(net(X_mnist_test')', [], 2);
    NN_mnist_error_alg_test(k) = sum(NN_class_test ~= (y_mnist_test+1))/test_size;
    
end

results_test_train_alg = ...
    table(train_alg', NN_mnist_error_alg, NN_mnist_error_alg_test, NN_mnist_time_alg, ...
            'VariableNames',{'Algorithm','Error_int','Error_test','Time'});
        
disp(' ')
disp(' ----------------------------- ')
disp('    Test Training Algorithms   ')
disp(' ----------------------------- ')
disp(' ')
disp(results_test_train_alg)
disp(' ')

%% PCA + Neural Network

NN_architecture  = { 10;
                     25;
                     100; 
                     [25, 15];
                     [25, 20, 15];
                     [100, 50, 20] };
                    
PCA_sizes        = [10, 25, 50, 100];

PN = length(PCA_sizes)*length(NN_architecture);

%% MNIST

NN_class = zeros(train_size,PN);

NN_PCA_mnist_er      = zeros(PN,1);
NN_PCA_mnist_er_test = zeros(PN,1);
 
tmp_PCA_disp = zeros(PN,1);
tmp_Archi_disp = {};

j = 1;

for PC_dim = PCA_sizes
    
    for l = 1:length(NN_architecture)

    % PCA of the MNIST data:
    X_mnist_pca = X_mnist * pca(X_mnist);
    X_mnist_pca = X_mnist_pca(:, 1:PC_dim);

    % Neural network:
    net = feedforwardnet(NN_architecture{l}, 'trainscg');
    net = train(net, X_mnist_pca',  Y_mnist', [], [], [], 'useParallel');
    
    % Test Performance:
    X_mnist_test_pca = X_mnist_test * pca(X_mnist_test);
    X_mnist_test_pca = X_mnist_test_pca(:, 1:PC_dim);
    [~, NN_class(:,j)] = max(net(X_mnist_pca')', [], 2);
    [~, NN_class_test] = max(net(X_mnist_test_pca')', [], 2);
    
    NN_PCA_mnist_er(j)      = sum(NN_class(:,j) ~= (y_mnist+1))/train_size;
    NN_PCA_mnist_er_test(j) = sum(NN_class_test ~= (y_mnist_test+1))/test_size;
    
    tmp_PCA_disp(j) =  PC_dim;
    
    j = j + 1;
    
    end
    
    tmp_Archi_disp = [ tmp_Archi_disp; NN_architecture ];

end

figure(figj)
figj = figj+1;
for l = 1:PN

    misclass = zeros(10, 10);
    for k = 1:10
        for j = 1:10
            tmp_class = NN_class(:,l);
            misclass(k, j) = ...
                sum(tmp_class((y_mnist+1) == k) == j)/...
                                (train_size/10);
        end
    end
    
    subplot(floor(PN/6), PN/floor(PN/6), l);
    imagesc(misclass)
    pbaspect([2 2 1])

end
% title('CNN classifier - Fashion - Misclassification matrix')

results_PCA_ANN_mnist = ...
    table( tmp_PCA_disp, tmp_Archi_disp,  ...
           NN_PCA_mnist_er, NN_PCA_mnist_er_test, ...
            'VariableNames',{'PCA_Size','NN_Architecture',...
                             'Error_int','Error_test'});
        
disp(' ')
disp(' ----------------------------- ')
disp('       MNIST - PCA + ANN       ')
disp(' ----------------------------- ')
disp(' ')
disp(results_PCA_ANN_mnist)
disp(' ')

%% notMNIST

FFT_2D_FLAG = 0; % IF = 1, use the 2D Fourier Transform on the data;

NN_class = zeros(train_size,PN);

NN_PCA_notMnist_er      = zeros(PN,1); 
NN_PCA_notMnist_er_test = zeros(PN,1); 

tmp_PCA_disp = zeros(PN,1);
tmp_Archi_disp = {};

% Create the 2D Fourier Transform of all images:
if (FFT_2D_FLAG == 1)
    X_notMnist_ft2      = zeros(size(X_notMnist));
    X_notMnist_test_ft2 = zeros(size(X_notMnist_test));
    for j = 1:train_size
        X_notMnist_ft2(j,:) = reshape( ...
                        abs(fftshift(fft2(reshape(X_notMnist(j,:), 28, 28)))), ...
                        1, 28*28);
    end
    for j = 1:test_size
        X_notMnist_test_ft2(j,:) = reshape( ...
                        abs(fftshift(fft2(reshape(X_notMnist_test(j,:), 28, 28)))), ...
                        1, 28*28);
    end
end

j = 1;

for PC_dim = PCA_sizes
    
    for l = 1:length(NN_architecture)

    % PCA of the MNIST data:
%     X_notMnist_pca = X_notMnist * pca(X_notMnist);
%     X_notMnist_pca = X_notMnist_pca(:, 1:PC_dim);
    if (FFT_2D_FLAG == 0)
        % PCA of the Fashion data:
        X_notMnist_pca = X_notMnist * pca(X_notMnist);
        X_notMnist_test_pca = X_notMnist_test * pca(X_notMnist_test);
    else
        % PCA for the 2D FFT data:
        X_notMnist_pca = X_notMnist_ft2 * pca(X_notMnist_ft2);
        X_notMnist_test_pca = X_notMnist_test_ft2 * pca(X_notMnist_test_ft2);
    end
    X_notMnist_pca = X_notMnist_pca(:, 1:PC_dim);
    X_notMnist_test_pca = X_notMnist_test_pca(:, 1:PC_dim);

    % Neural network:
    net = feedforwardnet(NN_architecture{l}, 'trainscg');
    net = train(net, X_notMnist_pca',  Y_notMnist', [], [], [], 'useParallel');
    
    % Test Performance:
%     X_notMnist_test_pca = X_notMnist_test * pca(X_notMnist_test);
%     X_notMnist_test_pca = X_notMnist_test_pca(:, 1:PC_dim);
    [~, NN_class(:,j)] = max(net(X_notMnist_pca')', [], 2);
    [~, NN_class_test] = max(net(X_notMnist_test_pca')', [], 2);
    
    NN_PCA_notMnist_er(j)      = sum(NN_class(:,j) ~= (y_notMnist+1))/train_size;
    NN_PCA_notMnist_er_test(j) = sum(NN_class_test ~= (y_notMnist_test+1))/test_size;
    
    tmp_PCA_disp(j) =  PC_dim;
    
    j = j + 1;
    
    end
    
    tmp_Archi_disp = [ tmp_Archi_disp; NN_architecture ];

end

figure(figj)
figj = figj+1;
for l = 1:PN

    misclass = zeros(10, 10);
    for k = 1:10
        for j = 1:10
            tmp_class = NN_class(:,l);
            misclass(k, j) = ...
                sum(tmp_class((y_notMnist+1) == k) == j)/...
                                (train_size/10);
        end
    end
    
    subplot(floor(PN/6), PN/floor(PN/6), l);
    imagesc(misclass)
    pbaspect([2 2 1])

end
% title('CNN classifier - Fashion - Misclassification matrix')


results_PCA_ANN_notMnist = ...
    table( tmp_PCA_disp, tmp_Archi_disp,  ...
           NN_PCA_notMnist_er, NN_PCA_notMnist_er_test, ...
            'VariableNames',{'PCA_Size','NN_Architecture',...
                             'Error_int','Error_test'});
        
disp(' ')
disp(' ----------------------------- ')
disp('      notMNIST - PCA + ANN     ')
disp(['        [ 2D_FFT = ',num2str(FFT_2D_FLAG),' ]         '])
disp(' ----------------------------- ')
disp(' ')
disp(results_PCA_ANN_notMnist)
disp(' ')

%% FASHION

FFT_2D_FLAG = 0; % IF = 1, use the 2D Fourier Transform on the data;

NN_class = zeros(train_size,PN);

NN_PCA_fashion_er      = zeros(PN,1);
NN_PCA_fashion_er_test = zeros(PN,1);

tmp_PCA_disp = zeros(PN,1);
tmp_Archi_disp = {};

% Create the 2D Fourier Transform of all images:
if (FFT_2D_FLAG == 1)
    X_fashion_ft2      = zeros(size(X_fashion));
    X_fashion_test_ft2 = zeros(size(X_fashion_test));
    for j = 1:train_size
        X_fashion_ft2(j,:) = reshape( ...
                        abs(fftshift(fft2(reshape(X_fashion(j,:), 28, 28)))), ...
                        1, 28*28);
    end
    for j = 1:test_size
        X_fashion_test_ft2(j,:) = reshape( ...
                        abs(fftshift(fft2(reshape(X_fashion_test(j,:), 28, 28)))), ...
                        1, 28*28);
    end
end

j = 1;

for PC_dim = PCA_sizes
    
    for l = 1:length(NN_architecture)

    if (FFT_2D_FLAG == 0)
        % PCA of the Fashion data:
        X_fashion_pca = X_fashion * pca(X_fashion);
        X_fashion_test_pca = X_fashion_test * pca(X_fashion_test);
    else
        % PCA for the 2D FFT data:
        X_fashion_pca = X_fashion_ft2 * pca(X_fashion_ft2);
        X_fashion_test_pca = X_fashion_test_ft2 * pca(X_fashion_test_ft2);
    end
    X_fashion_pca = X_fashion_pca(:, 1:PC_dim);
    X_fashion_test_pca = X_fashion_test_pca(:, 1:PC_dim);

    % Neural network:
    net = feedforwardnet(NN_architecture{l}, 'trainscg');
    net = train(net, X_fashion_pca',  Y_fashion', [], [], [], 'useParallel');
    
    % Test Performance:
    [~, NN_class(:,j)] = max(net(X_fashion_pca')', [], 2);
    [~, NN_class_test] = max(net(X_fashion_test_pca')', [], 2);
    
    NN_PCA_fashion_er(j) = sum(NN_class(:,j) ~= (y_fashion+1))/n_mnist;
    NN_PCA_fashion_er_test(j) = sum(NN_class_test ~= (y_fashion_test+1))/test_size;
    
    tmp_PCA_disp(j) =  PC_dim;
    
    j = j + 1;
    
    end
    
    tmp_Archi_disp = [ tmp_Archi_disp; NN_architecture ];

end

figure(figj)
figj = figj+1;
for l = 1:PN

    misclass = zeros(10, 10);
    for k = 1:10
        for j = 1:10
            tmp_class = NN_class(:,l);
            misclass(k, j) = ...
                sum(tmp_class((y_fashion+1) == k) == j)/...
                                (train_size/10);
        end
    end
    
    subplot(floor(PN/6), PN/floor(PN/6), l);
    imagesc(misclass)
    pbaspect([2 2 1])

end
% title('CNN classifier - Fashion - Misclassification matrix')


results_PCA_ANN_fashion = ...
    table( tmp_PCA_disp, tmp_Archi_disp,  ...
           NN_PCA_fashion_er, NN_PCA_fashion_er_test, ...
            'VariableNames',{'PCA_Size','NN_Architecture',...
                             'Error_int','Error_test'});
        
disp(' ')
disp(' ----------------------------- ')
disp('   Fashion MNIST - PCA + ANN   ')
disp(['        [ 2D_FFT = ',num2str(FFT_2D_FLAG),' ]         '])
disp(' ----------------------------- ')
disp(' ')
disp(results_PCA_ANN_fashion)
disp(' ')

%% Convolutional Neural Network

rng('default')

X_mnist_datastore    = zeros(28, 28, 1, n_mnist);
X_notMnist_datastore = zeros(28, 28, 1, n_mnist);
X_fashion_datastore  = zeros(28, 28, 1, n_mnist);

X_mnist_test_datastore    = zeros(28, 28, 1, test_size);
X_notMnist_test_datastore = zeros(28, 28, 1, test_size);
X_fashion_test_datastore  = zeros(28, 28, 1, test_size);

validationSize = 1000;

options = @(val_data) trainingOptions('sgdm',...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.2, ...
    'LearnRateDropPeriod', 5, ...
    'MaxEpochs', 20, ...
    'MiniBatchSize', 64, ...
    'L2Regularization',0.0005, ...
    'ValidationData', val_data, ...
    'ValidationPatience', 10 ...
    );
%     'Plots','training-progress');

CNN_layers = { [ ...
                imageInputLayer([28, 28, 1]);
                convolution2dLayer(5, 4, 'Padding', 2);
                reluLayer;
                fullyConnectedLayer(10);
                softmaxLayer();
                classificationLayer();
              ]; 
              [ ...
                imageInputLayer([28, 28, 1]);
                convolution2dLayer(5, 8, 'Padding', 2);
                reluLayer;
                fullyConnectedLayer(100);
                reluLayer;
                fullyConnectedLayer(10);
                softmaxLayer();
                classificationLayer();
              ];
% };
% CNN_layers = {
              [ ...
                imageInputLayer([28, 28, 1]);
                convolution2dLayer(3, 4, 'Padding', 1);
                batchNormalizationLayer
                reluLayer;
                fullyConnectedLayer(10);
                softmaxLayer();
                classificationLayer();
              ];
% };
% CNN_layers = {
              [ ...
                imageInputLayer([28, 28, 1]);
                convolution2dLayer(3, 8, 'Padding', 1);
                batchNormalizationLayer
                reluLayer;
%                 averagePooling2dLayer(2,'Stride',2)
%                 maxPooling2dLayer(2,'Stride',2)
                convolution2dLayer(5, 16, 'Padding', 2);
                batchNormalizationLayer
                reluLayer;
                fullyConnectedLayer(10);
                softmaxLayer();
                classificationLayer();
              ];
};

%% MNIST
disp(' ')
disp(' Training:          MNIST - CNN          ')
disp(' ')

for j = 1:n_mnist
    X_mnist_datastore(:,:,:,j) = reshape(X_mnist(j,:), 28, 28);
end
for j = 1:test_size
    X_mnist_test_datastore(:,:,:,j) = reshape(X_mnist_test(j,:), 28, 28);
end

CNN_mnist_error      = zeros(length(CNN_layers),1);
CNN_mnist_error_test = zeros(length(CNN_layers),1);
CNN_predict_mnist      = zeros(train_size-validationSize,length(CNN_layers));
CNN_predict_mnist_test = zeros(test_size,length(CNN_layers));

% Generate validation data:
trainImages = X_mnist_datastore;
trainLabels = categorical(y_mnist+1);

idx = randperm(train_size, 1000);
valImages = X_mnist_datastore(:,:,:,idx);
trainImages(:,:,:,idx) = [];
valLabels = trainLabels(idx);
trainLabels(idx) = [];

for l = 1:length(CNN_layers)
    
    CNN_net_mnist = ...
        trainNetwork(trainImages, trainLabels, ...
                        CNN_layers{l}, ...
                        options({valImages,valLabels}));
%     disp(' #1 ')           
    CNN_predict_mnist(:,l) = ...
        classify(CNN_net_mnist, trainImages);
    CNN_predict_mnist_test(:,l) = ...
        classify(CNN_net_mnist, X_mnist_test_datastore);
%     disp(' #2 ')
    CNN_mnist_error(l) = ...
                sum(CNN_predict_mnist(:,l) ~= double(trainLabels))/(train_size-validationSize);
    CNN_mnist_error_test(l) = ...
                sum(CNN_predict_mnist_test(:,l) ~= (y_mnist_test+1))/test_size;
%     disp(' #3 ')
    
end

% Plot

figure(5)
for l = 1:length(CNN_layers)

    misclass = zeros(10, 10);
    for k = 1:10
        for j = 1:10
            tmp_class = CNN_predict_mnist(:,l);
            misclass(k, j) = ...
                sum(tmp_class(double(trainLabels) == k) == j)/...
                                ((train_size-validationSize)/10);
        end
    end
    
    subplot(floor(length(CNN_layers)/2), ...
                    length(CNN_layers)/floor(length(CNN_layers)/2), l);
    imagesc(misclass)
    pbaspect([2 2 1])

end
% title('CNN classifier - MNIST - Misclassification matrix')

results_CNN_mnist = ...
    table([1:length(CNN_layers)]', CNN_mnist_error, CNN_mnist_error_test, ...
            'VariableNames',{'CNN_Architecture','Error_int','Error_test'});
        
disp(' ')
disp(' ----------------------------- ')
disp('          MNIST - CNN          ')
disp(' ----------------------------- ')
disp(' ')
disp(results_CNN_mnist)
disp(' ')

% clear trainImages valImages trainLabels valLabels

%% notMNIST
disp(' ')
disp(' Training:        notMNIST - CNN        ')
disp(' ')

for j = 1:n_notMnist
    X_notMnist_datastore(:,:,:,j) = reshape(X_notMnist(j,:), 28, 28);
end
for j = 1:test_size
    X_notMnist_test_datastore(:,:,:,j) = reshape(X_notMnist_test(j,:), 28, 28);
end

CNN_notMnist_error      = zeros(length(CNN_layers),1);
CNN_notMnist_error_test = zeros(length(CNN_layers),1);
CNN_predict_notMnist      = zeros(train_size-validationSize,length(CNN_layers));
CNN_predict_notMnist_test = zeros(test_size,length(CNN_layers));

% Generate validation data:
trainImages = X_notMnist_datastore;
trainLabels = categorical(y_notMnist+1);

idx = randperm(train_size, 1000);
valImages = X_mnist_datastore(:,:,:,idx);
trainImages(:,:,:,idx) = [];
valLabels = trainLabels(idx);
trainLabels(idx) = [];

for l = 1:length(CNN_layers)
    
    CNN_net_notMnist = ...
        trainNetwork(trainImages, trainLabels, ...
                        CNN_layers{l}, ...
                        options({valImages,valLabels}));
%     disp(' #1 ')           
    CNN_predict_notMnist(:,l) = ...
        classify(CNN_net_notMnist, trainImages);
    CNN_predict_notMnist_test(:,l) = ...
        classify(CNN_net_notMnist, X_notMnist_test_datastore);
%     disp(' #2 ')
    CNN_notMnist_error(l) = ...
                sum(CNN_predict_notMnist(:,l) ~= double(trainLabels))/(train_size-validationSize);
    CNN_notMnist_error_test(l) = ...
                sum(CNN_predict_notMnist_test(:,l) ~= (y_notMnist_test+1))/test_size;
%     disp(' #3 ')
    
end

% Plot

figure(5)
for l = 1:length(CNN_layers)

    misclass = zeros(10, 10);
    for k = 1:10
        for j = 1:10
            tmp_class = CNN_predict_notMnist(:,l);
            misclass(k, j) = ...
                sum(tmp_class(double(trainLabels) == k) == j)/...
                                ((train_size-validationSize)/10);
        end
    end
    
    subplot(floor(length(CNN_layers)/2), ...
                    length(CNN_layers)/floor(length(CNN_layers)/2), l);
    imagesc(misclass)
    pbaspect([2 2 1])

end
% title('CNN classifier - notMNIST - Misclassification matrix')

results_CNN_notMnist = ...
    table([1:length(CNN_layers)]', CNN_notMnist_error, CNN_notMnist_error_test, ...
            'VariableNames',{'CNN_Architecture','Error_int','Error_test'});

disp(' ')
disp(' ----------------------------- ')
disp('         notMNIST - CNN        ')
disp(' ----------------------------- ')
disp(' ')
disp(results_CNN_notMnist)
disp(' ')

% clear trainImages valImages trainLabels valLabels

%% FASHION
disp(' ')
disp(' Training:     Fashion MNIST - CNN      ')
disp(' ')

for j = 1:train_size
    X_fashion_datastore(:,:,:,j) = reshape(X_fashion(j,:), 28, 28);
%     X_fashion_datastore(:,:,:,j) = ...
%                 abs(fftshift(fft2(reshape(X_fashion(j,:), 28, 28))));
end
for j = 1:test_size
    X_fashion_test_datastore(:,:,:,j) = reshape(X_fashion_test(j,:), 28, 28);
%     X_fashion_test_datastore(:,:,:,j) = ...
%                 abs(fftshift(fft2(reshape(X_fashion_test(j,:), 28, 28))));  
end
  
CNN_fashion_error      = zeros(length(CNN_layers),1);
CNN_fashion_error_test = zeros(length(CNN_layers),1);
CNN_predict_fashion      = zeros(train_size-validationSize,length(CNN_layers));
CNN_predict_fashion_test = zeros(test_size,length(CNN_layers));

% Generate validation data:
trainImages = X_fashion_datastore;
trainLabels = categorical(y_fashion+1);

idx = randperm(train_size, 1000);
valImages = X_mnist_datastore(:,:,:,idx);
trainImages(:,:,:,idx) = [];
valLabels = trainLabels(idx);
trainLabels(idx) = [];

for l = 1:length(CNN_layers)
    
    CNN_net_fashion = ...
        trainNetwork(trainImages, trainLabels, ...
                        CNN_layers{l}, ...
                        options({valImages,valLabels}));
%     disp(' #1 ')           
    CNN_predict_fashion(:,l) = ...
        classify(CNN_net_fashion, trainImages);
    CNN_predict_fashion_test(:,l) = ...
        classify(CNN_net_fashion, X_fashion_test_datastore);
%     disp(' #2 ')
    CNN_fashion_error(l) = ...
                sum(CNN_predict_fashion(:,l) ~= double(trainLabels))/(train_size-validationSize);
    CNN_fashion_error_test(l) = ...
                sum(CNN_predict_fashion_test(:,l) ~= (y_fashion_test+1))/test_size;
%     disp(' #3 ')
    
end

% Plot

figure(5)
for l = 1:length(CNN_layers)

    misclass = zeros(10, 10);
    for k = 1:10
        for j = 1:10
            tmp_class = CNN_predict_fashion(:,l);
            misclass(k, j) = ...
                sum(tmp_class(double(trainLabels) == k) == j)/...
                                ((train_size-validationSize)/10);
        end
    end
    
    subplot(floor(length(CNN_layers)/2), ...
                    length(CNN_layers)/floor(length(CNN_layers)/2), l);
    imagesc(misclass)
    pbaspect([2 2 1])

end
% title('CNN classifier - Fashion - Misclassification matrix')

results_CNN_fashion = ...
    table([1:length(CNN_layers)]',  CNN_fashion_error, CNN_fashion_error_test, ...
            'VariableNames',{'CNN_Architecture','Error_int','Error_test'});
        
disp(' ')
disp(' ----------------------------- ')
disp('      Fashion MNIST - CNN      ')
disp(' ----------------------------- ')
disp(' ')
disp(results_CNN_fashion)
disp(' ')

% clear trainImages valImages trainLabels valLabels
