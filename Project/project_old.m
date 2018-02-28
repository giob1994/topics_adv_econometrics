%
%   TAE - Final Project
%
%   Giovanni Ballarin, Stefanie Bertele
%

clear;
close all;


% Import MNIST dataset


X_mnist = csvread('digits_data.csv');
y_mnist = csvread('digits_labels.csv');

n = size(X_mnist, 1);
p = size(X_mnist, 2);
num_labels = 10;

Y_mnist = zeros(n, num_labels);
for j = 1:n
    Y_mnist(j, y_mnist(j)) = 1;
end

%% PCA + ANN:

PC_dim          = 400;
NN_hidden_size  = 10;

% PCA of the MNIST data:

% X_mnist_pca = PCA(X_mnist, PC_dim);
% % Normalize the result:
% X_mnist_pca = X_mnist_pca + abs(min(X_mnist_pca, [], 2));
% X_mnist_pca = X_mnist_pca ./ repmat(abs(max(X_mnist_pca, [], 2)), 1, PC_dim);

X_mnist_pca = X_mnist * pca(X_mnist);
X_mnist_pca = X_mnist_pca(:, 1:PC_dim);

% r = randperm(n);
% figure(1)
% displayData(X_mnist(r(1:100), :));
% figure(2)
% displayData(X_mnist_pca(r(1:100), :));

% Neural network:

% net = feedforwardnet(NN_hidden_size);
% net = train(net, X_mnist_pca',  Y_mnist', [], [], [], 'useParallel');
% [~, NN_class] = max(net(X_mnist_pca')', [], 2);

% lambda = 0.02;
% alpha = 0.001;
% 
% input_layer_size  = PC_dim + 1;
% hidden_layer_size = NN_hidden_size + 1;
% 
% params0_1 = -0.5 + 0.5 * rand( input_layer_size*(hidden_layer_size-1), 1 );
% params0_2 = -0.5 + 0.5 * rand( hidden_layer_size*num_labels, 1 );
              
% load('paramsNN.mat')
% params0_1 = Theta1(:);
% params0_1 = params0_1(1:input_layer_size*(hidden_layer_size-1));
% params0_2 = Theta2(:);
% params0_2 = params0_2(1:hidden_layer_size*num_labels);
% params0 = [params0_1; params0_2];

% diff = 1;
% it = 0;
% 
% while ( diff > 10^-4 && it < 1000 )
%     
%     [J, grad1, grad2] = nnminuslogLikelihood(params0, input_layer_size, ...
%                             hidden_layer_size, num_labels, ...
%                             X_mnist, y_mnist, lambda);
%                         
%     params1_1 = params0_1 - alpha * grad1(:);
%     params1_2 = params0_2 - alpha * grad2(:);
%     params1 = [params1_1; params1_2];
%     
%     diff = norm([grad1(:); grad2(:)]);
%     
%     params0 = params1;
%     
%     it = it+1;
%     
%     if mod(it, 1) == 0
%         disp(['  [training] diff =', num2str(diff)])
%     end
%     
% end
% 
% params1_1 = reshape(params1_1, hidden_layer_size-1, input_layer_size);
% params1_2 = reshape(params1_2, num_labels, hidden_layer_size);
% 
% sig = @(z) 1./(1 + exp(-z));
% 
% A1 = [ones(size(X_mnist_pca,1), 1), X_mnist_pca];
% A2 = sig([ones(size(X_mnist_pca,1), 1), A1 * params1_1']);
% A3 = sig(A2 * params1_2');
% 
% [~, NN_class] = max(A3, [], 2);

NN_emp_error = sum(NN_class ~= y_mnist)/n;

