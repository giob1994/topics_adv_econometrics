function [J grad] = nnminuslogLikelihood(nn_params, input_layer_size, ...
                            hidden_layer_size, num_labels, X, y, lambda)
                        
% Function for backpropagation

sig     = @(z) 1./(1 + exp(-z));
sig_d1  = @(z) sig(z) .* (1-sig(z));

l = 0.01;

N = max(size(X));

hidden_params = reshape(nn_params(1:input_layer_size*hidden_layer_size), ...
                        hidden_layer_size, input_layer_size);
output_params = reshape(nn_params(input_layer_size*hidden_layer_size:end), ...
                        num_labels, hidden_layer_size);

Y = zeros(N, num_labels);
for j = 1:N
    Y(j,:) = (lambda == y);
end
                    
% (1) Compute likelihood:

A1 = sig(hidden_params * [ones(1, size(X,1)); X']);
A2 = sig(output_params * [ones(1, size(layer1',1)); A1]);

J = - 1/max(size(X)) * sum(sum(  Y    .* log(A2') + ...
                                (1-Y) .* log(1-A2') ));
                            
% (2) Compute the gradient with the backpropagation algorithm:

d_2 = A2 - Y;
d_1 = (hidden_params * d_2) .* sig_d1(A0);

Delta_1 = d_2 * A1;
Delta_0 = d_1 * X;

grad1 = NaN(hidden_layer_size, input_layer_size);
grad0 = NaN(num_labels, hidden_layer_size);

for j = 1:hidden_layer_size
    for k = 1:input_layer_size
        if ( k == 0 )
            grad1(j,k) = 1/N * sum()
        else
            
        end
    end
end

for j = 1:num_labels
    for k = 1:hidden_layer_size
        if ( k == 0 )
            grad0(j,k) = 1/N * sum()
        else
            
        end
    end
end


end