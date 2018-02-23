function [J, grad1, grad2] = nnminuslogLikelihood(nn_params, input_layer_size, ...
                            hidden_layer_size, num_labels, X, y, lambda)
                        
% Input format:
%
%   X  -  every ROW is an observation;
%   y  -  every ROW is a classification;
%
%
                        
%% Function for backpropagation

sig     = @(z) 1./(1 + exp(-z));
sig_d1  = @(z) sig(z) .* (1-sig(z));

N = max(size(X));

hidden_params = reshape(nn_params(1:input_layer_size*(hidden_layer_size-1)), ...
                        hidden_layer_size-1, input_layer_size);
output_params = reshape(nn_params((input_layer_size*(hidden_layer_size-1) + 1):end), ...
                        num_labels, hidden_layer_size);

Y = zeros(N, num_labels);
for j = 1:N
    Y(j, y(j)) = 1;
end
                    
%% (1) Compute likelihood:

%
%   A_(l)  -  every COLUMN is the filtering on 1 obs. by the layer (l)
%

% A1 = sig(hidden_params * [ones(1, size(X,1)); X']);
% A2 = sig(output_params * [ones(1, size(A1',1)); A1]);
% 
% J = - 1/max(size(X)) * sum(sum(  Y    .* log(A2') + ...
%                                 (1-Y) .* log(1-A2') ));

A1 = [ones(size(X,1), 1), X];
A2 = sig([ones(size(X,1), 1), A1 * hidden_params']);
A3 = sig(A2 * output_params');

% Compute the log-likelihood:
J = - 1/N * sum(sum( Y .* log(A3) + (1-Y) .* log(1-A3) ));
                            
%% (2) Compute the gradient with the backpropagation algorithm:

delta_3 = (A3 - Y)';   % every row is an observation (i)
delta_2 = (output_params' * delta_3) .* ...
                    sig_d1([ones(size(X,1), 1), A1 * hidden_params']');

% DELTA_1 = d_2 * A1;
% DELTA_0 = d_1 * X;

grad2 = zeros(size(output_params));
grad1 = zeros(size(hidden_params));

for i = 1:N

%     delta_3 = A3 - repmat(Y(i,:), N, 1);
%     delta_2 = (output_params' * d_2) .* sig_d1(A1(i));

    Delta_2 = delta_3(:,i) * A2(i,:);
    Delta_1 = delta_2(2:end,i) * A1(i,:);

    grad2 = grad2 + Delta_2 + ...
                lambda * [zeros(num_labels, 1), output_params(:,2:end)];
    grad1 = grad1 + Delta_1 + ...
                lambda * [zeros(hidden_layer_size-1, 1), hidden_params(:,2:end)];
    
%     if ( k == 0 )
%         grad1(j,k) = 1/N * sum(Delta_1(j,k));
%     else
%         grad1(j,k) = 1/N * sum(Delta_1(j,k) + lambda);
%     end
% 
%     if ( k == 0 )
%         grad0(j,k) = 1/N * sum()
%     else
% 
%     end
% 

end

grad2 = grad2/N;
grad1 = grad1/N;

end