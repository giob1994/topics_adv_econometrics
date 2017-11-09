function loglikeVal = getGaussianMixLogLikBicomp(x, theta)
% Function computes the 2-components'
% Gaussian Mixture Log-likelihood for the sample x with means mu1 = theta(1),
% mu2 = theta(2), variances sigma12 = theta(3), sigma22 = theta(4), and 
% weights of components w1 = theta(5), w2 = theta(6);

% Check that the sum of weights = 1
% if theta(5) + theta(6) ~= 1
%     error('Weights do not add up to 1')
% end
n = length(x);
loglikeVal = 0;
for i = 1:n
    loglikeVal = loglikeVal + log(theta(5) * (1 / sqrt(2 * pi * theta(3))) * ...
        exp(-(x(i) - theta(1)).^2/(2 * theta(3)))+...
        theta(6) * (1 / sqrt(2 * pi * theta(4))) * ...
        exp(-(x(i) - theta(2)).^2/(2 * theta(4))));
    % or
    %loglikeVal = loglikeVal + ...
    %    log(theta(5) * normpdf(theta(1), sqrt(theta(3))) +...
    %    theta(6) * normpdf(theta(2), sqrt(theta(4))));
end

