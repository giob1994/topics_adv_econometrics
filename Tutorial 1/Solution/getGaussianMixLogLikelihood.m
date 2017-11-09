function loglikeVal = getGaussianMixLogLikelihood(x, theta)
% Function computes the
% Gaussian Mixture Log-likelihood for any number of components at a sample x 
% with the vector theta of concatenated means mu, variances sigma2, and weights w

% Check that theta contains k means, k variances, and k weights
if mod(length(theta), 3)
    error('theta vector length is incorrect');
end
k = length(theta)/3;
mu = theta(1:k);
sigma2 = (theta(k + 1:2 * k));
w = theta(2 * k + 1:3 * k);
n = length(x);
loglikeVal = 0;
for i = 1:n
    compLik = 0;
    for j = 1:k
        compLik = compLik + w(j) * normpdf(x(i), mu(j), sqrt(sigma2(j)));
    end
    loglikeVal = loglikeVal + log(compLik);
end

