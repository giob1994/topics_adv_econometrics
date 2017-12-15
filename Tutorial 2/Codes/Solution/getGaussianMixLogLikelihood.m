function loglikeVal = getGaussianMixLogLikelihood(x, theta)
% Function computes the
% Gaussian Mixture Log-likelihood for any number of components at a sample x 
% with the vector theta of concatenated means mu, variances sigma2, and weights w

% theta contains (i) k d-dimensional means: k*d, (ii) k covariances:
% k*d*(d+1)/2, (iii) k weights. Total: k * (d + 1 + d * (d + 1)/2)
[n, d] = size(x);
% Check that theta contains k means, k variances, and k weights
if mod(length(theta), (d + 1 + d * (d + 1)/2))
    error('theta vector length is incorrect');
end

k = length(theta)/(d + 1 + d * (d + 1)/2);
mu = zeros(k, d);
j = 0;
for i = 1:k
   mu(i, :) = theta(j + 1:j + d);
   j = j + d;
end
sigma2 = zeros(k, d * (d + 1)/2);
for i = 1:k
   sigma2(i, :) = theta(j + 1:j + d * (d + 1)/2);
   j = j + d * (d + 1)/2;
end
w = zeros(k, 1);
for i = 1:k
   w(i, :) = theta(j + 1);
   j = j + 1;
end
loglikeVal = 0;
for i = 1:n
    compLik = 0;
    for j = 1:k
        if min(eigs(math(sigma2(j, :)))) <= 0 || isnan(min(sigma2(j, :)))
            error('non PSD');
        end
        compLik = compLik + w(j, :) * mvnpdf(x(i, :), mu(j, :), (math(sigma2(j, :))));
    end
    loglikeVal = loglikeVal + log(compLik);
end

