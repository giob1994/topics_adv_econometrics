function [ log_like ] = getMultiGaussianMixLogLikelihood( X, theta )
% Returns the log-likelihood of a Multivariate Gaussian Mixture
%
% Note: theta must be a struct containing { mu1, mu2, sigma1, sigma2,
%       w1, w2 }

if (isstruct(theta) == 0)
    
    n = theta(1);
    
    if (length(theta) == 2*n^3-1)
    
        mu = transpose(reshape(theta(2:1+n^2),n,n));
        sigma = reshape(theta(2+n^2:1+n^2+n^3),n,n,n);
        w = theta(end-n:end);
        
        theta = struct('n', n,...
               'mu', mu,...
               'sigma', sigma,...
               'w', w);
    
    else
        
        error('[!] theta is incorrectly specified [!]')
     
    end
    
end

N = theta.n;

like_ = zeros(size(X,1),1);

for i=1:N
    
    pdf_tmp = theta.w(i)*(mvnpdf(X, theta.mu(i,:), theta.sigma(:,:,i)));
    
    like_ = like_ + pdf_tmp;
    
end

log_like = sum(log(like_));

end

