function [ log_like ] = getBiGaussianMixLogLikelihood( X, theta, dims )
% Returns the log-likelihood of a Multivariate Gaussian Mixture

% 'theta' is assumed to have the following structure:
%
% [  w0,    w1,   [mu0],   [mu1],   [sigma0],     [sigma1]    ]
%   1x1    1x1    1xdims   1xdims   1xdims^2-1    1xdims^2-1
%

theta = theta(:)';

w0      = theta(1);
w1      = theta(2);
mu0     = theta(3:2+dims);
mu1     = theta(3+dims:2+2*dims);
Sigma0  = [ theta(3+2*dims), theta(4+2*dims);
            theta(4+2*dims), theta(5+2*dims); ];
Sigma1  = [ theta(3+2*dims+dims^2-1), theta(4+2*dims+dims^2-1);
            theta(4+2*dims+dims^2-1), theta(5+2*dims+dims^2-1); ];

[~,p0] = chol(Sigma0);
[~,p1] = chol(Sigma1);

if (p0 + p1 == 0)

    log_like = sum(log( w0*(mvnpdf(X, mu0, Sigma0)) + ...
                        w1*(mvnpdf(X, mu1, Sigma1)) ));
else
    
    log_like = -Inf;
    
end

end

