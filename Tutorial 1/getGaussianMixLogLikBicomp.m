function [ log_like ] = getGaussianMixLogLikBicomp( x, theta )
% Returns the log-likelihood of a Gaussian Mixture

pdf_1 = makedist('Normal', theta(1), sqrt(theta(3)));
pdf_2 = makedist('Normal', theta(2), sqrt(theta(4)));

y1 = theta(5)*pdf(pdf_1, x);
y2 = theta(6)*pdf(pdf_2, x);

log_like = sum(log(y1+y2));

end

