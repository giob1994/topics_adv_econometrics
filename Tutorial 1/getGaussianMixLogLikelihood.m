function [ log_like ] = getGaussianMixLogLikelihood( x, theta )
% Returns the log-likelihood of a Gaussian Mixture

split = (1:2) * round(length(theta)/3);

mus = theta(1:split(1));
sigmas = theta((split(1)+1):split(2));
ws = theta((split(2)+1):end);

if (mod(length(theta), 3) == 0 && sum(ws) == 1)
    
    like_ = zeros(size(x));
 
    for i = 1:round(length(theta)/3)
    
        pdf_tmp = makedist('Normal', mus(i), sqrt(sigmas(i)));
    
        like_ = like_ + ws(i)*pdf(pdf_tmp, x);
    
    end
    
    log_like = sum(log(like_));
    
else
   
    error('Vector "theta" is incorrectly specified');
    
end

end

