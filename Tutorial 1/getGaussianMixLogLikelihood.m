function [ log_like ] = getGaussianMixLogLikelihood( x, theta )
% Returns the log-likelihood of a Gaussian Mixture

tot_w = sum(floor(theta(2*length(theta)/3:end)))

if (mod(length(theta), 3) == 0 && tot_w == 1)
    
    log_like = 0;
 
    for i = 1:floor(length(theta)/3)
    
        pdf_tmp = makedist('Normal', theta(i), sqrt(theta(i+2)));
    
        log_like = log_like + sum(log(theta(i+4)*pdf(pdf_tmp, x)));
    
    end
    
else
   
    error('Vector "theta" is incorrectly specified');
    
end

end

