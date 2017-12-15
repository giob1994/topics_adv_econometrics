function [ logL ] = getLogisticLikelihood( x, y, beta )

logL = 0;

for i = 1:length(x)

    hbeta = exp([1, x(i)]*beta(:))./(1+exp([1, x(i)]*beta(:)));

    logL = logL + ( y(i).*log(hbeta) + (1-y(i)).*log(1-hbeta) );
    
end

end

