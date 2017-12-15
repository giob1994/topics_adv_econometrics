function [ eta ] = ...
    getEtaAnemia2features(x_current, w1, mu_x_1, Sigma_x_1, ...
                                     w0, mu_x_0, Sigma_x_0)


pdf1 = mvnpdf(x_current, mu_x_1(:)', Sigma_x_1);
pdf0 = mvnpdf(x_current, mu_x_0(:)', Sigma_x_0);
                                 
eta = w1*pdf1 ./ (w1*pdf1 + w0*pdf0);


end

