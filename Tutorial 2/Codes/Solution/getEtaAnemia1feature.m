function eta_x = getEtaAnemia1feature(x_current, p, mu_x_1, sigma_x_1, mu_x_0, sigma_x_0)

    eta_x = p * normpdf(x_current, mu_x_1, sigma_x_1)/(p * ...
    normpdf(x_current, mu_x_1, sigma_x_1) + (1-p) * normpdf(x_current, mu_x_0, sigma_x_0));
end