% -------------------------------------------
%
%   Compito 6 - TAE
%
% -------------------------------------------

clear

% Normal OLS log-likelihood for sample:
%  z = [ x, y, beta1, sigma ]
% loglike = @(z) -0.5*log(z(4)^2) - (1/(2*z(4)^2)) * (z(2) - z(1)*z(3))^2;
% 
% syms x y b s
% logl(x,y,b,s) = -0.5*log(s) - 1/(2*s) * (y - x*b)^2;
% 
% pd_b_1 = diff(logl, b);
% pd_b_2 = diff(pd_b_1, b);
% pd_s_1 = diff(logl, s);
% pd_s_2 = diff(pd_s_1, s);
% pd_bs = diff(pd_b_1, s);

% Set up the coefficients:

beta1 = 1;
sigma = 1;

n = 1000;

MC_size = 1000;

X = repmat(transpose(1:10), 100, 1);
X(500) = 10^(3);
X(1000) = 10^(3);

OLS_beta1 = zeros(MC_size, 1);
stand_t_ratios = zeros(MC_size, 2);
robust_t_ratios = zeros(MC_size, 2);

for j = 1:MC_size
    
     Y = transpose([beta1, 1] * [X, sqrt(sigma).*randn(n,1)]');
     
     % Approximate MLE with OLS estimates:
     OLS_beta1(j, :) = (X'*X) \ (X'*Y);
     
     res = Y - OLS_beta1(j, :)*X;
     OLS_sigma = (res'*res) / (n - 1);
     
     % Log-likelihood Gradient and Hessian estimation: 
     Gsym = [ (X'*res)/OLS_sigma; -n/(2*OLS_sigma)+(res'*res)/(2*sigma^2)];
     Hsym = - [ -(X'*X)/OLS_sigma, ...  
                    (X'*res)/OLS_sigma^2;
                (X'*res)/OLS_sigma^2, ...
                    n/(2*OLS_sigma^2) - (res'*res)/(sigma^3)];
   
    % Standard variance:            
    stand_t_ratios(j, :) = ...
        ([OLS_beta1(j, :), OLS_sigma]  - [beta1, sigma]) ./ ... 
                   [sqrt(Hsym(1,1)), sqrt(Hsym(2,2))];
    
    % Robust variance:
    OLS_robust_var = pinv(Hsym) * (Gsym * Gsym') * pinv(Hsym);
    robust_t_ratios(j, :) = ...
        ([OLS_beta1(j, :), OLS_sigma]  - [beta1, sigma]) ./ ... 
                   [sqrt(OLS_robust_var(1,1)), sqrt(OLS_robust_var(2,2))];
     
end 

%% Variance and t-ratios with STANDARD method:

stand_mean_t_b1 = mean(stand_t_ratios(:,1));
stand_var_t_b1  = var(stand_t_ratios(:,1));
stand_mean_t_sigma = mean(stand_t_ratios(:,2));
stand_var_t_sigma  = var(stand_t_ratios(:,2));

%% Variance and t-ratios with ROBUST method:        

rob_mean_t_b1 = mean(robust_t_ratios(:,1));
rob_var_t_b1  = var(robust_t_ratios(:,1));
rob_mean_t_sigma = mean(robust_t_ratios(:,2));
rob_var_t_sigma  = var(robust_t_ratios(:,2));
