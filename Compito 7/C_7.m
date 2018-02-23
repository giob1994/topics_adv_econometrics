% -------------------------------------------
%
%   Compito 7 - TAE
%
% -------------------------------------------

clear


beta1 = 1;
beta2 = 100;
sigma = 0.1;

n = 1000;
MC_size = 1000;

% Create X:
X = [ones(n,1), repmat(transpose(1:10), 100, 1)];

OLS_betas  = zeros(MC_size, 2);
OLS_sigmas = zeros(MC_size, 1);

stand_t_ratios = zeros(MC_size, 3);
robust_t_ratios = zeros(MC_size, 3);

for j = 1:MC_size
    
    Y = transpose([beta1, beta2, 1] * ...
                    [ones(n,1), log(X(:,2)), sqrt(sigma).*randn(n,1)]');
                    
    % Approximate MLE with OLS estimates:
     OLS_betas(j, :) = (X'*X) \ (X'*Y);
     
     res = Y - (OLS_betas(j, :)*X')';
     OLS_sigmas(j) = (res'*res) / (n - 1);
     
     % Log-likelihood Gradient and Hessian estimation: 
     Gsym = [ (X(:,2)'*res)/OLS_sigmas(j); 
              sum(res)/OLS_sigmas(j);
              -n/(2*OLS_sigmas(j))+(res'*res)/(2*OLS_sigmas(j)^2)];
     Hsym = - [ -(X(:,2)'*X(:,2))/OLS_sigmas(j), ...
                -sum(X(:,2))/OLS_sigmas(j), ...
                (X(:,2)'*res)/OLS_sigmas(j)^2;
                -sum(X(:,2))/OLS_sigmas(j), ...
                -n/OLS_sigmas(j), ...
                -sum(res)/OLS_sigmas(j)^2;
                (X(:,2)'*res)/OLS_sigmas(j)^2, ...
                -sum(res)/OLS_sigmas(j)^2, ...
                n/(2*OLS_sigmas(j)^2) - (res'*res)/(OLS_sigmas(j)^3)];
            
    % Standard variance:            
    stand_t_ratios(j, :) = real( ...
        ([OLS_betas(j, :), OLS_sigmas(j)]  - [beta1, beta2, sigma]) ./ ... 
                   [sqrt(Hsym(1,1)), sqrt(Hsym(2,2)), sqrt(Hsym(3,3))] );
    
    % Robust variance:
    OLS_robust_var = pinv(Hsym) * (Gsym * Gsym') * pinv(Hsym);
    robust_t_ratios(j, :) = real( ...
        ([OLS_betas(j, :), OLS_sigmas(j)]  - [beta1, beta2, sigma]) ./ ... 
                   [sqrt(OLS_robust_var(1,1)), ...
                    sqrt(OLS_robust_var(2,2)), ...
                    sqrt(OLS_robust_var(3,3))] );
     
     
end

% Monte Carlo means:
mc_m_b1     = mean(OLS_betas(j, 1));
mc_m_b2     = mean(OLS_betas(j, 2));
mc_m_sigma  = mean(OLS_sigmas);

%% Variance and t-ratios with STANDARD method:

stand_m_t_b1 = mean(stand_t_ratios(:,1));
stand_var_t_b1  = var(stand_t_ratios(:,1));

stand_m_t_b2 = mean(robust_t_ratios(:,2));
stand_var_t_b2  = var(robust_t_ratios(:,2));

stand_m_t_sigma = mean(stand_t_ratios(:,3));
stand_var_t_sigma  = var(stand_t_ratios(:,3));

%% Variance and t-ratios with ROBUST method:        

rob_m_t_b1 = mean(robust_t_ratios(:,1));
rob_var_t_b1  = var(robust_t_ratios(:,1));

rob_m_t_b2 = mean(robust_t_ratios(:,2));
rob_var_t_b2  = var(robust_t_ratios(:,2));

rob_m_t_sigma = mean(robust_t_ratios(:,3));
rob_var_t_sigma  = var(robust_t_ratios(:,3));