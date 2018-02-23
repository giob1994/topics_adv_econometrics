% -------------------------------------------
%
%   Compito 8 - TAE
%
% -------------------------------------------

clear


beta1 = 0;
beta2 = 0.1;
sigma = 0.5;

n = 1000;
MC_size = 100;

% Create X:
X = [ones(n,1), repmat(transpose(1:10), 100, 1)];

MLE_betas  = zeros(MC_size, 2);
MLE_sigmas = zeros(MC_size, 1);

stand_t_ratios = zeros(MC_size, 3);
robust_t_ratios = zeros(MC_size, 3);

for j = 1:MC_size
    
    Y = transpose([beta1, 1, 1] * ...
                    [ones(n,1), X(:,2).^beta2, sqrt(sigma).*randn(n,1)]');
                    
    % Approximate MLE with OLS estimates:
     llike = @(th)  n*0.5*log(th(3)^2) + ...
                 ((Y - X(:,1)*th(1) - X(:,2).^th(2))'*...
                           (Y - X(:,1)*th(1) - X(:,2).^th(2))) ./ (2*th(3)^2);
                       
     [M, ~, ~, ~, Gsym, Hsym] = fminunc(llike, [0.1, 0.2, 0.15]);
     MLE_betas(j, :) = M(1:2);
     MLE_sigmas(j) = M(3)^2;
     
     res = Y - (MLE_betas(j, :)*X')';
     
     % Log-likelihood Gradient and Hessian estimation: 
%      Gsym = [ (X(:,2)'*res)/MLE_sigmas(j); 
%               sum(res)/MLE_sigmas(j);
%               -n/(2*MLE_sigmas(j))+(res'*res)/(2*MLE_sigmas(j)^2)];
%      Hsym = - [ -(X(:,2)'*X(:,2))/MLE_sigmas(j), ...
%                 -sum(X(:,2))/MLE_sigmas(j), ...
%                 (X(:,2)'*res)/MLE_sigmas(j)^2;
%                 -sum(X(:,2))/MLE_sigmas(j), ...
%                 -n/MLE_sigmas(j), ...
%                 -sum(res)/MLE_sigmas(j)^2;
%                 (X(:,2)'*res)/MLE_sigmas(j)^2, ...
%                 -sum(res)/MLE_sigmas(j)^2, ...
%                 n/(2*MLE_sigmas(j)^2) - (res'*res)/(MLE_sigmas(j)^3)];
            
    % Standard variance:            
    stand_t_ratios(j, :) = ...
        ([MLE_betas(j, :), MLE_sigmas(j)]  - [beta1, beta2, sigma]) ./ ... 
                   [sqrt(Hsym(1,1)), sqrt(Hsym(2,2)), sqrt(Hsym(3,3))];
    
    % Robust variance:
    OLS_robust_var = pinv(Hsym) * (Gsym * Gsym') * pinv(Hsym);
    robust_t_ratios(j, :) = ...
        ([MLE_betas(j, :), MLE_sigmas(j)]  - [beta1, beta2, sigma]) ./ ... 
                   [sqrt(OLS_robust_var(1,1)), ...
                    sqrt(OLS_robust_var(2,2)), ...
                    sqrt(OLS_robust_var(3,3))];
     
end

% Monte Carlo means:
mc_m_b1     = mean(MLE_betas(j, 1));
mc_m_b2     = mean(MLE_betas(j, 2));
mc_m_sigma  = mean(MLE_sigmas);

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