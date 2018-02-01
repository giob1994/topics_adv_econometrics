% -------------------------------------------
%
%   Compito 2 - TAE
%
% -------------------------------------------

clear

% Set up the coefficients:

beta1 = 1;
beta2 = 1;

n = 1000;

MC_size = 1000;

X = [ones(n, 1), repmat(transpose(1:100), 10, 1)];

sigma_vec = [100; ones(n-1,1)];

%% MC replications:

OLS_betas = zeros(MC_size, 2);

robust_t_ratios = zeros(MC_size, 2);

for j = 1:MC_size
    
    tmp_Y = [beta1, beta2, 1] * [X, sqrt(sigma_vec).*randn(n,1)]';
    
    OLS_betas(j, :) = (X'*X) \ (X'*tmp_Y');
    
    % Robust variance:
    OLS_res = tmp_Y -  (OLS_betas(j, :) * X');
    S = zeros(2,2);
    for k = 1:n
        S = S + ( X(k,:)'*X(k,:)*OLS_res(k)^2 );
    end
    OLS_robust_var = ( (X'*X) \ S ) * (X'*X)^-1;
    robust_t_ratios(j, :) = (OLS_betas(j, :) - [beta1, beta2]) ./ ... 
                   [sqrt(OLS_robust_var(1,1)), sqrt(OLS_robust_var(2,2))];
    
end

%% Variance and t-ratios with STANDARD method:

OLS_stand_var = (X'*X)^-1;

stand_t_ratios = (OLS_betas - [beta1, beta2]) ./ ... 
                  [sqrt(OLS_stand_var(1,1)), sqrt(OLS_stand_var(2,2))];
                    
% jb_t_ratio_b1 = jbtest(stand_OLS_t_ratios(:,1));
% jb_t_ratio_b2 = jbtest(stand_OLS_t_ratios(:,2));

stand_mean_t_ratio_b1 = mean(stand_t_ratios(:,1));
stand_var_t_ratio_b1  = var(stand_t_ratios(:,1));
stand_mean_t_ratio_b2 = mean(stand_t_ratios(:,2));
stand_var_t_ratio_b2  = var(stand_t_ratios(:,2));

%% Variance and t-ratios with ROBUST method:

S = zeros(2,2);
for k = 1:n
    S = S + ( X(k,:)'*X(k,:)*sigma_vec(k)^2 );
end
OLS_robust_var_th = ( (X'*X) \ S ) * (X'*X)^-1;          

rob_mean_t_ratio_b1 = mean(robust_t_ratios(:,1));
rob_var_t_ratio_b1  = var(robust_t_ratios(:,1));
rob_mean_t_ratio_b2 = mean(robust_t_ratios(:,2));
rob_var_t_ratio_b2  = var(robust_t_ratios(:,2));