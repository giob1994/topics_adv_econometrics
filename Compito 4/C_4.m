% -------------------------------------------
%
%   Compito 4 - TAE
%
% -------------------------------------------

clear

% Normal PDF for sample:
% x, y, b1, b2, se
loglike = @(z) log(1./(2*pi*z(:,5)) .* ...
                exp(-(z(:,2) - z(:,3) - z(:,1).^z(:,4)).^2 ./ (2*z(:,5))));
            
syms x y b1 b2 se
logl(x,y,b1,b2,se) = log(1./(2*pi*se) * exp(-(y - b1 - x^b2)^2 / (2*se)));

pd_b1_1 = diff(logl, b1);
pd_b1_2 = diff(pd_b1_1, b1);
pd_b2_1 = diff(logl, b2);
pd_b2_2 = diff(pd_b2_1, b2);
pd_se_1 = diff(logl, se);
pd_se_2 = diff(pd_se_1, se);
pd_b1se = diff(pd_b1_1, se);
pd_b2se = diff(pd_b2_1, se);
pd_b1b2 = diff(pd_b1_1, b2);

% Parameters:

beta1 = 1;
beta2 = 1/10;
sigma_e = 0.1;

n = 1000;

% Sample generation:

X = [ones(n, 1), repmat(transpose(1:10), 100, 1)];
                
%% Newton-Rapson:

for j = 1:1

    Y = transpose([beta1, beta2, 1] * [X, sqrt(sigma_e).*randn(n,1)]');
    
    theta0 = [0.5, 0.2, 0.3];
    
    diff = 1;
    it = 0;
    
    while (diff > 10^-12 && it < 5)
    
        D1 = zeros(1,3);
        H = zeros(3);
        
%         D1sym = zeros(1,3);
%         Hsym = zeros(3);

        tic
        for k = 1:100
            
            % Numerical:
            D1_ = gradest(loglike, [X(k,2), Y(k), theta0]);
            D1 = D1 + D1_(3:end);

            H_ = hessian(loglike, [X(k,2), Y(k), theta0]);
            H = H + H_(3:end,3:end);
            
            % Symbolic:
%             D1sym = D1sym + ...
%                     double([pd_b1_1(X(k,2), Y(k), theta0(1), theta0(2), theta0(3)), ...
%                             pd_b2_1(X(k,2), Y(k), theta0(1), theta0(2), theta0(3)), ...
%                             pd_se_1(X(k,2), Y(k), theta0(1), theta0(2), theta0(3))]);
%                         
%             Hsym = Hsym + ...
%                    double([pd_b1_2(X(k,2), Y(k), theta0(1), theta0(2), theta0(3)), ...
%                             pd_b1b2(X(k,2), Y(k), theta0(1), theta0(2), theta0(3)), ...
%                             pd_b1se(X(k,2), Y(k), theta0(1), theta0(2), theta0(3)); 
%                             pd_b1b2(X(k,2), Y(k), theta0(1), theta0(2), theta0(3)), ...
%                             pd_b2_2(X(k,2), Y(k), theta0(1), theta0(2), theta0(3)), ...
%                             pd_b2se(X(k,2), Y(k), theta0(1), theta0(2), theta0(3));
%                             pd_b1se(X(k,2), Y(k), theta0(1), theta0(2), theta0(3)), ...
%                             pd_b2se(X(k,2), Y(k), theta0(1), theta0(2), theta0(3)), ...
%                             pd_se_2(X(k,2), Y(k), theta0(1), theta0(2), theta0(3));]);
        end
        toc

        theta = transpose(theta0(:) - (H \ D1(:)));
%         thetasym = transpose(theta0(:) - (Hsym \ D1sym(:)));
        
        diff = norm(theta - theta0);
        theta0 = theta;
        
        it = it + 1;
       
    end
    
end