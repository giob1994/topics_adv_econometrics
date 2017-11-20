function [ theta_em ] = fitMultiMixedGaussianEM( X, theta, ep_ )
% Note: theta must be a struct containing { mu1, mu2, sigma1, sigma2,
%       w1, w2 }

if (isstruct(theta) == 0)
    
    n = theta(1);
    
    if (length(theta) == 2*n^3-1)
    
        mu = transpose(reshape(theta(2:1+n^2),n,n));
        sigma = reshape(theta(2+n^2:1+n^2+n^3),n,n,n);
        w = theta(end-n+1:end);
        
        theta = struct('n', n,...
               'mu', mu,...
               'sigma', sigma,...
               'w', w);
    
    else
        
        error('[!] theta is incorrectly specified [!]')
     
    end
    
end

N = theta.n;

theta_em = struct('n', n,...
               'mu', (mu+1),...
               'sigma', (sigma+1),...
               'w', w);
           
pdfs_ = zeros(N, size(X,1));
           
[r_mu, ~] = max(abs(theta.mu - theta_em.mu),[],3);
[r_sigma, ~] = max(abs(theta.sigma - theta_em.sigma),[],3);
[r_w, ~] = max(abs(theta.w - theta_em.w),[],3);
           
while ((norm(sum(abs(r_mu)))+norm(r_sigma)+norm(r_w)) > ep_)
    
%     (norm(sum(abs(r_mu)))+norm(r_sigma)+norm(r_w))
    
    % Move one step forward the variables:
    theta.mu = theta_em.mu;
    theta.sigma = theta_em.sigma;
    theta.w = theta_em.w;
    
    for i=1:N
    
        pdfs_(i,:) = theta.w(i)*transpose(mvnpdf(X, ...
                            theta.mu(i,:), theta.sigma(:,:,i)));
        
    end
    
    % [ E-step ]
    omega_ = pdfs_ ./ sum(pdfs_);
    
    % [ M-step ]
    theta_em.w = 1/size(X,1) * sum(omega_, 2);
        
    for i=1:N
        
        theta_em.mu(i,:) = sum(omega_(i,:).*X', 2) ./ sum(omega_(i,:), 2);
        X_mu = X - repmat(theta.mu(i,:), size(X,1), 1);
        
        tmp_sigma = zeros(N,N);
        for j=1:size(X,1)
            tmp_sigma = tmp_sigma + omega_(i,j) * X_mu(j,:)' * X_mu(j,:);
        end
        theta_em.sigma(:,:,i) = tmp_sigma ./ sum(omega_(i,:), 2);
                
    end
    
        
%     x1 = linspace(min(X(:,1))-1, ...
%                 max(X(:,1))+1, 50);
%     x2 = linspace(min(X(:,2))-1, ...
%                 max(X(:,2))+1, 50);
%     [X1, X2] = meshgrid(x1,x2);
%     y1 = theta_em.w(1)*reshape(mvnpdf([X1(:), X2(:)],...
%                 theta_em.mu(1,:), theta_em.sigma(:,:,1)), length(x1), length(x2));
%     y2 = theta_em.w(2)*reshape(mvnpdf([X1(:), X2(:)],...
%                 theta_em.mu(2,:), theta_em.sigma(:,:,2)), length(x1), length(x2));
%     y_mixed = y1 + y2;
%     figure
%     surf(x1, x2, y_mixed, 'FaceAlpha', 0.7);
%     shading interp
%     colormap jet
%     pbaspect([1 1 0.5])
%     xlim([-6, 6]);
%     ylim([-6, 6]);
%     view(3)
%     grid on
%     rotate3d on


    
    [r_mu, ~] = max(abs(theta.mu - theta_em.mu),[],3);
    [r_sigma, ~] = max(abs(theta.sigma - theta_em.sigma),[],3);
    [r_w, ~] = max(abs(theta.w - theta_em.w),[],3);
    
end

end

