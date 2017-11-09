function [ theta_em ] = fitMixedGaussianEMmono( x, theta, ep_ )

% Find starting points of different parameters:
split = (1:2) * round(length(theta)/3);

% Create aliases for means, sigmas, and multinomial weights:
mus = theta(1:split(1));
sigmas = theta((split(1)+1):split(2));
ws = theta((split(2)+1):end);

% Generate next-step variables:
mus_em = mus+1;
sigmas_em = sigmas+1;
ws_em = ws+1;

% Pre-allocate matrix for Gaussian pdfs:
pdfs_ = zeros(length(ws), length(x));

while ((norm(mus_em-mus)+norm(sigmas_em-sigmas)+norm(ws_em-ws)) > ep_)
    
    % Move one step forward the variables:
    mus = mus_em;
    sigmas = sigmas_em;
    ws = ws_em;
    
    % Compute the Gaussian pdf at all points in 'x':
    for i=1:length(ws)
        pdfs_(i,:) = pdf(makedist('Normal', mus(i), sqrt(sigmas(i))), x); 
    end
    
    % [ E-step ]
    omega_ = pdfs_.*ws(:) ./ sum(pdfs_.*ws(:));
    
    % [ M-step ]
    ws_em = 1/length(x) * sum(omega_, 2);
    mus_em = sum(omega_.*x(:)', 2) ./ sum(omega_, 2);
    sigmas_em = ...
        sum(omega_.*((repmat(x(:)',2,1)-repmat(mus_em(:),1,length(x))).^2), 2)...
        ./ sum(omega_, 2);

end

% Concatenate EM results intoo output vector:
theta_em = [mus_em(:)', sigmas_em(:)', ws_em(:)'];

end
