% -------------------------------------------
%
%   Compito 5 - TAE
%
% -------------------------------------------

clear

load('ESAME001.mat')

% Fit logit:

Y = ESAME1(:,1);
X = ESAME1(:,3:end);

fit_logit = glmfit(X, Y, 'binomial', 'link', 'logit');

% Predict:
