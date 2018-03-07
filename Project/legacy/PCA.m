function [ X_pca ] = PCA(X, k)

% Step 1:

mX = X - mean(X);

% Step 2:

% Covariance matrix:
C = cov(mX);

% Eigenvalues & eigenvectors:
[V, D] = eig(C);

% Step 3:

[~, I] = sort(diag(D));

V = V(:, I);
mX = mX(:, I);

X_pca = mX * V(:, 1:k);

% PCA = zeros(size(C,1),1);
% PCA(j) = b;
% PCA = diag(PCA);

end

