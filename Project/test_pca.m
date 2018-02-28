r = rand(1000, 1);
rr = 0.5 + 0.5*rand(1000, 1);

x = rr .* cos( r * 2 * pi );
y = rr .* sin( r * 2 * pi );
z = x;

figure
plot3(x,y,z,'o')
grid on

out = PCA([x,y,z], 2);

coeff = pca([x,y,z]);
new = [x,y,z] * coeff;

figure
plot(new(:,1), new(:,2), '+')