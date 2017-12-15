function [ decisionmap ] = getDecisionBoundaryPlot( xrange, yrange, ...
                                                    func,  inc)
 
% generate grid coordinates. this will be the basis of the decision
% boundary visualization.
[x, y] = meshgrid(xrange(1):inc:xrange(2), yrange(1):inc:yrange(2));
 
% size of the (x, y) image, which will also be the size of the 
% decision boundary image that is used as the plot background.
image_size = size(x);
 
xy = [reshape(x, image_size(1)*image_size(2), 1), ...
                reshape(y, image_size(1)*image_size(2), 1)];

xyclass = (func(xy) > 0);

% reshape the idx (which contains the class label) into an image.
decisionmap = reshape(xyclass, image_size);


end

