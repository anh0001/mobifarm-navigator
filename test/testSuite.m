% Create an instance of PointNetLayer
layer = PointNetLayer(3, [64 64], 64, [128 1024], 'Name', 'pc_pointnet');

% Check the sizes of learnable parameters
disp('Size of inputTransformWeights:');
disp(size(layer.InputTransformWeights));
disp('Size of inputTransformBias:');
disp(size(layer.InputTransformBias));