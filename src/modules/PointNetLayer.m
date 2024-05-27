classdef PointNetLayer < nnet.layer.Layer & nnet.layer.Acceleratable
    % PointNetLayer
    % This class defines a custom layer for the PointNet architecture used in deep learning
    % for processing point clouds in 3D space. The layer includes learnable parameters and 
    % methods for prediction and encoding.

    properties
        % PointNet encoder properties
        InputTransformSize       % Size of the input transform network
        SharedMLP1Sizes          % Sizes of the first shared MLP layers
        FeatureTransformSize     % Size of the feature transform network
        SharedMLP2Sizes          % Sizes of the second shared MLP layers
    end
    
    properties (Learnable)
        % PointNet encoder learnable parameters
        InputTransformNetwork            % Learnable network for input transform
        InputTransformPredictionNetwork  % Learnable network for input transform prediction
        SharedMLP1Network                % Learnable network for the first shared MLP
        FeatureTransformNetwork          % Learnable network for feature transform
        FeatureTransformPredictionNetwork % Learnable network for feature transform prediction
        SharedMLP2Network                % Learnable network for the second shared MLP
    end
    
    methods
        function layer = PointNetLayer(inputTransformSize, sharedMLP1Sizes, featureTransformSize, sharedMLP2Sizes, varargin)
            % Constructor method to initialize the PointNetLayer with specified sizes
            
            % Set PointNet encoder properties
            layer.InputTransformSize = inputTransformSize;
            layer.SharedMLP1Sizes = sharedMLP1Sizes;
            layer.FeatureTransformSize = featureTransformSize;
            layer.SharedMLP2Sizes = sharedMLP2Sizes;

            % Parse optional Name parameter
            p = inputParser;
            addParameter(p, 'Name', '', @(x) ischar(x) || isstring(x));
            parse(p, varargin{:});
            layer.Name = p.Results.Name;

            % Initialize PointNet encoder learnable parameters as dlnetwork objects
            layer.InputTransformNetwork = layer.createTNet(inputTransformSize, 'pcInputTransform');
            layer.InputTransformPredictionNetwork = layer.createTransformPredictionNetwork(inputTransformSize, 'pcInputTransformPrediction');
            layer.SharedMLP1Network = layer.createMLPNetwork(sharedMLP1Sizes, 'pcSharedMLP1');
            layer.FeatureTransformNetwork = layer.createTNet(featureTransformSize, 'pcFeatureTransform');
            layer.FeatureTransformPredictionNetwork = layer.createTransformPredictionNetwork(featureTransformSize, 'pcFeatureTransformPrediction');
            layer.SharedMLP2Network = layer.createMLPNetwork(sharedMLP2Sizes, 'pcSharedMLP2');
        end
        
        function Z = predict(layer, X)
            % Predict method for the PointNetLayer
            % Encodes the input point cloud data using the PointNet encoder and returns the encoded output
            
            Z = layer.PointNetEncoder(X, layer.InputTransformNetwork, ...
                layer.InputTransformPredictionNetwork, layer.SharedMLP1Network, ...
                layer.FeatureTransformNetwork, layer.FeatureTransformPredictionNetwork, ...
                layer.SharedMLP2Network);
        end
    end
    
    methods (Access = private)
        function dlnet = createTNet(~, size, prefix)
            % Create a T-Net (transform network) for affine transformations
            layers = [
                featureInputLayer(size, 'Name', [prefix, '_input'])
                fullyConnectedLayer(size, 'Name', [prefix, '_fc1'])
                reluLayer('Name', [prefix, '_relu1'])
                fullyConnectedLayer(size, 'Name', [prefix, '_fc2'])
                reluLayer('Name', [prefix, '_relu2'])
                fullyConnectedLayer(size, 'Name', [prefix, '_fc3'])
                reluLayer('Name', [prefix, '_relu3'])
            ];
            lgraph = layerGraph(layers);
            dlnet = dlnetwork(lgraph);
        end

        function dlnet = createTransformPredictionNetwork(~, size, prefix)
            % Create a network for predicting affine transformation matrices
            layers = [
                featureInputLayer(size, 'Name', [prefix, '_input'])
                fullyConnectedLayer(size, 'Name', [prefix, '_fc4'])
                reluLayer('Name', [prefix, '_relu4'])
                fullyConnectedLayer(size^2, 'Name', [prefix, '_fc5'])
            ];
            lgraph = layerGraph(layers);
            dlnet = dlnetwork(lgraph);
        end
        
        function dlnet = createMLPNetwork(~, sizes, prefix)
            % Create a multi-layer perceptron (MLP) network
            layers = [
                featureInputLayer(sizes(1), 'Name', [prefix, '_input'])
            ];
            for i = 2:numel(sizes)
                layers = [
                    layers
                    fullyConnectedLayer(sizes(i), 'Name', [prefix, '_fc', num2str(i-1)])
                    reluLayer('Name', [prefix, '_relu', num2str(i-1)])
                ];
            end
            lgraph = layerGraph(layers);
            dlnet = dlnetwork(lgraph);
        end
        
        function Z = PointNetEncoder(layer, X, inputTransformNet, inputTransformPredictionNet, sharedMLP1Net, featureTransformNet, featureTransformPredictionNet, sharedMLP2Net)
            % PointNet encoder
            % Encodes the input point cloud data using a series of MLPs and transformation networks

            % Input is point clouds with dimensions (S, C, B)
            % where S is the number of points, C is 3 for 3D points, and B is the batch size
        
            % Ensure the input is in 'SCB' format
            if size(X, 2) ~= 3
                error('Input X must have dimensions [numPoints, 3, batchSize]');
            end
    
            % Initialize output
            numPoints = size(X, 1);
            numChannels = 3;
            batchSize = size(X, 3);
            outputSize = size(sharedMLP2Net.Layers(end-1).Weights, 1);  % Assuming the last fully connected layer size
            Z = zeros(1, outputSize, batchSize, 'like', X);  % Adjust the size for the output SxSxCxB
            
            for b = 1:batchSize
                % Extract the batch
                X_batch = X(:, :, b);

                % Permute to 'CB' format for processing
                X_permuted = permute(X_batch, [2, 1]);  % Now [3 numPoints]
        
                % Input transform (T-net)
                X_feature = layer.SharedMLP(X_permuted, inputTransformNet);
                T = layer.PredictTransform(X_feature, inputTransformPredictionNet);
                X_transformed = pagemtimes(X_batch, T);
                X_transformed = permute(X_transformed, [2, 1]);  % Now [3 numPoints]
        
                % Shared MLP 1
                X_mlp1 = layer.SharedMLP(X_transformed, sharedMLP1Net);
        
                % Feature transform (T-net)
                X_feature = layer.SharedMLP(X_mlp1, featureTransformNet);
                T = layer.PredictTransform(X_feature, featureTransformPredictionNet);
                X_mlp1 = permute(X_mlp1, [2, 1]);
                X_transformed = pagemtimes(X_mlp1, T);
                X_transformed = permute(X_transformed, [2, 1]);
        
                % Shared MLP 2
                X_mlp2 = layer.SharedMLP(X_transformed, sharedMLP2Net);
        
                % Max pooling along the points dimension
                X_maxpooled = max(X_mlp2, [], 2);  % Now [numChannels, 1]
        
                % Store the result for the batch
                Z(1, :, b) = reshape(X_maxpooled, [], 1);
            end
            
            % Convert Z to a dlarray before returning
            Z = dlarray(Z);
        end
        
        function T = PredictTransform(~, X, transformPredictionNet)
            % Predict affine transformation matrix
            % Predicts the transformation matrix from the encoded features

            % Symmetric function (max pooling)
            X = max(X, [], 2);

            % Affine transformation matrix prediction
            X = predict(transformPredictionNet, dlarray(X, 'CB'));
            T = reshape(extractdata(X), [sqrt(size(X, 1)), sqrt(size(X, 1))]);
        end

        function X = SharedMLP(~, X, dlnet)
            % Shared MLP
            % Applies a shared multi-layer perceptron to the input data

            % No need to permute again here, as it is already in 'CB' format
            X = predict(dlnet, dlarray(X, 'CB'));
            X = extractdata(X);
        end
    end
end
