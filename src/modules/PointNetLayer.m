classdef PointNetLayer < nnet.layer.Layer & nnet.layer.Acceleratable
    
    properties
        % PointNet encoder properties
        InputTransformSize
        SharedMLP1Sizes
        FeatureTransformSize
        SharedMLP2Sizes
    end
    
    properties (Learnable)
        % PointNet encoder learnable parameters
        InputTransformWeights
        InputTransformBias
        SharedMLP1Weights
        SharedMLP1Bias
        FeatureTransformWeights
        FeatureTransformBias
        SharedMLP2Weights
        SharedMLP2Bias
    end
    
    methods
        function layer = PointNetLayer(inputTransformSize, sharedMLP1Sizes, featureTransformSize, sharedMLP2Sizes, varargin)
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

            % Initialize PointNet encoder learnable parameters
            [layer.InputTransformWeights, layer.InputTransformBias] = layer.InitializeMLPParameters([3, layer.InputTransformSize, layer.InputTransformSize^2]);
            [layer.SharedMLP1Weights, layer.SharedMLP1Bias] = layer.InitializeMLPParameters(layer.SharedMLP1Sizes);
            layer.FeatureTransformWeights = randn(layer.SharedMLP1Sizes(end), layer.FeatureTransformSize, 'single'); % Adjusted dimensions
            layer.FeatureTransformBias = zeros(1, layer.FeatureTransformSize, 'single');
            [layer.SharedMLP2Weights, layer.SharedMLP2Bias] = layer.InitializeMLPParameters(layer.SharedMLP2Sizes);

            % Convert cell arrays to numeric arrays for learnable properties
            layer.SharedMLP1Weights = cell2mat(layer.SharedMLP1Weights);
            layer.SharedMLP1Bias = cell2mat(layer.SharedMLP1Bias);
            layer.SharedMLP2Weights = cell2mat(layer.SharedMLP2Weights);
            layer.SharedMLP2Bias = cell2mat(layer.SharedMLP2Bias);
        end
        
        function Z = predict(layer, X)
            % PointNet encoder
            Z = layer.PointNetEncoder(X, layer.InputTransformWeights, layer.InputTransformBias, ...
                layer.SharedMLP1Weights, layer.SharedMLP1Bias, ...
                layer.FeatureTransformWeights, layer.FeatureTransformBias, ...
                layer.SharedMLP2Weights, layer.SharedMLP2Bias);
            
            disp('Size of Z at output of predict:');
            disp(size(Z));
            
            % Reshape the output to match the expected format [1, 1, featureSize]
            Z = reshape(Z, [1, 1, size(Z, 2)]);
        end
    end
    
    methods (Access = private)
        function [weights, bias] = InitializeMLPParameters(layer, sizes)
            numLayers = numel(sizes) - 1;
            weights = cell(1, numLayers);
            bias = cell(1, numLayers);
            for i = 1:numLayers
                weights{i} = randn(sizes(i+1), sizes(i), 'single');
                bias{i} = zeros(sizes(i+1), 1, 'single');
            end
        end
        
        function X = PointNetEncoder(layer, X, inputTransformWeights, inputTransformBias, ...
                                     sharedMLP1Weights, sharedMLP1Bias, ...
                                     featureTransformWeights, featureTransformBias, ...
                                     sharedMLP2Weights, sharedMLP2Bias)
            
            % Reshape the input to [numPoints, 3]
            numPoints = size(X, 1);
            X = reshape(X, numPoints, 3);  % Added reshaping of input data
            
            disp('Size of X at input to PointNetEncoder:');
            disp(size(X));
            
            % Input transform (T-net)
            T = layer.TNet(X, inputTransformWeights, inputTransformBias);
            X = X * T;
                        
            % Shared MLP 1
            X = layer.SharedMLP(X, sharedMLP1Weights, sharedMLP1Bias);

            % Feature transform (T-net)
            T = layer.TNet(X, featureTransformWeights, featureTransformBias);
            X = X * T;

            % Shared MLP 2
            X = layer.SharedMLP(X, sharedMLP2Weights, sharedMLP2Bias);

            % Max pooling
            X = max(X, [], 1);
            
            disp('Size of X after max pooling:');
            disp(size(X));
        end
        
        function T = TNet(layer, X, weights, bias)
            % Point-independent feature extraction
            X = layer.SharedMLP(X, weights(1:2), bias(1:2));

            % Max pooling layer
            X = max(X, [], 1);

            % Fully connected layers to predict affine transformation matrix
            X = layer.SharedMLP(X, weights(3:end), bias(3:end));

            % Reshape to form affine transformation matrix
            T = reshape(X, sqrt(size(X, 2)), sqrt(size(X, 2)));

            disp('Affine transformation matrix T:');
            disp(T);
        end

        function X = SharedMLP(layer, X, weights, bias)
            for i = 1:numel(weights)
                X = relu(X * weights{i} + bias{i}'); % Adjusted dimensions
                disp(['Size of X after layer ', num2str(i), ' in shared MLP:']);
                disp(size(X));
            end
        end
    end
end