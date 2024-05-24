classdef PointNetLayer < nnet.layer.Layer
    
    properties
        % PointNet encoder properties
        InputTransformSize
        SharedMLP1Sizes
        FeatureTransformSize
        SharedMLP2Sizes
    end
    
    properties (Learnable)
        % PointNet encoder learnable parameters as numeric arrays
        InputTransformWeights
        InputTransformBiases
        SharedMLP1Weights
        SharedMLP1Biases
        FeatureTransformWeights
        FeatureTransformBiases
        SharedMLP2Weights
        SharedMLP2Biases
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

            % Initialize PointNet encoder learnable parameters as numeric arrays
            layer.InputTransformWeights = randn([inputTransformSize, 3]);
            layer.InputTransformBiases = randn([inputTransformSize, 1]);
            layer.SharedMLP1Weights = randn([sharedMLP1Sizes, inputTransformSize]);
            layer.SharedMLP1Biases = randn([sharedMLP1Sizes, 1]);
            layer.FeatureTransformWeights = randn([featureTransformSize, sharedMLP1Sizes]);
            layer.FeatureTransformBiases = randn([featureTransformSize, 1]);
            layer.SharedMLP2Weights = randn([sharedMLP2Sizes, featureTransformSize]);
            layer.SharedMLP2Biases = randn([sharedMLP2Sizes, 1]);
        end
        
        function Z = predict(layer, X)
            % PointNet encoder
            Z = layer.PointNetEncoder(X, ...
                layer.InputTransformWeights, layer.InputTransformBiases, ...
                layer.SharedMLP1Weights, layer.SharedMLP1Biases, ...
                layer.FeatureTransformWeights, layer.FeatureTransformBiases, ...
                layer.SharedMLP2Weights, layer.SharedMLP2Biases);
            
            % Ensure the output is of type dlarray
            Z = dlarray(Z);
        end

        function [dLdX, dLdW] = backward(layer, X, ~, dLdZ, ~)
            % Backward function for computing the gradients of the loss with respect to
            % the input data X and the learnable parameters W.
            
            % Convert dLdZ to dlarray
            dLdZ = dlarray(dLdZ);
            
            % Compute gradients for Shared MLP2
            [dLdX_mlp2, dLdW_mlp2] = dlgradient(dLdZ, {layer.SharedMLP2Weights, layer.SharedMLP2Biases});
            
            % Compute gradients for Feature Transform Network
            [dLdX_feature, dLdW_feature] = dlgradient(dLdX_mlp2, {layer.FeatureTransformWeights, layer.FeatureTransformBiases});
            
            % Compute gradients for Shared MLP1
            [dLdX_mlp1, dLdW_mlp1] = dlgradient(dLdX_feature, {layer.SharedMLP1Weights, layer.SharedMLP1Biases});
            
            % Compute gradients for Input Transform Network
            [dLdX_input, dLdW_input] = dlgradient(dLdX_mlp1, {layer.InputTransformWeights, layer.InputTransformBiases});
            
            % Collect all gradients
            dLdW = [dLdW_input, dLdW_mlp1, dLdW_feature, dLdW_mlp2];
            
            % Compute gradient of the loss with respect to input X
            dLdX = dlgradient(dLdX_input, X);
        end
    end
    
    methods (Access = private)
        function Z = PointNetEncoder(layer, X, inputTransformWeights, inputTransformBiases, ...
            sharedMLP1Weights, sharedMLP1Biases, featureTransformWeights, featureTransformBiases, ...
            sharedMLP2Weights, sharedMLP2Biases)
            % Input is point clouds with dimensions (S, C, B)
            % where S is the number of points, C is 3 for 3D points, and B is the batch size
        
            % Ensure the input is in 'SCB' format
            if size(X, 2) ~= 3
                error('Input X must have dimensions [numPoints, 3, batchSize]');
            end
        
            % Initialize output
            numPoints = size(X, 1);
            batchSize = size(X, 3);
            outputSize = size(sharedMLP2Weights, 1);  % Assuming the last fully connected layer size
            Z = zeros(1, outputSize, batchSize, 'like', X);
            
            for b = 1:batchSize
                % Extract the batch
                X_batch = X(:, :, b);
        
                % Permute to 'CB' format for processing
                X_permuted = permute(X_batch, [2, 1]);  % Now [3, numPoints]
        
                % Input transform (T-net)
                X_feature = X_permuted * inputTransformWeights + inputTransformBiases;
                T = layer.PredictTransform(X_feature, inputTransformWeights, inputTransformBiases);
                X_transformed = pagemtimes(X_batch, T);
                X_transformed = permute(X_transformed, [2, 1]);
        
                % Shared MLP 1
                X_mlp1 = X_transformed * sharedMLP1Weights + sharedMLP1Biases;
        
                % Feature transform (T-net)
                X_feature = X_mlp1 * featureTransformWeights + featureTransformBiases;
                T = layer.PredictTransform(X_feature, featureTransformWeights, featureTransformBiases);
                X_transformed = pagemtimes(X_mlp1, T);
                X_transformed = permute(X_transformed, [2, 1]);
        
                % Shared MLP 2
                X_mlp2 = X_transformed * sharedMLP2Weights + sharedMLP2Biases;
        
                % Max pooling along the points dimension
                X_maxpooled = max(X_mlp2, [], 2);  % Now [numChannels, 1]
        
                % Store the result for the batch
                Z(1, :, b) = reshape(X_maxpooled, [], 1);  % Flatten to [outputSize, 1]
            end
            
            % Convert Z to a dlarray before returning
            Z = dlarray(Z);
        end
        
        function T = PredictTransform(~, X, transformWeights, transformBiases)
            % Symmetric function (max pooling)
            X = max(X, [], 2);
            
            % Affine transformation matrix prediction
            T = X * transformWeights + transformBiases;
            T = reshape(T, [sqrt(numel(T)), sqrt(numel(T))]);
            
            % Ensure T is of type dlarray
            T = dlarray(T);
        end

        function X = SharedMLP(~, X, weights, biases)
            % Apply the shared MLP layers
            X = X * weights + biases;
            
            % Apply ReLU activation
            X = max(X, 0);  % ReLU function
            
            % Ensure X is of type dlarray
            X = dlarray(X);
        end
    end
end