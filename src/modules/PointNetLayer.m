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
        InputTransformNetwork
        InputTransformPredictionNetwork
        SharedMLP1Network
        FeatureTransformNetwork
        FeatureTransformPredictionNetwork
        SharedMLP2Network
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

            % Initialize PointNet encoder learnable parameters as dlnetwork objects
            layer.InputTransformNetwork = layer.createTNet(inputTransformSize, 'pcInputTransform');
            layer.InputTransformPredictionNetwork = layer.createTransformPredictionNetwork(inputTransformSize, 'pcInputTransformPrediction');
            layer.SharedMLP1Network = layer.createMLPNetwork(sharedMLP1Sizes, 'SharedMLP1');
            layer.FeatureTransformNetwork = layer.createTNet(featureTransformSize, 'pcFeatureTransform');
            layer.FeatureTransformPredictionNetwork = layer.createTransformPredictionNetwork(featureTransformSize, 'pcFeatureTransformPrediction');
            layer.SharedMLP2Network = layer.createMLPNetwork(sharedMLP2Sizes, 'pcSharedMLP2');
        end
        
        function Z = predict(layer, X)
            % PointNet encoder
            Z = layer.PointNetEncoder(X, layer.InputTransformNetwork, ...
                layer.InputTransformPredictionNetwork, layer.SharedMLP1Network, ...
                layer.FeatureTransformNetwork, layer.FeatureTransformPredictionNetwork, ...
                layer.SharedMLP2Network);
            
            % disp('Size of Z at output of predict:');
            % disp(size(Z));
        end
    end
    
    methods (Access = private)
        function dlnet = createTNet(~, size, prefix)
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
            % Input is point clouds SCB format
            % where C is 3 for 3D points, S is the number of points, and B is the batch size

            % Reshape the input to [3, numPoints, 1] for 'CB' format
            X_permuted = permute(X, [2, 1, 3]);
            
            % disp('Size of X at input to PointNetEncoder:');
            % disp(size(X_permuted));
            
            % Input transform (T-net)
            X_feature = layer.SharedMLP(X_permuted, inputTransformNet);
            T = layer.PredictTransform(X_feature, inputTransformPredictionNet);
            X = X * T;
            X_permuted = permute(X, [2, 1, 3]);
            
            % Shared MLP 1
            X = layer.SharedMLP(X_permuted, sharedMLP1Net);

            % Feature transform (T-net)
            X_feature = layer.SharedMLP(X, featureTransformNet);
            T = layer.PredictTransform(X_feature, featureTransformPredictionNet);
            X = permute(X, [2, 1, 3]);
            X = X * T;
            X_permuted = permute(X, [2, 1, 3]);

            % Shared MLP 2
            X = layer.SharedMLP(X_permuted, sharedMLP2Net);

            % Max pooling
            X = max(X, [], 2);
            
            % disp('Size of X after max pooling:');
            % disp(size(X));

            % Reshape the output to match the expected format SxC
            if size(X, 1) ~= 1 || size(X, 2) ~= 1
                X = reshape(X, [1, numel(X)]);
            end
            
            % Convert Z to a dlarray before returning
            X = dlarray(X);

            % Assign the final output to Z
            Z = X;
        end
        
        function T = PredictTransform(~, X, transformPredictionNet)
            % Symmetric function (max pooling)
            X = max(X, [], 2);

            % Affine transformation matrix prediction
            X = predict(transformPredictionNet, dlarray(X, 'CB'));
            T = reshape(extractdata(X), [sqrt(size(X, 1)), sqrt(size(X, 1))]);

            % disp('Affine transformation matrix T:');
            % disp(T);
        end

        function X = SharedMLP(~, X, dlnet)
            % No need to permute again here, as it is already in 'CB' format
            X = predict(dlnet, dlarray(X, 'CB'));
            X = extractdata(X);
            
            % disp(['Size of X after shared MLP:']);
            % disp(size(X));
        end
    end
end