classdef FeatureTransformLayer < nnet.layer.Layer
    properties
        % Learnable parameters
        ConvLayers
    end
    
    methods
        function layer = FeatureTransformLayer(numPoints, name)
            % Create a FeatureTransformLayer
            layer.Name = name;
            layer.Description = "Feature Transformation Layer";
            
            % Define the convolutional layers
            layers = [
                convolution2dLayer([1 numPoints], 64, 'Padding', 'same')
                reluLayer
                convolution2dLayer([1 1], 128, 'Padding', 'same')
                reluLayer
                convolution2dLayer([1 1], 1024, 'Padding', 'same')
                maxPooling2dLayer([numPoints 1], 'Stride', 1)];
            layer.ConvLayers = layerGraph(layers);
        end
        
        function Z = predict(layer, X)
            % Forward pass through the FeatureTransform layer
            dlnet = dlnetwork(layer.ConvLayers);
            Z = predict(dlnet, X);
        end
        
        function [Z, memory] = forward(layer, X)
            % Forward pass through the FeatureTransform layer
            dlnet = dlnetwork(layer.ConvLayers);
            Z = predict(dlnet, X);
            memory = [];
        end
    end
end