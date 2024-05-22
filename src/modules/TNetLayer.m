classdef TNetLayer < nnet.layer.Layer
    properties
        % Learnable parameters
        Weights
        Bias
    end
    
    methods
        function layer = TNetLayer(numPoints, numFeatures, name)
            % Create a TNetLayer
            layer.Name = name;
            layer.Description = "T-Net layer";
            
            % Initialize the learnable parameters
            layer.Weights = randn([numFeatures numPoints]);
            layer.Bias = zeros([numFeatures 1]);
        end
        
        function Z = predict(layer, X)
            % Forward pass through the TNet layer
            Z = X * layer.Weights' + layer.Bias';
        end
        
        function [Z, memory] = forward(layer, X)
            % Forward pass through the TNet layer
            Z = X * layer.Weights' + layer.Bias';
            memory = [];
        end
    end
end