classdef ReshapeSCBtoSSCBLayer < nnet.layer.Layer & nnet.layer.Formattable
    % ReshapeSCBtoSSCBLayer
    % This custom layer reshapes input data from SCB format (Spatial, Channel, Batch)
    % to SSCB format (Spatial, Spatial, Channel, Batch) for use in neural network layers.

    properties
        % Define any properties if necessary
    end
    
    methods
        function layer = ReshapeSCBtoSSCBLayer(name)
            % Constructor method to initialize the layer
            % Set layer name
            layer.Name = name;
            
            % Set layer description
            layer.Description = "Reshape layer from SCB to SSCB";
        end
        
        function Z = predict(layer, X)
            % Forward input data through the layer at prediction time
            % X: Input data of size [S, C, B]
            % Z: Output data of size [1, S, C, B]
            
            % Get the size of the input
            sz = size(X);
            
            % Extract dimensions
            S = sz(1);  % Spatial dimension
            C = sz(2);  % Channel dimension
            B = sz(3);  % Batch dimension
            
            % Reshape input from [S, C, B] to [S, C, B, 1] and then permute to [1, S, C, B]
            Z = reshape(X, [S, C, B, 1]);
            Z = permute(Z, [4, 1, 2, 3]);  % Permute dimensions to [1, S, C, B]
            
            % Convert the output to a dlarray with the format 'SSCB'
            Z = dlarray(Z, 'SSCB');
        end
        
        function [dLdX] = backward(layer, X, ~, dLdZ, ~)
            % Backward propagate the derivative of the loss function
            % through the layer
            % X: Input data of size [S, C, B]
            % dLdZ: Gradients propagated to the output [1, S, C, B]
            % dLdX: Gradients propagated to the input [S, C, B]
            
            % Permute back the gradients from [1, S, C, B] to [S, C, B, 1]
            dLdZ_permuted = permute(dLdZ, [2, 3, 4, 1]);
            
            % Reshape the permuted gradients to match the size of the input
            dLdX = reshape(dLdZ_permuted, size(X));
        end
    end
end
