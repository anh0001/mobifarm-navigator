classdef ReshapeSCBtoSSCBLayer < nnet.layer.Layer & nnet.layer.Formattable
    properties
        % Define any properties if necessary
    end
    
    methods
        function layer = ReshapeSCBtoSSCBLayer(name)
            % Set layer name
            layer.Name = name;
            
            % Set layer description
            layer.Description = "Reshape layer from SCB to SSCB";
        end
        
        function Z = predict(layer, X)
            % Forward input data through the layer at prediction time
            % X: Input data of size [S, C, B]
            % Z: Output data of size [S, C, B] or [S, S, C, B]
            
            % Get the size of the input
            sz = size(X);
            
            % % Display the size of the input
            % disp('Size of X at input of predict:');
            % disp(sz);
            
            % If input is of size [S, C, B], reshape to [S, C, B, 1] and permute to [1, S, C, B]
            S = sz(1);
            C = sz(2);
            B = sz(3);
            Z = reshape(X, [S, C, B, 1]);
            Z = permute(Z, [4, 1, 2, 3]);
            Z = dlarray(Z, 'SSCB');
            
            % % Display the size of the output
            % disp('Size of Z at output of predict:');
            % disp(size(Z));
        end
        
        function [dLdX] = backward(layer, X, ~, dLdZ, ~)
            % Backward propagate the derivative of the loss function
            % through the layer
            % X: Input data of size [S, C, B]
            % Z: Output data of size [S, S, C] or [S, S, C, B]
            % dLdZ: Gradients propagated to the output [S, S, C] or [S, S, C, B]
            % dLdX: Gradients propagated to the input [S, C] or [S, C, B]
            
            % Permute back the gradients
            dLdZ_permuted = permute(dLdZ, [2, 3, 4, 1]);
            dLdX = reshape(dLdZ_permuted, size(X));
        end
    end
end