% Create a custom layer to print sizes before concatenation
classdef DebugLayer < nnet.layer.Layer
    methods
        function layer = DebugLayer(name)
            layer.Name = name;
            layer.Description = "Layer to print the size of its input";
        end
        
        function Z = predict(layer, X)
            disp(['Size of input to ', layer.Name, ':']);
            disp(size(X));
            Z = X; % Pass the input to the output
        end
    end
end