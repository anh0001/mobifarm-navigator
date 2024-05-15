function lgraph = modifyLearningRates(lgraph, layerNames, lrMultiplier)
    % Modify the learning rates for specified layers in the layer graph.
    % lgraph: Layer graph of the model
    % layerNames: Cell array of names of layers to modify
    % lrMultiplier: Multiplier for learning rates (0 to freeze, >0 to enable learning)

    % Iterate over all layers in the graph
    for i = 1:numel(lgraph.Layers)
        layer = lgraph.Layers(i);
        layerName = layer.Name;

        % Check if this layer's name is in the list of layers to be modified
        if ismember(layerName, layerNames)
            if isprop(layer, 'WeightLearnRateFactor')
                layer.WeightLearnRateFactor = lrMultiplier;
            end
            if isprop(layer, 'BiasLearnRateFactor')
                layer.BiasLearnRateFactor = lrMultiplier;
            end

            % Update the layer in the layer graph
            lgraph = replaceLayer(lgraph, layerName, layer);
        end
    end
end
