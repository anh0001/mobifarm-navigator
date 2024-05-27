function accuracy = evaluateModel(model, dataTest)
    % evaluateModel.m
    % This function loads the trained deep learning model from a .mat file
    % and evaluates it using the test dataset.
    % It returns the accuracy of the model.

    % Check the input names of the model
    inputNames = model.InputNames;

    % Prepare the inputs for the predict function
    inputs = cell(1, numel(inputNames));
    for i = 1:numel(inputNames)
        % Extract the i-th input from the combined datastore
        data = readall(dataTest.UnderlyingDatastores{i});
        
        % If data is a cell array, extract its contents
        if iscell(data)
            data = cell2mat(data);
        end

        % Convert to single if necessary
        if ~isa(data, 'single') && ~isa(data, 'double') && ~isa(data, 'logical')
            data = single(data);
        end
        
        % Convert to dlarray if necessary
        if ~isa(data, 'dlarray')
            data = dlarray(data);
        end
        
        inputs{i} = data;
    end
    
    % Predict the outputs using the model
    predictedOutputs = predict(model, inputs{:});

    % Extract the ground truth labels from the test dataset
    labelsDS = dataTest.UnderlyingDatastores{end}; % Assuming the labels are the last datastore
    actualOutputs = readall(labelsDS);
    
    % If actualOutputs is a cell array, extract its contents
    if iscell(actualOutputs)
        actualOutputs = cell2mat(actualOutputs);
    end

    % Convert labels to appropriate format if necessary
    if ~isa(actualOutputs, 'single') && ~isa(actualOutputs, 'double') && ~isa(actualOutputs, 'logical')
        actualOutputs = single(actualOutputs);
    end
    if ~isa(actualOutputs, 'dlarray')
        actualOutputs = dlarray(actualOutputs);
    end

    % Calculate accuracy
    correctPredictions = sum(extractdata(predictedOutputs) == extractdata(actualOutputs));
    totalPredictions = numel(extractdata(actualOutputs));
    accuracy = correctPredictions / totalPredictions;

    disp(['Model accuracy: ', num2str(accuracy * 100), '%']);
end
