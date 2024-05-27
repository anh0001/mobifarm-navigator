function accuracy = evaluateModel(model, dataTest, batchSize)
    % evaluateModel
    % This function evaluates the model in batches to handle large datasets.
    % Inputs:
    %   - model: the trained model
    %   - dataTest: the test dataset
    %   - batchSize: size of each batch
    % Output:
    %   - accuracy: overall accuracy of the model

    numObservations = numel(dataTest.UnderlyingDatastores{1}.Files);
    correctPredictions = 0;
    totalPredictions = 0;

    for startIdx = 1:batchSize:numObservations
        endIdx = min(startIdx + batchSize - 1, numObservations);
        batchIndices = startIdx:endIdx;

        % Extract the batch data
        batchDataTest = subset(dataTest, batchIndices);

        % Evaluate the batch
        batchAccuracy = evaluateBatch(model, batchDataTest);
        
        % Update overall accuracy
        correctPredictions = correctPredictions + batchAccuracy.correct;
        totalPredictions = totalPredictions + batchAccuracy.total;
    end

    % Compute the overall accuracy
    accuracy = correctPredictions / totalPredictions;

    disp(['Overall model accuracy: ', num2str(accuracy * 100), '%']);
end

function batchAccuracy = evaluateBatch(model, batchDataTest)
    % evaluateBatch
    % This function evaluates a batch of data.
    % Inputs:
    %   - model: the trained model
    %   - batchDataTest: a batch of test data
    % Output:
    %   - batchAccuracy: a structure with correct and total predictions

    % Check the input names of the model
    inputNames = model.InputNames;
    
    % Prepare the inputs for the predict function
    inputs = cell(1, numel(inputNames));
    for i = 1:numel(inputNames)
        % Extract the i-th input from the combined datastore
        data = readall(batchDataTest.UnderlyingDatastores{i});
        
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

    % Extract the ground truth labels from the batch data
    labelsDS = batchDataTest.UnderlyingDatastores{end}; % Assuming the labels are the last datastore
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

    % Calculate batch accuracy
    correctPredictions = sum(extractdata(predictedOutputs) == extractdata(actualOutputs));
    totalPredictions = numel(extractdata(actualOutputs));
    
    % Store results in a structure
    batchAccuracy.correct = correctPredictions;
    batchAccuracy.total = totalPredictions;
end
