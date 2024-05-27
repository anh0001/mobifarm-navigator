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
        inputs{i} = dataTest.(inputNames{i});
    end
    
    % Predict the outputs using the model
    predictedOutputs = predict(model, inputs{:});

    % Extract the ground truth labels from the test dataset
    actualOutputs = dataTest.Labels;

    % Calculate accuracy
    correctPredictions = sum(predictedOutputs == actualOutputs);
    totalPredictions = numel(actualOutputs);
    accuracy = correctPredictions / totalPredictions;

    disp(['Model accuracy: ', num2str(accuracy * 100), '%']);
end
