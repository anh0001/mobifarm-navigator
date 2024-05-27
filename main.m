%% Main Script for Mobile Robot Navigation Project
% This script coordinates the initialization, data preparation, model building,
% training, and evaluation for mobile robot navigation using deep learning.

% Add project directories to path
addpath(genpath('src'));   % Add source code directory
addpath(genpath('lib'));   % Add library directory
addpath(genpath('tools')); % Add tools directory

% Clear workspace and command window
clear; % Clear all variables from the workspace
clc;   % Clear the command window

disp('Starting the Mobile Robot Navigation Project...');

%% Initialization
% Set up parameters and configurations
disp('Initializing project...');
initialize; % Call the initialization script

% Ask the user for the mode (training, resume from checkpoints, or evaluate model)
mode = input('Enter the mode (training, resume, evaluate): ', 's');

%% Data Preparation
% Load and preprocess data
disp('Loading and preprocessing data...');
[dataTrain, dataVal] = dataPrep(); % Load and preprocess training and validation data
dataTest = dataVal;                % Assign validation data to test data for this example

%% Model Building
% Check if there is a checkpoint to resume from
checkpointDir = 'data/checkpoints';        % Directory containing checkpoint files
checkpointFiles = dir(fullfile(checkpointDir, '*.mat')); % List all .mat files in the checkpoint directory

% Check if there is a .mat file in the models/ folder
modelDir = 'models';           % Directory containing model files
modelFiles = dir(fullfile(modelDir, 'model_*.mat')); % List all .mat files in the model directory

if strcmp(mode, 'resume') && ~isempty(checkpointFiles)
    % Load the latest checkpoint if resuming
    [~, idx] = max([checkpointFiles.datenum]); % Find the latest checkpoint file
    latestCheckpoint = fullfile(checkpointFiles(idx).folder, checkpointFiles(idx).name);
    disp(['Resuming from checkpoint: ', latestCheckpoint]);
    load(latestCheckpoint, 'net', 'trainingOptions'); % Load the network and training options
    model = net; % Rename loaded network to model for consistency
    
elseif strcmp(mode, 'training')
    % Build the model from scratch if training
    disp('Building the model...');
    model = modelBuild(); % Call function to build the model
    
    % Analyze the network
    disp('Analyzing the network...');
    analyzeNetwork(model); % Call function to analyze the network
    
    % Define the layers to train
    layersToTrain = {'conv1', 'conv1_1', 'pc_pointnet', 'rgbd_conv1', 'rgbd_conv1', 'combine_fc1', 'combine_fc2'};
    
    % Freeze all layers by setting their learning rate multipliers to 0
    model = modifyLearningRates(model, {model.Layers.Name}, 0);
    
    % Set the learning rate multipliers for the specified layers to 1 (or another value)
    model = modifyLearningRates(model, layersToTrain, 1);
end

if strcmp(mode, 'training') || strcmp(mode, 'resume')
    %% Model Training
    % Train the model with the training data
    disp('Training the model...');
    [model, trainInfo] = trainModel(model, dataTrain, dataVal); % Call function to train the model
    
    % Optionally, save the model and training information
    if ~isempty(modelFiles)
        % Extract the index of the latest model file
        latestModelIndex = max(cellfun(@(x) str2double(x{1}), regexp({modelFiles.name}, 'model_(\d+)\.mat', 'tokens')));
    else
        % No existing model files, start from 0
        latestModelIndex = 0;
    end
    newModelIndex = latestModelIndex + 1;
    save(fullfile(modelDir, sprintf('model_%03d.mat', newModelIndex)), 'model'); % Save the trained model
    save(fullfile(modelDir, sprintf('trainingInfo_%03d.mat', newModelIndex)), 'trainInfo'); % Save the training information
end

if strcmp(mode, 'evaluate')
    %% Model Evaluation
    % Select the latest model file
    [~, idx] = max([modelFiles.datenum]); % Find the latest model file
    latestModel = fullfile(modelFiles(idx).folder, modelFiles(idx).name);
    disp(['Evaluating the latest model: ', latestModel]);
    load(latestModel, 'model'); % Load the model
    
    % Evaluate the model on the test data
    disp('Evaluating the model...');
    batchSize = 10; % Set an appropriate batch size for evaluation
    rmse = evaluateModel(model, dataVal, batchSize); % Call function to evaluate the model
    disp(['Overall model RMSE: ', num2str(rmse)]);
end

% Remove paths added at the beginning
rmpath(genpath('src'));   % Remove source code directory
rmpath(genpath('lib'));   % Remove library directory
rmpath(genpath('tools')); % Remove tools directory
