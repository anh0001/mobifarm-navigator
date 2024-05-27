%% Main Script for Mobile Robot Navigation Project
% This script coordinates the initialization, data preparation, model building,
% training, and evaluation for mobile robot navigation using deep learning.

% Add project directories to path
addpath(genpath('src'));
addpath(genpath('lib'));
addpath(genpath('tools'));

% Clear workspace and command window
clear;
clc;

disp('Starting the Mobile Robot Navigation Project...');

%% Initialization
% Set up parameters and configurations
disp('Initializing project...');
initialize;

% Ask the user for the mode (training, resume from checkpoints, or evaluate model)
mode = input('Enter the mode (training, resume, evaluate): ', 's');

%% Data Preparation
% Load and preprocess data
disp('Loading and preprocessing data...');
[dataTrain, dataVal] = dataPrep();
dataTest = dataVal;

%% Model Building
% Check if there is a checkpoint to resume from
checkpointDir = 'data/checkpoints';
checkpointFiles = dir(fullfile(checkpointDir, '*.mat'));

% Check if there is a .mat file in the models/ folder
modelDir = 'models';
modelFiles = dir(fullfile(modelDir, '*.mat'));

if strcmp(mode, 'resume') && ~isempty(checkpointFiles)
    % Load the latest checkpoint
    [~, idx] = max([checkpointFiles.datenum]);
    latestCheckpoint = fullfile(checkpointFiles(idx).folder, checkpointFiles(idx).name);
    disp(['Resuming from checkpoint: ', latestCheckpoint]);
    load(latestCheckpoint, 'net', 'trainingOptions'); % Load the network and training options
    model = net; % Rename loaded network to model for consistency
    
elseif strcmp(mode, 'training')
    % No checkpoint found, build the model from scratch
    disp('Building the model...');
    model = modelBuild();
    
    % Analyze the network
    disp('Analyzing the network...');
    analyzeNetwork(model);
    
    % Define the layers you want to train (e.g., the last three layers)
    layersToTrain = {'conv1', 'conv1_1', 'pc_pointnet', 'rgbd_conv1', 'rgbd_conv1', 'combine_fc1', 'combine_fc2'};
    
    % Freeze all layers first by setting their learning rate multipliers to 0
    model = modifyLearningRates(model, {model.Layers.Name}, 0);
    
    % Now set the learning rate multipliers for the last few layers to 1 (or another value to train)
    model = modifyLearningRates(model, layersToTrain, 1);
end

if strcmp(mode, 'training') || strcmp(mode, 'resume')
    %% Model Training
    % Train the model with the training data
    disp('Training the model...');
    [model, trainInfo] = trainModel(model, dataTrain, dataVal);
    
    % Optionally, you can save the model and results
    save('model.mat', 'model');
    save('trainingInfo.mat', 'trainInfo');
end

if strcmp(mode, 'evaluate')
    %% Model Evaluation
    
    % select the latest model file
    [~, idx] = max([modelFiles.datenum]);
    latestModel = fullfile(modelFiles(idx).folder, modelFiles(idx).name);
    disp(['Evaluating the latest model: ', latestModel]);
    load(latestModel, 'model'); % Load the model
    
    % Evaluate the model on the test data
    disp('Evaluating the model...');
    batchSize = 10; % Set an appropriate batch size
    accuracy = evaluateModel(model, dataVal, batchSize);
    disp(['Test Accuracy: ', num2str(accuracy)]);
end

% Remove paths added at the beginning
rmpath(genpath('src'));
rmpath(genpath('lib'));
rmpath(genpath('tools'));