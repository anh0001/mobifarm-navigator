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

%% Data Preparation
% Load and preprocess data
disp('Loading and preprocessing data...');
[dataTrain, dataVal] = dataPrep();
dataTest = dataVal;

%% Model Building
% Build and compile the deep learning model
disp('Building the model...');
model = modelBuild();

% Analyze the network
disp('Analyzing the network...');
analyzeNetwork(model);

% Define the layers you want to train (e.g., the last three layers)
layersToTrain = {'conv1', 'conv1_1', 'conv', 'conv_1', 'conv_2', ...
    'batchnorm', 'batchnorm_1', 'batchnorm_2', 'fc', 'fc_1', 'fc_2'};

% Freeze all layers first by setting their learning rate multipliers to 0
model = modifyLearningRates(model, {model.Layers.Name}, 0);

% Now set the learning rate multipliers for the last few layers to 1 (or another value to train)
model = modifyLearningRates(model, layersToTrain, 1);

%% Model Training
% Train the model with the training data
disp('Training the model...');
[model, trainInfo] = trainModel(model, dataTrain, dataVal);

% %% Model Evaluation
% % Evaluate the model on the test data
% disp('Evaluating the model...');
% accuracy = evaluateModel(model, dataTest);
% 
% %% Results
% % Display or save the results
% disp(['Test Accuracy: ', num2str(accuracy)]);
% disp('Project completed successfully.');
% 
% % Optionally, you can save the model and results
% save('modelFinal.mat', 'model');
% save('trainingInfo.mat', 'trainInfo');
% 
% % Remove paths added at the beginning
% rmpath(genpath('src'));
% rmpath(genpath('lib'));
% rmpath(genpath('tools'));
