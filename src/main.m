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
[dataTrain, dataVal, dataTest] = dataPrep();

%% Model Building
% Build and compile the deep learning model
disp('Building the model...');
model = modelBuild();

%% Model Training
% Train the model with the training data
disp('Training the model...');
[model, trainInfo] = trainModel(model, dataTrain, dataVal);

%% Model Evaluation
% Evaluate the model on the test data
disp('Evaluating the model...');
accuracy = evaluateModel(model, dataTest);

%% Results
% Display or save the results
disp(['Test Accuracy: ', num2str(accuracy)]);
disp('Project completed successfully.');

% Optionally, you can save the model and results
save('modelFinal.mat', 'model');
save('trainingInfo.mat', 'trainInfo');

% Remove paths added at the beginning
rmpath(genpath('src'));
rmpath(genpath('lib'));
rmpath(genpath('tools'));
