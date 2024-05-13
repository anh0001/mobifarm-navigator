function initialize()
    % initialize.m
    % This function sets up the initial environment and configurations for
    % the mobile robot navigation project.

    disp('Initializing environment and configurations...');

    %% Add Project-Specific Paths
    % Include directories for custom functions, utilities, and additional libraries
    addpath('src/modules');
    addpath('tools');
    addpath('lib');

    %% Configure Environment
    % Check for the availability of the required toolboxes
    requiredToolboxes = {'Deep Learning Toolbox', 'Robotics System Toolbox', ...
                         'Computer Vision Toolbox', 'Sensor Fusion and Tracking Toolbox'};
    checkToolboxes(requiredToolboxes);

    %% GPU Configuration
    % Configure the GPU if it is available and intended for use
    if gpuDeviceCount > 0
        gpuDevice(1); % Selects the first available GPU
        disp(['Using GPU: ' gpuDevice(1).Name]);
    else
        disp('GPU not available, using CPU instead.');
    end

    %% Set Global Variables and Constants
    % Define any global variables and constants used across various scripts
    global IMAGE_SIZE;
    IMAGE_SIZE = [224 224]; % Example image size for neural network input

    global BATCH_SIZE;
    BATCH_SIZE = 64; % Example batch size for training

    disp('Initialization complete.');
end

function checkToolboxes(toolboxes)
    % Helper function to check for the presence of required MATLAB toolboxes
    v = ver;
    installedToolboxes = {v.Name};
    missingToolboxes = setdiff(toolboxes, installedToolboxes);
    if ~isempty(missingToolboxes)
        error(['Required toolboxes are missing: ', strjoin(missingToolboxes, ', ')]);
    end
    disp('All required toolboxes are installed.');
end
