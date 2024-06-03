function createGui(model, dataVal)
    % Create the main figure
    fig = uifigure('Name', 'MobiFarm Navigator', 'Position', [100 100 800 600]);

    % Create axes for displaying images
    ax = uiaxes(fig, 'Position', [50 200 300 300]);
    
    % Create labels for displaying predictions
    uilabel(fig, 'Position', [400 450 100 30], 'Text', 'Linear Velocity:');
    linVelLabel = uilabel(fig, 'Position', [500 450 100 30], 'Text', '');

    uilabel(fig, 'Position', [400 400 100 30], 'Text', 'Angular Velocity:');
    angVelLabel = uilabel(fig, 'Position', [500 400 100 30], 'Text', '');

    % Create labels for displaying ground truth
    uilabel(fig, 'Position', [400 350 150 30], 'Text', 'Ground Truth Linear Velocity:');
    linVelGTLabel = uilabel(fig, 'Position', [550 350 100 30], 'Text', '');

    uilabel(fig, 'Position', [400 300 150 30], 'Text', 'Ground Truth Angular Velocity:');
    angVelGTLabel = uilabel(fig, 'Position', [550 300 100 30], 'Text', '');

    % Button for starting image display and prediction
    btnStart = uibutton(fig, 'Position', [200 50 100 30], 'Text', 'Start', ...
        'ButtonPushedFcn', @(btnStart,event) startImageDisplay(ax, linVelLabel, angVelLabel, linVelGTLabel, angVelGTLabel, model, dataVal));

    % Button for stopping image display
    btnStop = uibutton(fig, 'Position', [350 50 100 30], 'Text', 'Stop', ...
        'ButtonPushedFcn', @(btnStop,event) stopImageDisplay());
end

function startImageDisplay(ax, linVelLabel, angVelLabel, linVelGTLabel, angVelGTLabel, model, dataVal)
    % Ensure the tools directory is on the path
    addpath(genpath('src'));   % Add source code directory
    addpath(genpath('lib'));   % Add library directory
    addpath(genpath('tools')); % Add tools directory
    
    global stopFlag;
    stopFlag = false;

    % Continuously display images and make predictions until stopFlag is set to true
    while ~stopFlag
        reset(dataVal); % Reset the datastore to the beginning
        while hasdata(dataVal) && ~stopFlag
            data = read(dataVal);
            img = data{1};    % RGB image
            lidar = data{2};  % Distance map (Lidar)
            pcd = data{3};    % Point cloud data
            groundTruth = data{4}; % Ground truth labels
            
            imshow(img, 'Parent', ax);

            % Convert inputs to required formats
            img = single(img);
            lidar = single(lidar);
            pcd = single(pcd);

            % Convert to dlarray
            img = dlarray(img, 'SSCB');
            lidar = dlarray(lidar, 'SSCB');
            pcd = dlarray(pcd, 'SCB');

            % Perform prediction
            prediction = predict(model, img, lidar, pcd);
            prediction = extractdata(prediction);

            % Assuming the model returns linear and angular velocities
            linearVelocity = prediction(1);
            angularVelocity = prediction(2);

            % Extract ground truth velocities
            groundTruthLinVel = groundTruth(1);
            groundTruthAngVel = groundTruth(2);

            % Update labels with predictions
            linVelLabel.Text = num2str(linearVelocity);
            angVelLabel.Text = num2str(angularVelocity);

            % Update labels with ground truth
            linVelGTLabel.Text = num2str(groundTruthLinVel);
            angVelGTLabel.Text = num2str(groundTruthAngVel);

            pause(0.1);  % Adjust the pause duration as needed
        end
    end
end

function stopImageDisplay()
    global stopFlag;
    stopFlag = true;
end
