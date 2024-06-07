function createGui(model, dataVal)
    % Create the main figure
    fig = uifigure('Name', 'MobiFarm Navigator', 'Position', [100 100 640 800]);

    % Create axes for displaying images
    ax = uiaxes(fig, 'Position', [20 300 640 480]); % Adjusted position to fit the width
    ax.XTick = [];
    ax.YTick = [];
    ax.DataAspectRatio = [480 640 1]; % Set the aspect ratio for 640x480 resolution

    % Center position calculation for the linear velocity gauge
    centerX = 20 + (640 / 2); % Calculate the center X position of the image display
    gaugeWidth = 400; % Width of the gauge

    % Create gauge for displaying linear velocity
    uilabel(fig, 'Position', [centerX - (gaugeWidth / 2) - 75, 250, 150, 30], 'Text', 'Predicted Velocity:');
    linVelGauge = uigauge(fig, 'linear', 'Position', [20, 200, gaugeWidth, 50], 'Limits', [-1 1]);
    linVelGauge.MajorTicks = [-1 -0.5 0 0.5 1];

    % Create semicircular gauge for displaying angular velocity
    % uilabel(fig, 'Position', [650 300 150 30], 'Text', 'Angular Velocity:');
    angVelGauge = uigauge(fig, 'semicircular', 'Position', [425 200 200 200], 'Limits', [-1 1]);
    angVelGauge.MajorTicks = [-1 -0.5 0 0.5 1];

    % Create gauge for displaying ground truth linear velocity
    uilabel(fig, 'Position', [centerX - (gaugeWidth / 2) - 75, 150, 150, 30], 'Text', 'GT Velocity:');
    linVelGTGauge = uigauge(fig, 'linear', 'Position', [20, 100, gaugeWidth, 50], 'Limits', [-1 1]);
    linVelGTGauge.MajorTicks = [-1 -0.5 0 0.5 1];

    % Create semicircular gauge for displaying ground truth angular velocity
    % uilabel(fig, 'Position', [650 175 200 30], 'Text', 'GT Angular Velocity:');
    angVelGTGauge = uigauge(fig, 'semicircular', 'Position', [425 100 200 200], 'Limits', [-1 1]);
    angVelGTGauge.MajorTicks = [-1 -0.5 0 0.5 1];

    % Button for starting image display and prediction
    btnStart = uibutton(fig, 'Position', [200 50 100 30], 'Text', 'Start', ...
        'ButtonPushedFcn', @(btnStart,event) startImageDisplay(ax, linVelGauge, angVelGauge, linVelGTGauge, angVelGTGauge, model, dataVal));

    % Button for stopping image display
    btnStop = uibutton(fig, 'Position', [350 50 100 30], 'Text', 'Stop', ...
        'ButtonPushedFcn', @(btnStop,event) stopImageDisplay());
end

function startImageDisplay(ax, linVelGauge, angVelGauge, linVelGTGauge, angVelGTGauge, model, dataVal)
    % Ensure the tools directory is on the path
    addpath(genpath('src'));   % Add source code directory
    addpath(genpath('lib'));   % Add library directory
    addpath(genpath('tools')); % Add tools directory
    
    global stopFlag;
    stopFlag = false;

    % Reset the datastore to the beginning
    reset(dataVal);

    % Continuously display images and make predictions until stopFlag is set to true
    while ~stopFlag
        while hasdata(dataVal) && ~stopFlag
            data = read(dataVal);
            img = data{1};    % RGB image
            lidar = data{2};  % Distance map (Lidar)
            pcd = data{3};    % Point cloud data
            groundTruth = data{4}; % Ground truth labels
            
            imshow(img, 'Parent', ax);
            ax.DataAspectRatioMode = 'auto'; % Set aspect ratio mode to 'auto'

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

            % Update gauges with predictions
            linVelGauge.Value = linearVelocity;
            angVelGauge.Value = angularVelocity * -1;

            % Update gauges with ground truth
            linVelGTGauge.Value = groundTruthLinVel;
            angVelGTGauge.Value = groundTruthAngVel * -1;

            pause(0.1);  % Adjust the pause duration as needed
        end
    end
end

function stopImageDisplay()
    global stopFlag;
    stopFlag = true;
end