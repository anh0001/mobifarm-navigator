function createGui(dataVal)
    % Create the main figure
    fig = uifigure('Name', 'MobiFarm Navigator', 'Position', [100 100 800 600]);

    % Create axes for displaying images
    ax = uiaxes(fig, 'Position', [50 200 300 300]);
    
    % Create labels for displaying predictions
    uilabel(fig, 'Position', [400 450 100 30], 'Text', 'Linear Velocity:');
    linVelLabel = uilabel(fig, 'Position', [500 450 100 30], 'Text', '');

    uilabel(fig, 'Position', [400 400 100 30], 'Text', 'Angular Velocity:');
    angVelLabel = uilabel(fig, 'Position', [500 400 100 30], 'Text', '');

    % Button for starting image display
    btnStart = uibutton(fig, 'Position', [200 50 100 30], 'Text', 'Start', ...
        'ButtonPushedFcn', @(btnStart,event) startImageDisplay(ax, linVelLabel, angVelLabel, dataVal));

    % Button for stopping image display
    btnStop = uibutton(fig, 'Position', [350 50 100 30], 'Text', 'Stop', ...
        'ButtonPushedFcn', @(btnStop,event) stopImageDisplay());
end

function startImageDisplay(ax, linVelLabel, angVelLabel, dataVal)
    % Ensure the tools directory is on the path
    addpath(genpath('tools'));

    global stopFlag;
    stopFlag = false;

    % Continuously display images until stopFlag is set to true
    while ~stopFlag
        reset(dataVal); % Reset the datastore to the beginning
        while hasdata(dataVal) && ~stopFlag
            img = read(dataVal);
            imshow(img{1}, 'Parent', ax);
            pause(0.5);  % Adjust the pause duration as needed
        end
    end
end

function stopImageDisplay()
    global stopFlag;
    stopFlag = true;
end
