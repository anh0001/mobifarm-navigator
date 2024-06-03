function predictVelocityCallback(ax, linVelLabel, angVelLabel)
    % Load the pre-trained deep learning model
    model = load('path/to/your/model.mat');
    
    % Get the current image from the axes UserData property
    img = ax.UserData;

    if isempty(img)
        uialert(fig, 'Please load an image first.', 'Error');
        return;
    end
    
    % Perform prediction (replace 'model' and 'predictFunction' with actual model and function)
    [linearVel, angularVel] = predictFunction(model, img);
    
    % Update the labels with the predicted velocities
    linVelLabel.Text = num2str(linearVel);
    angVelLabel.Text = num2str(angularVel);
end
