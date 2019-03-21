function compute_L2_trajectories_new(opts)
% Computes single-camera trajectories from tracklets
for iCam = 1:8

    % Initialize
    % -- load tracklets from L1-tracklet.mat
    % -- trajectoriesFromTracklets include detection start/endFrame and
    % -- segmentStart/End
    load(fullfile(opts.experiment_root, opts.experiment_name, 'L1-tracklets', sprintf('tracklets%d_%s.mat',iCam,opts.sequence_names{opts.sequence})));
    trajectoriesFromTracklets = trackletsToTrajectories(tracklets,1:length(tracklets));
    
    opts.current_camera = iCam;
    sequence_interval = opts.sequence_intervals{opts.sequence};
    % -- use 1 second long windows ,overlap 50 % to product 10second long
    % -- single camera trajecories
    startFrame = global2local(opts.start_frames(opts.current_camera), sequence_interval(1) - opts.trajectories.window_width);
    endFrame   = global2local(opts.start_frames(opts.current_camera), sequence_interval(1) + opts.trajectories.window_width);

    trajectories = trajectoriesFromTracklets; 
    
    while startFrame <= global2local(opts.start_frames(opts.current_camera), sequence_interval(end))
        % Display loop state
        clc; fprintf('Cam: %d - Window %d...%d\n', iCam, startFrame, endFrame);

        % Compute trajectories in current time window
        trajectories = createTrajectories_new( opts, trajectories, startFrame, endFrame);

        % Update loop range
        startFrame = endFrame   - opts.trajectories.overlap;
        endFrame   = startFrame + opts.trajectories.window_width;
    end


end
end