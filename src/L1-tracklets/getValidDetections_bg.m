function [valid, detections_in_interval] = getValidDetections_bg(detections_in_interval, detections_in_interval_bg, detections_conf, num_visible, opts, iCam)
%detections_in_interval = detections_in_window;
%detections_in_interval_bg = detections_in_window_bg;

% Flags valid OpenPose detections
valid = true(size(detections_in_interval,1),1);
for k = 1:size(detections_in_interval,1)
    pose = detections_in_interval(k, 3:end);
    bb = pose2bb(pose, opts.render_threshold);
    [newbb, newpose] = scale_bb(bb, pose,1.25);
    feet = feetPosition(newbb);
    detections_in_interval(k,[3 4 5 6]) = newbb;
    
    pose_bg = detections_in_interval_bg(k, 3:end);
    
    if (pose_bg(6) <5000 && pose_bg(6) ~=1) 
        valid(k) = 0;
        continue;
    end
    
    % Drop small and large detections
    if newbb(3) < 20 || newbb(4) < 20 || newbb(4) > 450
        valid(k) = 0;
        continue;
    end
    
    % Check detection confidence
    if num_visible(k) < 5 || detections_conf(k) < 4
        valid(k) = 0;
        continue;
    end
    
    % Filter feet outside of ROI
    if ~inpolygon(feet(:,1),feet(:,2),opts.ROIs{iCam}(:,1),opts.ROIs{iCam}(:,2))
        valid(k) = 0;
        continue;
    end

end

