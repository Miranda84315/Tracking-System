function detections = ValidDetection(opts, detections_total, iCam)

detections_total(:, 5) = detections_total(:, 5) - detections_total(:, 3);
detections_total(:, 6) = detections_total(:, 6) - detections_total(:, 4);
valid = true(size(detections_total,1),1);
for k = 1:size(detections_total,1)
    global_time = opts.start_frames(iCam) + detections_total(k, 2) - 1;
    if ~ismember(global_time, opts.sequence_intervals{opts.sequence}) 
        valid(k) = 0;
    else
        if detections_total(k, 7)<= 0.9
            valid(k) = 0;
        end
    end
    newbb = detections_total(k, 3:6);
    feet = feetPosition(newbb);
    
       if ~inpolygon(feet(:,1),feet(:,2),opts.ROIs{iCam}(:,1),opts.ROIs{iCam}(:,2))
            valid(k) = 0;
        end
    
end

detections = detections_total(valid,:);

end