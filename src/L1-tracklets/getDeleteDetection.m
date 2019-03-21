function getDeleteDetection(opts)
for iCam=1:8
    load(fullfile(opts.detection,opts.detection_name, sprintf('camera%d.mat',iCam)));
    for k = 1:size(detections, 1)
        pose = detections(k, 3:end);
        bb = pose2bb(pose, opts.render_threshold);
        [newbb, newpose] = scale_bb(bb, pose, 1.25);
        detections(k,[3 4 5 6]) = newbb;
    end
    detections = detections(:, 1:6);
    file_name = fullfile(opts.detection, 'openpose_bb', sprintf('camera%d.mat',iCam))
    save(file_name, 'detections');
end
end