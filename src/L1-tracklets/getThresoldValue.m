function value = getThresoldValue(opts)

load(fullfile(opts.detection,opts.detection_name, sprintf('camera%d.mat',iCam)));
newDetections = detections(detections(:, 8) ~= 1, 8);

end