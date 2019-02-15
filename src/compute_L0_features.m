function compute_L0_features(opts)
% Computes features for the input poses

for iCam = 6:8
    opts.current_camera = iCam;
    % Load poses
    % -- iI change the file path and delete the original part about openpose(1, 18) to detections(1, 6) --
    filename = fullfile(opts.detection,'top1', sprintf('camera%d.txt',iCam))
    detections_total = dlmread(filename);
    detections = ValidDetection(opts, detections_total ,iCam);
    %detections = detections(:,1:6);
    detections_save = fullfile(opts.detection,'top1', sprintf('camera%d.mat',iCam))
    save(detections_save,'detections');
    
    detections = load(detections_save, 'detections')
    detections = detections.detections;
    detections = NMS(opts, detections, iCam);
    % ---------------
    opts.experiment_name = 'demo';
    % load(fullfile(opts.dataset_path, 'detections','top1', sprintf('camera%d.mat',iCam)));
    % detections = detections(:,1:6);
    % Compute feature embeddings
    features = embed_detections(opts,detections);
    
    % Save features
    % h5write(sprintf('%s/%s/L0-features/features%d.h5',opts.experiment_root,opts.experiment_name,iCam),'/emb', features);
    filename_save = sprintf('%s/%s/L0-features/top1/features%d.mat',opts.experiment_root,opts.experiment_name,iCam);
    save(filename_save,'features');
end