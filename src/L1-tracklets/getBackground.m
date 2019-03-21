function getBackground(opts)

frame = [179020; 179020; 100765; 125940; 151720; 114305; 109530; 109185];

for iCam=1:8
    img = opts.reader.getFrame(iCam, frame(iCam));
    figure, imshow(img);
    file_name = sprintf('dataset/background/bg_cam%d.jpg', iCam);
    imwrite(img, file_name);
end
end