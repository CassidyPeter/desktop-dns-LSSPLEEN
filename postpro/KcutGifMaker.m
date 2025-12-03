clc; close all; clear;

% Parameters
% folder = 'C:\Users\cassidy\OneDrive - vki.ac.be\Documents\Technical\CAD and CFD\desktop-dns-main\outputs\main\task3highturb\vortz_350dpi';
% imageFiles = dir(fullfile(folder, 'Kcut_*.png'));

folder = 'C:\Users\cassidy\OneDrive - vki.ac.be\Documents\Technical\CAD and CFD\desktop-dns-main\outputs\meshsensitivity\finemesh\vortz_Blade';
imageFiles = dir(fullfile(folder, 'Kcut_*_Blade.png'));

folder = 'C:\Users\cassidy\OneDrive - vki.ac.be\Documents\Technical\CAD and CFD\desktop-dns-main\outputs\meshsensitivity\finemesh\vortz_SSBL';
imageFiles = dir(fullfile(folder, 'Kcut_*_SSBL.png'));

% folder = 'C:\Users\cassidy\OneDrive - vki.ac.be\Documents\Technical\CAD and CFD\desktop-dns-main\outputs\meshsensitivity\finemesh\vortz_SSBLCLOSE';
% imageFiles = dir(fullfile(folder, 'Kcut_*_SSBLCLOSE.png'));


% Sort image files based on numeric suffix
[~, idx] = sort( str2double( regexp({imageFiles.name}, '\d+', 'match', 'once') ) );
imageFiles = imageFiles(idx);

% Initialize GIF and Video parameters
gifFilename = fullfile(folder, 'output_animation.gif');
videoFilename = fullfile(folder, 'output_video.avi'); % Or change to .mp4

% Create video writer
video = VideoWriter(videoFilename, 'Motion JPEG AVI'); % For .mp4 use 'MPEG-4'
% video.FrameRate = 5; % for SSBLCLOSE
video.FrameRate = 15;
% video.FrameRate = 30; % 2 fps = 0.5s per frame
% video.FrameRate = 45; % 2 fps = 0.5s per frame
open(video);

% Loop through images
for i = 1:length(imageFiles)
    % Read image
    img = imread(fullfile(folder, imageFiles(i).name));

    % Write to video
    writeVideo(video, img);

    % Write to GIF
    [A, map] = rgb2ind(img, 256); % Convert to indexed image
    if i == 1
        imwrite(A, map, gifFilename, 'gif', 'LoopCount', Inf, 'DelayTime', 1/video.FrameRate);
    else
        imwrite(A, map, gifFilename, 'gif', 'WriteMode', 'append', 'DelayTime', 1/video.FrameRate);
    end
end

% Close video writer
close(video);

disp('GIF and video creation complete!');
