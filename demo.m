%% Demo for cascade detection using hierarchical filter trees
%% Load model and hierarchical filter tree
startupOID;
oidconf = oidConfig;
part = 'nose';  % choose between 'nose' and 'verticalStabilizer'

load(fullfile(oidconf.paths.modelsDirectory,  [part,'_final']),'model');
load(fullfile(oidconf.paths.cascadeDirectory, ['filterTree_' part '_comp20']));
model       = alignModel(model);  % make sure model components are aligned (see inside function for details)
nLevels     = getTreeLevels(filterTree);
model       = pruneModel(model,filterTree,nLevels-1);

% Visualize hierarchical filter tree and construct merged model
visualizeFilterTree(filterTree,0);
disp('Full tree - press any key to continue to interactive tree visualization'); pause;

visualizeFilterTreeInteractive(filterTree);
disp('Press any key to continue'); pause;


%% Empirical thresholds

load(['posExScores_' part '_pe' num2str(oidconf.pe) '.mat'])
[t, info] = getEmpiricalThresholds(totalScores, -0.5, levelScores, 0.99);        
for i=1:length(t)
    model.mergedModel.filterTree(i+model.numfilters/2).thresh = t(i);
end

%% Test and time
im = imread('0534838.jpg');
for do_cascade = [0,1]
    [ds, bs, ~, ~] = imgdetect(im, model, model.thresh,do_cascade);
    [ds, bs, ~]    = clipboxes(im, ds, bs); I = nms(ds, 0.5);
    if do_cascade ==1   % bound cascade
        ds_csc = ds(I,:);
    else                % vanilla DPM
        ds_dpm = ds(I,:);
    end
end

%% Visualize detections
figure; imshow(im); drawBoxes(ds_dpm(:,1:4),ds_dpm(:,end)); title('Vanilla dpm top detections');
figure; imshow(im); drawBoxes(ds_csc(:,1:4),ds_csc(:,end)); title('Cascade dpm top detections');
