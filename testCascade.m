% TESTCASCADE   Set up and evaluate an hierarchical tree cascade as it is 
% described in [1].
% 
% [1] Understanding Objects in Detail with Fine-grained Attributes (Vedaldi
% et al. CVPR 2014)
% 
%       [ap, recall, prec] = testCascade(model)
% 
% INPUT
%   model:  a trained DPM model (root-only)
%
% Stavros Tsogkas <stavros.tsogkas@ecp.fr> 
% Last update: June 2014

function [ap, recall, precision] = testCascade(model)

oidconf = oidConfig;
anno = jarLoadAnno;

% Align model and build filter tree
model      = alignModel(model);         
filterTree = buildFilterTree(model);    
nLevels    = getTreeLevels(filterTree);
model      = pruneModel(model,filterTree,nLevels-1);

% Cache empirical thresholds for each node
load(fullfile(oidconf.paths.cascadeDirectory, ...
    ['posExScores_' model.class '_pe' num2str(oidconf.pe) '.mat']))
t = getEmpiricalThresholds(totalScores, -0.5, levelScores, oidconf.empiricalPerc);
for i=1:length(t)
    model.mergedModel.filterTree(i+model.numfilters/2).thresh = t(i);
end

% Test model
[ap, recall, precision] = jar_test(model,anno,'aeroplane',oidconf.testSet,...
    oidconf.onlyTypical, model.note);


