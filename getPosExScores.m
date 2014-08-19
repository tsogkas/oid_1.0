% getPosExScores
% Collect positive examples scores per hierarchy level in the filter tree. 
% These scores can be used to derive empirical thresholds per node or per 
% tree level.
% 
%       [levelScores, totalScores] = getPosExScores(part)
% 
% Stavros Tsogkas <stavros.tsogkas@ecp.fr>
% Last update: June 2014

function [levelScores, totalScores] = getPosExScores(part)

partList = {'aeroplane','verticalStabilizer','nose','wing','wheel'};
assert(ismember(part,partList),'Invalid aeroplane part');

% Load the training data
oidconf = oidConfig();

% positive examples data
temp = load(fullfile(oidconf.paths.dataDirectory, [part '_jar_data_from_aeroplane_typical_trainval.mat']));
pos = temp.pos;

% Load hierarchical filter tree and prune model
temp = load(fullfile(oidconf.paths.cascadeDirectory, ['filterTree_' part '_comp20.mat']));
filterTree = temp.filterTree;
temp  = load(fullfile(oidconf.paths.modelsDirectory, [part '_final.mat']));
model = temp.model;
model = alignModel(model);
[nLevels, levels] = getTreeLevels(filterTree);
model = pruneModel(model,filterTree,nLevels-1);
class = model.class;

pixels  = model.minsize * model.sbin / 2;
minsize = prod(pixels);
nPos = length(pos);
pars = cell(nPos, 1);
useCascade = true;
isleaf = cellfun(@isempty,{filterTree(:).children});
nLeaves = nnz(isleaf);
nInternalNodes = nnz(~isleaf);
levelScores = zeros(nInternalNodes, nPos);
% levelScores = zeros(nLevels-1,nPos);
parfor i = 1:nPos
    pars{i}.scores           = [];
    pars{i}.total_scores     = [];
    fprintf('%s %s: cascade data: %d/%d\n', procid(), class, i, nPos);
    bbox = pos(i).boxes;
    % skip small examples
    if (bbox(3)-bbox(1)+1)*(bbox(4)-bbox(2)+1) < minsize
        continue;
    end
    % get example
    [~,fileName,ext] = fileparts(pos(i).im);
    pos(i).im = [fileName ext];
    im = imreadx(pos(i));
    [im, bbox] = croppos(im, bbox);
    [pyra, model_dp] = gdetect_pos_prepare(im, model, bbox, 0.5,useCascade);
    [ds, bs, trees] = gdetect_pos(pyra, model_dp, 1, 1, 0.5);
    if ~isempty(ds)
        % collect cascade score statistics
        pars{i}.scores = get_block_scores(pyra, model, trees);
        pars{i}.total_scores = ds(1,end);
        %         levelScores(:,i) = backtrackHierarchyLevel(model_dp,trees,filterTree);
        levelScores(:,i) = backtrackHierarchyNode(model_dp,trees,filterTree,nInternalNodes,nLeaves);
    end
end
% Collate
pars = cell2mat(pars);
scores = cat(1, pars(:).scores);

% Sanity check that detection scores match sum of block scores
% computed in get_block_scores
totalScores = cat(1, pars(:).total_scores);
assert(max(abs(scores(:,1) - totalScores)) < 1e-5);

save(fullfile(oidconf.paths.cascadeDirectory, ...
    ['posExScores_' part '_pe' num2str(oidconf.pe) '.mat']),...
    'totalScores','scores','pars','levelScores');


% --- Backtrack in the hierarchical filter tree responses to get detection
% scores at every level of the cascade (thresholds per node) --------------
function levelScores = backtrackHierarchyNode(model,detectionTrees,filterTree,nInternalNodes, nLeaves)

s = tree_mat_to_struct(detectionTrees{1});
s = s(1);
scale = s.l;
flip = model.filters(s.rule_index).flip;
% Get full array of cascade upper bounds at given detection scale (these
% arrays have been precomputed inside gdetect_dp_ub.m)
if flip
    scores = model.fullCascadeScoresFlip{scale};
    ind = s.rule_index / 2;
else
    scores = model.fullCascadeScores{scale};
    ind = (s.rule_index + 1) / 2;
end
% make sure that score is the sum of the compute score + bias
offset = model_get_block(model, model.rules{1}(s.rule_index).offset);
assert(abs(scores(s.y, s.x, ind) + offset*model.features.bias - s.score) < 10e-5);

levelScores = inf(nInternalNodes,1);   % vector of scores at all cascade levels
node = filterTree(ind);
x = s.x; y = s.y;   % initialize coordinates
while ~isempty(node.parent)
    parent = filterTree(node.parent);
    offset = node.relativeShift;
    y = y - offset(1);
    % Adjust coordinates for parent node
    if flip
        x = x - parent.shape(2) + offset(2) + node.shape(2);
    else
        x = x - offset(2);
    end
    assert(max(max(scores(:,:,parent.id) == scores(y,x,parent.id))));
    levelScores(parent.id - nLeaves) = scores(y,x,parent.id);
    node = parent;
end

% --- Backtrack in the hierarchical filter tree responses to get detection
% scores at every level of the cascade. (Thresholds per hierarchy level) --
function levelScores = backtrackHierarchyLevel(model,detectionTrees,filterTree)

s = tree_mat_to_struct(detectionTrees{1});
s = s(1);
scale = s.l;
flip = model.filters(s.rule_index).flip;
% Get full array of cascade upper bounds at given detection scale (these
% arrays have been precomputed inside gdetect_dp_ub.m)
if flip
    scores = model.fullCascadeScoresFlip{scale};
    ind = s.rule_index / 2;
else
    scores = model.fullCascadeScores{scale};
    ind = (s.rule_index + 1) / 2;
end
% make sure that score is the sum of the compute score + bias
offset = model_get_block(model, model.rules{1}(s.rule_index).offset);
assert(abs(scores(s.y, s.x, ind) + offset*model.features.bias - s.score) < 10e-5);

nLevels = getTreeLevels(filterTree);
levelScores = inf(nLevels-1,1);   % vector of scores at all cascade levels
node = filterTree(ind);
x = s.x; y = s.y;   % initialize coordinates
while ~isempty(node.parent)
    parent = filterTree(node.parent);
    offset = node.relativeShift;
    y = y - offset(1);
    % Adjust coordinates for parent node
    if flip
        x = x - parent.shape(2) + offset(2) + node.shape(2);
    else
        x = x - offset(2);
    end
    assert(max(max(scores(:,:,parent.id) == scores(y,x,parent.id))));
    levelScores(parent.level) = scores(y,x,parent.id);
    node = parent;
end
