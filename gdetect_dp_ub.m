% GDETECT_DP_UB 
% Compute dynamic programming tables used for finding detections.
% 
%       model = gdetect_dp_ub(pyra, model)
%
%   This is essentially gdetect_dp, which is included in [1], extended with
%   the cascade for efficient detection described in [2].
% 
% 
% Return value
%   model   Object model augmented to store the dynamic programming tables
%
% Arguments
%   pyra    Feature pyramid returned by featpyramid.m
%   model   Object model
% 
%   [1] Discriminatively Trained Deformable Part Models, Release 5
%       http://people.cs.uchicago.edu/~rbg/latent-release5/
% 
%   [2] Understanding Objects in Detail with Fine-grained Attributes
% 
% Stavros Tsogkas <stavros.tsogkas@ecp.fr>
% Last update: June 2014



function model = gdetect_dp_ub(pyra, model)

% cache filter response
oidconf = oidConfig;
if oidconf.time
    tic; model = filter_responsesMex(model, pyra); toc;
else
    model = filter_responsesMex(model, pyra);
end

% compute detection scores
L = model_sort(model);
for s = L
  for r = model.rules{s}
    model = apply_rule(model, r, pyra.pady, pyra.padx);
  end
  model = symbol_score(model, s, pyra);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% compute score pyramid for symbol s
function model = symbol_score(model, s, pyra)
% model  object model
% s      grammar symbol

% take pointwise max over scores for each rule with s as the lhs
rules = model.rules{s};
score = rules(1).score;

for r = rules(2:end)
  for i = 1:length(r.score)
    score{i} = max(score{i}, r.score{i});
  end
end
model.symbols(s).score = score;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% compute score pyramid for rule r
function model = apply_rule(model, r, pady, padx)
% model  object model
% r      structural|deformation rule
% pady   number of rows of feature map padding
% padx   number of cols of feature map padding

if r.type == 'S'
  model = apply_structural_rule(model, r, pady, padx);
else
  model = apply_deformation_rule(model, r);
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% compute score pyramid for structural rule r
function model = apply_structural_rule(model, r, pady, padx)
% model  object model
% r      structural rule
% pady   number of rows of feature map padding
% padx   number of cols of feature map padding

% structural rule -> shift and sum scores from rhs symbols
% prepare score for this rule
score      = model.scoretpt;
bias       = model_get_block(model, r.offset) * model.features.bias; 
% bias       = model.mergedModel.blocks(2).w * model.features.bias; 
loc_w      = model_get_block(model, r.loc);
loc_f      = loc_feat(model, length(score));
loc_scores = loc_w * loc_f;
for i = 1:length(score)
  score{i}(:) = bias + loc_scores(i);
end

% sum scores from rhs (with appropriate shift and down sample)
for j = 1:length(r.rhs)
  ax = r.anchor{j}(1);
  ay = r.anchor{j}(2);
  ds = r.anchor{j}(3);
  % step size for down sampling
  step = 2^ds;
  % amount of (virtual) padding to halucinate
  virtpady = (step-1)*pady;
  virtpadx = (step-1)*padx;
  % starting points (simulates additional padding at finer scales)
  starty = 1+ay-virtpady;
  startx = 1+ax-virtpadx;
  % score table to shift and down sample
  s = model.symbols(r.rhs(j)).score;
  for i = 1:length(s)
    level = i - model.interval*ds;
    if level >= 1
      % ending points
      endy = min(size(s{level},1), starty+step*(size(score{i},1)-1));
      endx = min(size(s{level},2), startx+step*(size(score{i},2)-1));
      % y sample points
      iy = starty:step:endy;
      oy = sum(iy < 1);
      iy = iy(iy >= 1);
      % x sample points
      ix = startx:step:endx;
      ox = sum(ix < 1);
      ix = ix(ix >= 1);
      % sample scores
      sp = s{level}(iy, ix);
      sz = size(sp);
      % sum with correct offset
      stmp = -inf(size(score{i}));
      stmp(oy+1:oy+sz(1), ox+1:ox+sz(2)) = sp;
      score{i} = score{i} + stmp;
    else
      score{i}(:) = -inf;
    end
  end
end
model.rules{r.lhs}(r.i).score = score;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% compute score pyramid for deformation rule r
function model = apply_deformation_rule(model, r)
% model  object model
% r      deformation rule

% deformation rule -> apply distance transform
def_w      = model_get_block(model, r.def);
score      = model.symbols(r.rhs(1)).score;
bias       = model_get_block(model, r.offset) * model.features.bias;
loc_w      = model_get_block(model, r.loc);
loc_f      = loc_feat(model, length(score));
loc_scores = loc_w * loc_f;
for i = 1:length(score)
  score{i} = score{i} + bias + loc_scores(i);
  % Bounded distance transform with +/- 4 HOG cells (9x9 window)
  [score{i}, Ix{i}, Iy{i}] = bounded_dt(score{i}, def_w(1), def_w(2), ...
                                        def_w(3), def_w(4), 4);
  % Unbounded distance transform
  %[score{i}, Ix{i}, Iy{i}] = dt(score{i}, def_w(1), def_w(2), ...
  %                              def_w(3), def_w(4));
end
model.rules{r.lhs}(r.i).score = score;
model.rules{r.lhs}(r.i).Ix    = Ix;
model.rules{r.lhs}(r.i).Iy    = Iy;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function model = filter_responsesMex(model, pyra)
% model    object model
% pyra     feature pyramid


% Gather filters and their flipped versions
oidconf     = oidConfig;
tree        = model.mergedModel.filterTree;
treeLength  = length(tree);
filters     = cell(treeLength, 1);
filtersFlip = cell(treeLength, 1);
varCell     = cell(treeLength, 1);  % the entries corresponding to leaves will be empty
varCellFlip = cell(treeLength, 1);  % this is normal
for i = 1:treeLength
    filters{i}     = single(tree(i).filter);
    varCell{i}     = 32*single(tree(i).varCell); % we multiply by 32 to calculate the Chebysev bound
    filtersFlip{i} = flipfeat(filters{i});
    varCellFlip{i} = fliplr(varCell{i});
end

% Prepare cascade arguments
[nLevels,levels] = getTreeLevels(tree,treeLength);
opts.pe          = oidconf.pe;
opts.thresh      = oidconf.cascadeThresh;
opts.nodeThresh  = oidconf.threshPerNode;
opts.rootSize    = int32(tree(end).shape);
opts.nLevels     = int32(nLevels);
opts.nNodes      = int32(cellfun(@length,levels));
opts.levels      = cellfun(@int32,levels,'UniformOutput',false);
opts.isleaf      = cellfun(@isempty,{tree(:).children});
opts.filters     = filters;
opts.filtersFlip = filtersFlip;
opts.varCell     = varCell;
opts.varCellFlip = varCellFlip;
opts.tree        = tree;
opts.treeLength  = int32(treeLength);
opts.numFilters  = model.numfilters;
% Cast into int32 for use in mex file
for i = 1:length(opts.tree)
    opts.tree(i).shape          = int32(opts.tree(i).shape);
    opts.tree(i).parent         = int32(opts.tree(i).parent);
    opts.tree(i).relativeShift  = int32(opts.tree(i).relativeShift);
    opts.tree(i).level          = int32(opts.tree(i).level);
    opts.tree(i).offsetFromRoot = int32(opts.tree(i).offsetFromRoot);
    opts.tree(i).children       = int32(opts.tree(i).children);
    opts.tree(i).nLeaves        = int32(nnz(opts.isleaf(getDescendants(tree,i))));   % number of descendants that are leaves
end
feat        = pyra.feat;
nScales     = length(feat);
for scale = 1:nScales     % compute scores at all scales
    if ~pyra.valid_levels(scale)
        % not processing this level, so set default values
        model.scoretpt{scale} = 0;
        for i = 1:model.numfilters
            model.symbols(model.filters(i).symbol).score{scale} = -inf;
        end
        continue;
    end
    
    [ub,ubFlip] = cascadeHFT(feat{scale},opts);
    [ub,ubFlip] = fillBoundaryValues(ub,ubFlip,opts);
    %         [diff, diffFlip] = cascadeSanityCheck(ub,ubFlip,feat{scale},model);
        
    % set filter response as the score for each filter terminal
    for  i  = 1:model.numfilters
        fsym = model.filters(i).symbol;
        if model.filters(i).flip
            model.symbols(fsym).score{scale} = double(ubFlip(:,:,i/2));
        else
            model.symbols(fsym).score{scale} = double(ub(:,:,(i+1)/2));
        end
    end
    model.scoretpt{scale} = zeros(size(feat{scale},1),size(feat{scale},2));
    
    % Used for computing empirical cascade thresholds
    if oidconf.storeFullScores
        model.fullCascadeScores{scale} = ub;
        model.fullCascadeScoresFlip{scale} = ubFlip;
    end
end


% -------------------------------------------------------------------------
% Fill in values to account for offset of children from root (this will
% give slightly different results near the right and bottom boundaries
% with respect to the vanilla dpm)
function [ub,ubFlip] = fillBoundaryValues(ub,ubFlip,opts)

rootSize = opts.rootSize;
tree     = opts.tree;
featSize = int32(size(ub));
for i=1:(opts.numFilters/2)
    offset    = tree(i).offsetFromRoot;
    fillStart = featSize(1:2)-rootSize(1:2)+offset;
    fillEnd   = featSize-tree(i).shape + 1; fillEnd = fillEnd(1:2);
    % repmat equivalent but faster
    tmp = ub(:,offset(2)+1,i); ub(:,1:offset(2),i) = tmp(:, ones(1,offset(2))); 
    tmp = ub(offset(1)+1,:,i); ub(1:offset(1),:,i) = tmp(ones(offset(1),1),:);
    tmp = ub(:,fillStart(2)-1, i); ub(:,fillStart(2):fillEnd(2) ,i) = tmp(:,ones(1, fillEnd(2)-fillStart(2)+1));
    tmp = ub(fillStart(1)-1,:, i); ub(fillStart(1):fillEnd(1),:,i)  = tmp(ones(fillEnd(1)-fillStart(1)+1, 1),:);
    % Same for flipped-filters scores
    offset    = tree(i).offsetFromRoot;
    offset(2) = rootSize(2) - tree(i).shape(2) - tree(i).offsetFromRoot(2);
    fillStart = featSize(1:2)-rootSize(1:2)+offset;
    fillEnd   = featSize-tree(i).shape + 1; fillEnd = fillEnd(1:2);
    tmp = ubFlip(:,offset(2)+1,i); ubFlip(:,1:offset(2),i) = tmp(:, ones(1,offset(2)));
    tmp = ubFlip(offset(1)+1,:,i); ubFlip(1:offset(1),:,i) = tmp(ones(offset(1),1),:);
    tmp = ubFlip(:,fillStart(2)-1, i); ubFlip(:,fillStart(2):fillEnd(2) ,i) = tmp(:,ones(1, fillEnd(2)-fillStart(2)+1));
    tmp = ubFlip(fillStart(1)-1,:, i); ubFlip(fillStart(1):fillEnd(1),:,i)  = tmp(ones(fillEnd(1)-fillStart(1)+1, 1),:);
end


% --- Make sure results are the same as in the original dpm (cascade
% threshold must be set to -inf) ------------------------------------------
function [diff,diffFlip] = cascadeSanityCheck(ub,ubFlip,feat,model)

% Make sure results are the same with original dpm
filtersPedro = cell(model.numfilters, 1);
for i = 1:model.numfilters
    filtersPedro{i} = single(model_get_block(model, model.filters(i)));
end
exactScores = fconvsse_ST(feat,filtersPedro,1,length(filtersPedro));
featSize = size(feat);
for i = 1:length(exactScores)
    exactScores{i} = padarray(exactScores{i},featSize(1:2) - size(exactScores{i}),-inf,'post');
end

% Numerical check
diff = zeros(model.numfilters/2,1);
diffFlip = zeros(model.numfilters/2,1);
ub(isinf(ub)) = 0; ubFlip(isinf(ubFlip)) = 0;
for i=1:length(exactScores), exactScores{i}(isinf(exactScores{i})) = 0; end
for i=1:model.numfilters/2
    diff(i) = norm(ub(:,:,i) - exactScores{2*i-1});
    diffFlip(i) = norm(ubFlip(:,:,i) - exactScores{2*i});
end

