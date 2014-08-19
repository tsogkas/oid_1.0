% BUILDFILTERTREE  Build hierarchical filter tree, based on sequential
% greedy clustering of HOG filter pairs, using their distance from their
% aligned mean.
%
%   filterTree = buildFilterTree(model)
%
%   model: a DPM model (e.g. trained using [1]).
%
%   [1] Discriminatively Trained Deformable Part Models, Release 5
%       http://people.cs.uchicago.edu/~rbg/latent-release5/
%
% Stavros Tsogkas <stavros.tsogkas@ecp.fr>
% Last update: June 2014

function filterTree = buildFilterTree(model)

oidconf  = oidConfig;
% Reshape HOG templates
wReshaped = reshapeHOGweights(model);
nModels   = length(wReshaped);
dmPath = ['distance_matrix_' model.class '_' model.note ...
    '_comp' num2str(model.numfilters/2) '.mat'];

% Calculate total distance of every HOG pair from its aligned mean
try
    load(fullfile(oidconf.paths.cascadeDirectory,dmPath));
catch
    distance = diag(inf(nModels,1));
    parfor i=1:nModels
        disp(['Align template ' num2str(i)])
        w1       = wReshaped{i};
        tempdist = distance(i,:);
        for j=i+1:nModels
            w2          = wReshaped{j};
            aligned     = alignHOGfilters(w1,w2);
            tempdist(j) = aligned.dist;
        end
        distance(i,:)   = tempdist;
    end
    % Make matrices symmetric
    distance = distance + distance';
    save(fullfile(oidconf.paths.cascadeDirectory, dmPath), 'distance','model');
end


% Initialization of filter tree structure
filterTree(1:2*nModels) = struct('children',[],'parent',[],'id',[],'filter',[]);
for i=1:nModels
    filterTree(i).filter = wReshaped{i};
    filterTree(i).mergeDegree = 0; % merging level
    filterTree(i).shape = size(wReshaped{i});
end
for i=1:length(filterTree)
    filterTree(i).id = i;
end

% Greedily select next pair of HOG filters that will be merged and replace
% that pair of entries with the upper bound HOG filter until the root
% ueber-node has been calculated. During this process, entries
% corresponding to col indices are removed and entries corresponding to row
% indices are replaced with new data. When possible, we make sure that all
% nodes at depth k are produced by merging pairs of nodes at depth k+1.

ftPath = ['filterTree_' model.class '_' model.note ...
    '_comp' num2str(model.numfilters/2) '.mat'];
try
    load(fullfile(oidconf.paths.cascadeDirectory, ftPath), 'filterTree');
catch
    reachedRoot  = false;
    position     = nModels+1;
    idArray      = 1:nModels; % array that holds the current tree position for every filter in the (dynamic) cell array wReshaped
    currentLevel = 0;
    indCurrentLevel = [];
    while ~reachedRoot && (position <= 2*nModels)
        indCurrentLevel = [find([filterTree(:).mergeDegree] == currentLevel) indCurrentLevel]; % nodes remaining to be merged at currentLevel
        while length(indCurrentLevel) >= 2
            disp(['Creating merged filter at position ' num2str(position)]);
            % Find pair of nodes with minimum distance
            minDist = inf;
            for j=1:length(indCurrentLevel)
                for k=1:length(indCurrentLevel)
                    trow = find(indCurrentLevel(j) == idArray,1);
                    tcol = find(indCurrentLevel(k) == idArray,1);
                    assert(~isempty(trow) && ~isempty(tcol));
                    if distance(trow,tcol) < minDist
                        minDist = distance(trow,tcol);
                        row     = trow;
                        col     = tcol;
                        remove  = [j,k];
                    end
                end
            end
            indCurrentLevel(remove) = []; % remove merged leaf nodes from current level list
            aligned = alignHOGfilters(wReshaped{row},wReshaped{col});
            filterTree(position).filter      = aligned.upperBoundCropped;
            filterTree(position).shape       = size(aligned.upperBoundCropped);
            filterTree(position).eCropped    = aligned.eCropped;
            filterTree(position).children(1) = idArray(row);    % filters that were used to produce new upper bound
            filterTree(position).children(2) = idArray(col);
            filterTree(position).mergeDegree = max(filterTree(idArray(row)).mergeDegree,...
                filterTree(idArray(row)).mergeDegree) + 1;
            filterTree(idArray(row)).parent  = position;
            filterTree(idArray(col)).parent  = position;
            filterTree(idArray(row)).relativeShift  = [0 0];
            filterTree(idArray(col)).relativeShift  = aligned.relativeShift;
            assert(filterTree(position).children(1) == idArray(row));
            assert(filterTree(position).children(2) == idArray(col));
            idArray(row)   = position;  % update idArray with position of added filter
            wReshaped{row} = aligned.upperBoundCropped; % replace one filter with the pair upper bound
            w1 = aligned.upperBoundCropped;
            tempdist = distance(row,:);
            parfor i=1:length(wReshaped)
                if i~=row
                    w2 = wReshaped{i};
                    aligned = alignHOGfilters(w1,w2);
                    tempdist(i) = aligned.dist;
                end
            end
            distance(row,:) = tempdist;
            distance(:,col) = [];   % remove row and column that correspond to the removed filter
            distance(col,:) = [];   % CAUTION!: first update values in distance matrix and THEN remove entries
            wReshaped(col)  = [];   % remove one entry from filter array
            idArray(col)    = [];
            position = position+1;  % increase position of next merged filter in the tree
            if numel(distance) == 1
                reachedRoot = true;
            end
        end
        currentLevel = currentLevel + 1;
    end
    % The level of each node is its depth (distance from root) + 1
    [nLevels,levels] = getTreeLevels(filterTree);
    for i=1:nLevels
        for j=1:length(levels{i})
            filterTree(levels{i}(j)).level = i;
        end
    end
    filterTree(position-1).relativeShift = [0 0];
    filterTree(position:end) = [];  % this way the root is always the last element of the array
    filterTree = computeNodeVariances(filterTree);
    
    save(fullfile(oidconf.paths.cascadeDirectory,ftPath),'filterTree','wReshaped');
end


% --- Reshape all filters in a h x w x k form (k is typically 32). --------
function wReshaped = reshapeHOGweights(model)
nModels = length(model.filters)/2;
wReshaped = cell(nModels,1);
for i=1:nModels
    ind = 2*i -1;
    indBlock = model.filters(ind).blocklabel;
    wReshaped{i} = reshape(model.blocks(indBlock).w,model.blocks(indBlock).shape);
end



% --- Compute maximum variance for every HOG cell of every node in the ----
% hierarchical filter tree, based on the error between the said node and
% its leaf descendants.
function filterTree = computeNodeVariances(filterTree)

% Align HOG filters so that their upper-left points match.
maxSize    = [-inf -inf];
nLeaves    = sum([filterTree(:).mergeDegree]==0);
treeLength = length(filterTree);
for i = 1:nLeaves
    maxSize = max(maxSize, [size(filterTree(i).filter,1), size(filterTree(i).filter,2)]);
end
alignedFilters = cell(treeLength,1);
for i = 1:treeLength
    temp = padarray(filterTree(i).filter, [maxSize(1) maxSize(2)], 0, 'pre');
    temp = padarray(temp, [3*maxSize(1) - size(temp,1) 3*maxSize(2) - size(temp,2)], 0, 'post');
    alignedFilters{i} = temp;
end

% Adjust each node's relative offset. Originally only the right child of
% each node has a non-zero offset from its parent. Since all filters have
% to be aligned so that upper-left points match, if this offset is
% negative, we shift the left child instead.
for i = 1:treeLength - 1
    parent = filterTree(i).parent;
    ch1 = filterTree(parent).children(1);
    if filterTree(i).relativeShift(1) < 0 && i ~= ch1
        filterTree(ch1).relativeShift(1) = -filterTree(i).relativeShift(1);
        filterTree(i).relativeShift(1) = 0;
    end
    if filterTree(i).relativeShift(2) < 0 && i ~= ch1
        filterTree(ch1).relativeShift(2) = -filterTree(i).relativeShift(2);
        filterTree(i).relativeShift(2) = 0;
    end
end

% Compute leaf nodes' offset with respect to the root. For every
% intermediate node, the offset relative to the root is the sum of the
% offset from its immediate parent and the parent offset relative to the
% root.
offsetArray = zeros(treeLength,2);
for i = 1:treeLength
    current = i;
    parent  = filterTree(current).parent;
    offset  = filterTree(current).relativeShift;
    while ~isempty(parent)
        offset  = offset + filterTree(parent).relativeShift;
        current = parent;
        parent  = filterTree(current).parent;
    end
    offsetArray(i,:) = offset;
    filterTree(i).offsetFromRoot = offset;
end

% Shift all filters with respect to the root
shiftedFilters = alignedFilters;
for i = 1:treeLength-1
    shiftedFilters{i} = circshift(alignedFilters{i},offsetArray(i,:));
end

% Make sure that all intermediate nodes up to the root can be reconstructed
% by the leaf nodes and calculate maximum variance for every HOG cell.
varCell = cell(treeLength-nLeaves,1);
for i = nLeaves+1:treeLength;
    descendants = getDescendants(filterTree,i);
    leaves      = descendants([filterTree(descendants).mergeDegree] == 0);
    rootFilter  = zeros(size(shiftedFilters{i}));
    for j = leaves
        rootFilter = rootFilter + shiftedFilters{j}/2^(filterTree(j).level - filterTree(i).level);
    end
    assert(sum((rootFilter(:) - shiftedFilters{i}(:)).^2) < eps);
    varCell{i-nLeaves} = zeros(size(shiftedFilters{i},1),size(shiftedFilters{i},2));
    for j = leaves
        varCell{i-nLeaves} = max(varCell{i-nLeaves},...
            sum((shiftedFilters{j}-shiftedFilters{i}).^2,3)/32);
    end
    filterTree(i).varCell = getFilterSupport(varCell{i-nLeaves});
end