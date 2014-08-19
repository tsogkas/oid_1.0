% pruneModel
% Replace model filters with their upper bounds contained in a filterTree 
% hierarchy. We first define a number of tree levels that will be pruned
% (nLevelsPruned). We remove all leaf-filters in the nLevelsPruned bottom
% tree levels and add new upper-bound filters. 
% 
%       model = pruneModel(model,filterTree,nLevelsPruned)
% 
% The model entries that have to be altered/removed are the following:
% - For every filter that is replaced with a new one, the only thing that
% has to change is the woffset in the corresponding blocks.
% - For every filter that will be entirely removed, we must remove the
% entries corresponding to itself and its symmetric version, as well as the
% corresponding symbols and rules. We must also update the relevant
% blocklabel entries in the model.rules structure. 
% 
% add:  indices of upper-bound filters in the filterTree struct that were 
%       added in the model
% indRemoved: indices of leaf node filters in the filterTree struct that 
%             were replaced with their upper bounds (since we are using
%             also flipped versions of every filter, the respective filter 
%             indices in the model struct are: 
%             indsInModelStruct = 2* indRemoved - 1;
% 
% Stavros Tsogkas <stavros.tsogkas@ecp.fr>
% Last update: June 2014

function model = pruneModel(model,filterTree,nLevelsPruned)

newModel = model;
remove   = [];
add      = [];
newModel.nLevelsMerged = nLevelsPruned;
if nLevelsPruned == 0
    newModel.added      = add;
    newModel.removed    = remove;
    newModel.filterTree = filterTree;
    model.mergedModel   = newModel;
    return
end
[nLevels, levels] = getTreeLevels(filterTree);
assert(nLevelsPruned>0 && rem(nLevelsPruned,1)==0,...
    'The number of levels to be pruned must be a non-negative integer')
assert(nLevelsPruned < nLevels,'Tree not deep enough. Choose a smaller pruning factor');

% Find nodes that will be removed
for i=1:nLevelsPruned   % for every level that will be pruned
    iLevel = nLevels-i+1;
    nNodes = numel(levels{iLevel});
    for iNode=1:nNodes         
        if isempty(filterTree(levels{iLevel}(iNode)).children) % remove leaf nodes
            remove = [remove, filterTree(levels{iLevel}(iNode)).id];
        end        
    end
end

% Find nodes that will be added
for i=1:numel(levels{nLevels-nLevelsPruned})
    if ~isempty(filterTree(levels{nLevels-nLevelsPruned}(i)).children)
        add = [add, filterTree(levels{nLevels-nLevelsPruned}(i)).id];
    end
end
add = sort(add);
indRemoved = remove;    % keep track of indices for removed leaves before sorting

% Calculate upper bounds for weight offsets
woffset   = -inf(size(add));
leafNodes = [];
for i=nLevels-nLevelsPruned+1:nLevels
    for j=1:length(levels{i})
        if isempty(filterTree(levels{i}(j)).children) % find all leaf nodes
            leafNodes = [leafNodes, filterTree(levels{i}(j)).id];
        end
    end
end
% Backtrack to find parent
for i=1:length(leafNodes)
    iLevel = 1;
    leaf = filterTree(leafNodes(i));
    currentNode = leaf;
    % Climb up the tree until you reach the node to be added or until you
    % reach the number of levels to be removed.
    while iLevel <= nLevelsPruned && ~ismember(currentNode.parent,add)
        currentNode = filterTree(currentNode.parent);
        iLevel = iLevel+1;
    end
    if ismember(currentNode.parent,add)
        indAdd = find(ismember(add, currentNode.parent));
        ind = 2*leaf.id - 1;
        blocklabel = model.rules{model.start}(ind).offset.blocklabel;
        woffset(indAdd) = max(woffset(indAdd), model.blocks(blocklabel).w);
    end
end

% Replace the first n = length(add) leaf nodes that are to be removed with
% the corresponding upper bound filters; then, remove the remaining leaf
% nodes.
blocksRemoved  = [];
symbolsRemoved = [];
rulesRemoved   = [];
remove         = sort(remove);
indFilter      = 2*remove-1;
for i=1:length(add)
    node = filterTree(add(i));
    sz = size(node.filter);
    ind = indFilter(i);
    newModel.blocks(newModel.filters(ind).blocklabel).w = node.filter(:);
    newModel.blocks(newModel.filters(ind).blocklabel).shape = sz;
    newModel.blocks(newModel.rules{newModel.start}(ind).offset.blocklabel).w = woffset(i);
    newModel.filters(ind).size   = [sz(1) sz(2)];
    newModel.filters(ind+1).size = [sz(1) sz(2)];
    newModel.rules{newModel.start}(ind).detwindow   = [sz(1) sz(2)];
    newModel.rules{newModel.start}(ind+1).detwindow = [sz(1) sz(2)];
end
remove(1:length(add))    = []; % The first n = length(add) entries have been replaced - no need to remove them.
indFilter(1:length(add)) = []; 
for i=1:length(remove)
    % Remove symbols and blocks corresponding to both the original filter
    % and its flipped version. The blocklabels for flipped filters are the
    % same as for the original ones, so no need to add something there.
    ind            = indFilter(i);
    symbolsRemoved = [symbolsRemoved, model.filters(ind).symbol,model.filters(ind+1).symbol];
    blocksRemoved  = [blocksRemoved,  model.filters(ind).blocklabel];
    rule           = model.rules{model.start}(ind);
    %  Remove rules
    while ~isempty(rule)
        rulesRemoved  = [rulesRemoved,  rule.rhs];
        symbolsRemoved = [symbolsRemoved, rule.rhs];
        blocksRemoved = [blocksRemoved, rule.blocks];
        rule          = model.rules{rule.rhs};
    end
    rule = model.rules{model.start}(ind+1);   % the same for flipped filter
    while ~isempty(rule)
        rulesRemoved = [rulesRemoved, rule.rhs];
        symbolsRemoved = [symbolsRemoved, rule.rhs];
        rule         = model.rules{rule.rhs};
    end
end
rulesRemoved        = unique(rulesRemoved);
symbolsRemoved      = unique(symbolsRemoved);
blocksRemoved       = unique(blocksRemoved);
newModel.numblocks  = newModel.numblocks  - length(blocksRemoved);
newModel.numsymbols = newModel.numsymbols - length(symbolsRemoved);
newModel.numfilters = newModel.numfilters - 2*length(remove);    
newModel.blocks(blocksRemoved)   = [];                % remove blocks
newModel.symbols(symbolsRemoved) = [];                % remove symbols
newModel.rules(rulesRemoved)     = [];                % remove rules
newModel.rules{newModel.start}([indFilter, indFilter+1]) = []; % remove filters
newModel.filters([indFilter, indFilter+1]) = [];
assert(newModel.numblocks  == length(newModel.blocks));
assert(newModel.numsymbols == length(newModel.symbols));
assert(newModel.numfilters == length(newModel.filters));    
% Re-direct block, symbol and filter indexes. This is necessary since
% intermediate blocks/rules/symbols have been removed. For each blocklabel  
% we find how many other blocks with a smaller index have been removed and
% subtract this number from the original value to get the new index.
for i=1:newModel.numfilters
    blocklabel = newModel.filters(i).blocklabel;
    blocklabel = blocklabel - sum(blocklabel>blocksRemoved);
    symbol     = newModel.filters(i).symbol;
    symbol     = symbol - sum(symbol>symbolsRemoved);
    newModel.filters(i).blocklabel = blocklabel;
    newModel.filters(i).symbol = symbol;
    newModel.symbols(symbol).filter = i;
end
for j=1:length(newModel.rules)
    for i=1:length(newModel.rules{j})
        offsetBlock = newModel.rules{j}(i).offset.blocklabel;
        offsetBlock = offsetBlock - sum(offsetBlock>blocksRemoved);
        locBlock = newModel.rules{j}(i).loc.blocklabel;
        locBlock = locBlock - sum(locBlock>blocksRemoved);
        if isfield(newModel.rules{j}(i),'def')  % structural rule
            defBlock = newModel.rules{j}(i).def.blocklabel;
            defBlock = defBlock - sum(defBlock>blocksRemoved);
            newModel.rules{j}(i).def.blocklabel = defBlock;
            assert(defBlock == offsetBlock+1);        
            assert(locBlock == defBlock+1);        
        else        % deformation rule
            defBlock = [];
            assert(locBlock == offsetBlock+1); 
        end
        lhs = newModel.rules{j}(i).lhs;
        rhs = newModel.rules{j}(i).rhs;
        newModel.rules{j}(i).lhs = lhs - sum(lhs>rulesRemoved);
        newModel.rules{j}(i).rhs = rhs - sum(rhs>rulesRemoved);
        newModel.rules{j}(i).i = i;
        newModel.rules{j}(i).offset.blocklabel = offsetBlock;
        newModel.rules{j}(i).loc.blocklabel = locBlock;
        newModel.rules{j}(i).blocks = [offsetBlock defBlock locBlock];
    end
end

% Calculate maxsize and minsize for model
minsize = [inf inf];
maxsize = [0 0];
for i=1:length(newModel.numfilters)
    minsize = min([minsize; newModel.filters(i).size]);
    maxsize = max([maxsize; newModel.filters(i).size]);
end
newModel.added      = add;
newModel.removed    = indRemoved;
newModel.filterTree = filterTree;
newModel.minsize    = minsize;
newModel.maxsize    = maxsize;

% Sanity checks
assert(all([newModel.symbols(:).filter] <= newModel.numfilters));
assert(all([newModel.filters(:).symbol] <= newModel.numsymbols));
assert(all([newModel.filters(:).blocklabel] <= newModel.numblocks));
assert(all([newModel.rules{newModel.start}(:).blocks] <= newModel.numblocks));
assert(all([newModel.rules{newModel.start}(:).rhs] <= length(newModel.rules)));
assert(length(newModel.rules{newModel.start}) == newModel.numfilters);
assert(length(newModel.rules) == length(newModel.symbols));
for i=2:length(newModel.rules)
    if ~isempty(newModel.rules{i})
        assert(newModel.rules{i}.lhs <= length(newModel.rules));
        assert(newModel.rules{i}.rhs <= length(newModel.rules));
    end
end
model.mergedModel = newModel;

