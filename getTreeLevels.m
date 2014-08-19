% GETTREELEVELS  Compute number of tree levels and return the nodes at each level 
% 
%   [n, levels] = getTreeLevels(tree)
%   [n, levels] = getTreeLevels(tree,rootIndex)
% 
% Stavros Tsogkas <stavros.tsogkas@ecp.fr>
% Last update: October 2013


function [n, levels] = getTreeLevels(tree, rootIndex)

if nargin < 2
    rootIndex = getRootPosition(tree);
end

levels{1} = rootIndex;
n = 1;
while ~isempty(levels{n})
    levels{n+1} = [];
    for j=1:numel(levels{n})
        levels{n+1} = [levels{n+1}, tree(levels{n}(j)).children];
    end
    n = n + 1;
end
levels(end) = [];
n = n - 1;



