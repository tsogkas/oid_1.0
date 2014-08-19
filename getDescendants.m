% GETDESCENDANTS   Get all descendants of a node (all indices in the 
% subtree under the node)
% 
%   descendants = getDescendants(tree,node)
% 
% Stavros Tsogkas <stavros.tsogkas@ecp.fr>
% Last update: October 2013

function descendants = getDescendants(tree,node)

descendants = tree(node).children;
current = descendants;
while ~isempty(current)
    added = [];
    for i=1:length(current)
        added = [added, tree(current(i)).children];
    end
    current = added;
    descendants = [descendants added];
end

