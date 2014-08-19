% getRootPosition   Get root index (tree is represented as a struct array)
% 
%   indRoot = getRootPosition(tree)
% 
% Stavros Tsogkas <stavros.tsogkas@ecp.fr>
% Last update: October 2013


function indRoot = getRootPosition(tree)

for i=1:length(tree)
    if ~isempty(tree(i)) && isempty(tree(i).parent)
        indRoot = tree(i).id;
        break
    end
end
