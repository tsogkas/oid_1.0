% splitLeftRight   Split components into left-right facing clusters based 
% on viewpoint information.
% 
%       [indLeft, indRight] = splitLeftRight(pos, useFrontBack)
% 
% INPUT:
%   pos: struct containing the training example information
%   useFrontBack: use examples facing front/back {true,false}
% 
% OUTPUT:
%   indLeft: indices of left-facing examples
%   indRight: indices of right-facing examples
% 
% Stavros Tsogkas <stavros.tsogkas@ecp.fr>
% Last update: August 2014

function [indLeft, indRight] = splitLeftRight(pos, useFrontBack)

if nargin < 2, useFrontBack = 1; end

indLeft  = find(strcmp('left' ,{pos(:).facingString}));
indRight = find(strcmp('right',{pos(:).facingString}));
if useFrontBack
    indBack  = find(strcmp('back',{pos(:).facingString}));
    indFront = find(strcmp('front',{pos(:).facingString}));
    indLeft  = sort([indLeft,  indFront(1:2:end-1) indBack(1:2:end-1)], 'ascend');
    indRight = sort([indRight, indFront(2:2:end) indBack(2:2:end)], 'ascend');
end