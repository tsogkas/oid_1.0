% getSetIndices   Returns linear indices of images belonging to input set
% 
%     [ind,setFlags] = getSetIndices(set,anno) 
% 
%   set: one out of {'train', 'trainval', 'val', 'test'}
%   anno: annotation .mat file
% 
% Stavros Tsogkas, <stavros.tsogkas@ecp.fr>
% Last update: June 2014


function [ind,setFlags] = getSetIndices(set,anno)

switch set
    case 'train'
        setFlags = anno.image.set == 1;
    case 'trainval'
        setFlags = anno.image.set == 1 | anno.image.set == 2;    
    case 'val'
        setFlags = anno.image.set == 2;
    case 'test'
        setFlags = anno.image.set == 3;
    otherwise
        error('Invalid test set')
end

ind = find(setFlags);