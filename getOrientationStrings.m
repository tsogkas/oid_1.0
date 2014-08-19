% getOrientationStrings   Turn each vector of orientation probabilities
% into one of {'left','right','front','back'} strings. 
% 
%       [facingStrings, ignore] = getOrientationStrings(pos)
% 
% OUTPUT:
%   facingStrings: cell array containing viewpoint strings
%   ignore: indices of unreliable examples (examples that are assigned the
%   same 'left' or 'right' label for both the original and the flipped
%   version.
% 
% Stavros Tsogkas <stavros.tsogkas@ecp.fr>
% Last update: August 2014


function [facingStrings, ignore] = getOrientationStrings(pos)

orientationMatrix  = cat(2, pos(:).facing);
rightScore = orientationMatrix(1,:) + orientationMatrix(2,:) + orientationMatrix(8,:);
leftScore  = orientationMatrix(4,:) + orientationMatrix(5,:) + orientationMatrix(6,:);
frontScore = orientationMatrix(3,:);
backScore  = orientationMatrix(7,:);

scoreMatrix = [rightScore; leftScore; frontScore; backScore];
[~, argmax] = max(scoreMatrix);

facingStrings = arrayfun(@indOrient2str, argmax, 'UniformOutput',0);
dir1 = facingStrings(1:2:end-1);
dir2 = facingStrings(2:2:end);
ignore = 2*find((strcmp(dir1,'left') & strcmp(dir2,'left')) | ...
                (strcmp(dir1,'right') & strcmp(dir2,'right')));
ignore = sort([ignore ignore-1],'ascend');            
assert(all(strcmp(facingStrings(ignore(1:2:end-1)), facingStrings(ignore(2:2:end)))))


function facing = indOrient2str(indOrient)

switch indOrient
    case 1
        facing = 'right';    
    case 2
        facing = 'left';
    case 3
        facing = 'front';
    case 4
        facing = 'back';
    otherwise
        error('Invalid orientation index');
end
