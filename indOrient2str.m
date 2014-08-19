% indOrient2str
% Turns [~, argmax] = max(orientation_vector) from fgvc annotations to
% left/right/frontal/back facing direction string (deprecated).
% 
%   facing = indOrient2str(indOrient)
% 
% Stavros Tsogkas <stavros.tsogkas@ecp.fr>
% Last update: June 2014

function facing = indOrient2str(indOrient)

switch indOrient
    case {1,2,8}
        facing = 'right';    
    case {4,5,6}
        facing = 'left';
    case 3
        facing = 'front';
    case 7
        facing = 'back';
    otherwise
        error('Invalid orientation index');
end
