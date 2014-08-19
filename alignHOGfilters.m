% ALIGNHOGFILTERS Align two HOG filters of different dimensions. We slide 
% one template over the other at all possible positions and we calculate 
% the sum of the distances of both templates from their mean. We place the 
% second template at the position that minimizes this sum of distances.
% 
%   aligned = alignHOGfilters(w1,w2)
% 
% Stavros Tsogkas, <stavros.tsogkas@ecp.fr>
% Last update: June 2014

function aligned = alignHOGfilters(w1,w2)

size1 = size(w1);
size2 = size(w2);
if isequal(size1,size2)
    upperBound = (w1+w2)/2;
    distance1  = sum(abs(w1(:) - upperBound(:)));
    distance2  = sum(abs(w2(:) - upperBound(:)));
    minDist    = distance1 + distance2;
    w1padded   = w1;
    w2shifted  = w2;
    ishift     = 0;
    jshift     = 0;
    iRelativeShift = 0;
    jRelativeShift = 0;
else
    % Minimum padding needed in case filters don't have the same size
    w1padded   = padarray(w1,[size2(1),size2(2),0],0,'both');
    newSize    = size(w1padded);
    w2padded   = padarray(w2,[newSize(1)-size2(1), newSize(2)-size2(2),0],0,'post');
    minDist    = inf;
    % Find overlap that gives minimum sum of distances from the upper bound
    for i=1:newSize(1)-1
        for j=1:newSize(2)-1
            w2shifted  = circshift(w2padded,[i,j]);
            upperBound = (w1padded + w2shifted)/2;
            distance1  = sum(abs(w1padded(:)  - upperBound(:)));
            distance2  = sum(abs(w2shifted(:) - upperBound(:)));
            totalDistance = distance1+distance2;
            if totalDistance < minDist
                minDist = totalDistance;
                ishift = i;
                jshift = j;
            end
        end
    end
    w2shifted  = circshift(w2padded,[ishift,jshift]);
    upperBound = (w1padded + w2shifted)/2;
    % relative positions of second filter with respect to the reference filter
    iRelativeShift = ishift - size2(1);
    jRelativeShift = jshift - size2(2);
end
aligned.w1     = w1padded;
aligned.w2     = w2shifted;
aligned.dist   = minDist;
aligned.shift  = [ishift jshift];
aligned.relativeShift = [iRelativeShift jRelativeShift];
aligned.upperBound          = upperBound;
aligned.upperBoundCropped   = getFilterSupport(upperBound);
aligned.e        = w1padded-upperBound;
aligned.eCropped = getFilterSupport(aligned.e);