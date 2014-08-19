%provides the list of points in the boundary of polygon
function boundary = poly2boundary(pts)
    pts = round(pts);
    pts(end+1,:) = pts(1,:);
    boundary = zeros(0,2);
    for i = 1:size(pts,1)-1
        [x,y] = bresenham(pts(i,1), pts(i,2), pts(i+1,1), pts(i+1,2));
        boundary = [boundary; x(1:end-1) y(1:end-1)];
    end
end