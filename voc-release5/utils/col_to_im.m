function im = col_to_im(col, sz)
% im = col_to_im(col, sz)
%   Reshape a matrix of column pixels into an image
%   of size sz(1) x sz(2).

im = reshape(shiftdim(col, 1), [sz(1) sz(2) size(col,1)]);
