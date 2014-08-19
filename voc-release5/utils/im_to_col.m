function col = im_to_col(im)
% col = im_to_col(im)
%   Convert an MxNxC image into a Cx(M*N) matrix where each column
%   is a pixel and each row is a color channel

sz = size(im);
col = reshape(shiftdim(im, 2), [sz(3) sz(1)*sz(2)]);
