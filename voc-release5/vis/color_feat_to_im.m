function im = color_feat_to_im(f, ctrs)

im = zeros([size(f,1) size(f,2) 3 size(f,3)]);
for i = 1:size(im, 4)
  im(:,:,:,i) = repmat(reshape(ctrs(:,i), [1 1 3]), ...
                       [size(f,1) size(f,2) 1]);
  im(:,:,:,i) = im(:,:,:,i) .* repmat(f(:,:,i), [1 1 3]);
end

im = uint8(sum(im, 4));
