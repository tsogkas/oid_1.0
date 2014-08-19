function [hists, im_vq, im_hists] = vq_pixels(im, ctrs, sbin)

debug = false;

% Rearrange so each pixel is a column
imc = im_to_col(im);

% Distance between each pixel and each ctr
D = vl_alldist2(ctrs, cast(imc, class(ctrs)));

% Center assignments
[~, A] = min(D, [], 1);

if nargout > 1
  im_vq = uint8(ctrs(:, A));
  im_vq = col_to_im(im_vq, size(im));

  subplot(1,3,1);
  imagesc(im); axis image;
  subplot(1,3,2);
  imagesc(im_vq); axis image;
end

hists = color_word_hists(col_to_im(A, size(im)), size(ctrs, 2), sbin);
hists = hists./repmat(sum(hists,3), [1 1 size(hists,3)]);

if nargout > 2
  im_hists = zeros([size(hists,1) size(hists,2) 3 size(hists,3)]);
  for i = 1:size(im_hists, 4)
    im_hists(:,:,:,i) = repmat(reshape(ctrs(:,i), [1 1 3]), ...
                               [size(hists,1) size(hists,2) 1]);
    im_hists(:,:,:,i) = im_hists(:,:,:,i) .* repmat(hists(:,:,i), [1 1 3]);
  end

  subplot(1,3,3);
  imagesc(uint8(sum(im_hists, 4))); axis image;
end
