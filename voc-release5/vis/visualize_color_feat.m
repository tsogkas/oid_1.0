function im = visualize_color_feat(f, ctrs)

im = color_feat_to_im(f, ctrs);

if nargout == 0
  imagesc(im); 
  axis image;
end
