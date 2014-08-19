function [ds, bs, I] = clipboxes(im, ds, bs)
% Clip detection windows to image the boundary.
%   [ds, bs, I] = clipboxes(im, ds, bs)
%
%   Any detection that is entirely outside of the image (i.e., it is entirely
%   inside the padded region of the feature pyramid) is removed.
%
% Return values
%   ds      Set of detection bounding boxes after clipping 
%           and (possibly) pruning
%   bs      Set of filter bounding boxes after clipping and
%           (possibly) pruning
%   I       Indicies of pruned entries in the original ds and bs 
%
% Arguments
%   im      Input image
%   ds      Detection bounding boxes (see pascal_test.m)
%   bs      Filter bounding boxes (see pascal_test.m)

if nargin < 3
  bs = [];
end

if ~isempty(ds)
  imh = size(im, 1);
  imw = size(im, 2);
  ds(:,1) = min(max(ds(:,1), 1), imw);
  ds(:,2) = min(max(ds(:,2), 1), imh);
  ds(:,3) = min(max(ds(:,3), 1), imw);
  ds(:,4) = min(max(ds(:,4), 1), imh);

  % remove invalid detections
  w = ds(:,3)-ds(:,1);
  h = ds(:,4)-ds(:,2);
  I = find((w == 0) | (h == 0));
  ds(I,:) = [];
  if ~isempty(bs)
    bs(I,:) = [];
  end
end
