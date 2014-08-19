function [ds, bs, trees,scales] = imgdetect(im, model, thresh, useCascade)
% Wrapper around gdetect.m that computes detections in an image.
%   [ds, bs, trees] = imgdetect(im, model, thresh)
%
% Return values (see gdetect.m)
%
% Arguments
%   im        Input image
%   model     Model to use for detection
%   thresh    Detection threshold (scores must be > thresh)
%   useUpperBound: use hierarchical filter tree to prune unpromising
%   locations (statsogk)

if nargin < 4, useCascade = false; end

im = color(im);
pyra = featpyramid(im, model);
scales = pyra.scales;
[ds, bs, trees] = gdetect(pyra, model, thresh, inf, useCascade);