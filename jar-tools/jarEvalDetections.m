function [ap, recall, prec] = jarEvalDetections(det, anno)
% [ap, recall, prec] = jarEvalDetections(det, anno)
%
% Candidate detections are stored in the det structure:
%   det.imageSet
%   det.objType = element from anno.objTypes
%   det.box     = [ ... ] 4 x N (x1, y1, x2, y2) in PASCAL format
%   det.score   = [ ... ] 1 x N confidence
%   det.imageId = [ ... ] 1 x N source image id
% Optional (for scoring attribute classification):
%   det.attrType  = attribute field anno.(objType).(attrType)
%   det.attrLabel = attribute value being classified
%
% anno is the annotations structure
% tmp = load('~/oid_project/oid/results/nose/nose_aspect ratio clustering with k = 6/nose_comp6_boxes_test.mat');
% gt = tmp.gt;
attrVoteThresh = 0.8;
imageIds = jarReadImageSet(anno, det.objClass, det.imageSets, det.onlyTypical);
nImages = length(imageIds);
usingAttributes = isfield(det, 'attrType') & isfield(det, 'attrLabel');

det.processed = false(1, length(det.imageId));

for i = 1:nImages
  imageId = imageIds(i);
  detInds = find(det.imageId == imageId);
  detBoxes = det.box(:, detInds);
  detScores = det.score(detInds);

  % evalDetections requires sorted inputs
  [~, ord] = sort(detScores, 'descend');
  detBoxes = detBoxes(:, ord);
  detScores = detScores(ord);
  detInds = detInds(ord);

  % Get ground-truth boxes
  gtInds = find(anno.(det.objType).imageId == imageId);
  if usingAttributes
    % Reduce ground-truth set to only those that match the attribute type 
    % and label being evaluated
    attrInds = find(anno.(det.objType).attribute.([det.attrType 'Label']) ...
                    == det.attrLabel);
    gtInds = intersect(gtInds, attrInds);
  end
  gtBoxes = anno.(det.objType).box(:, gtInds);
  gtDifficult = anno.(det.objType).isDifficult(gtInds);
%   if ~isequal(gtBoxes',gt(i).BB)
%       keyboard
%   end
  if usingAttributes
    % Difficult if either the object or the attribute is difficult
    gtDifficult = gtDifficult | ...
                  anno.(det.objType).attribute.([det.attrType 'IsDifficult'])(gtInds);
  end

  matches(i) = evalDetections(gtBoxes, gtDifficult, detBoxes, detScores);
  det.processed(detInds) = true;
end

assert(all(det.processed));

labels = cat(2, matches(:).labels);
scores = cat(2, matches(:).scores);

[recall, prec, info] = vl_pr(labels, scores);
% PASCAL >= 2010 style AP: area under the tighest 
% piecewise constant upperbound on the actual AP
ap = averagePrecision(recall, prec);
