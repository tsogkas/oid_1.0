function [ap, prec, recall] = jarEvalLayouts(layout, anno)
% [ap, prec, recall] = jarEvalLayouts(layout, anno)
%   jarEvalLayouts      - evaluate performance for the layout challenge
%   layout.imageSet     - imageSet to evaluate
%   layout.objTypes     - list of object types being detected
%   layout.scores       - 1 X L scores of each layout 
%   layout.imageId      - 1 X L imageId for each of the layout
%   layout.box          - 4 X N listing [x1 x2 y1 y2]
%   layout.objTypeId    - 1 X N listing the objectType for each detection
%   layout.id           - 1 x N listing of the index of each layout 
%
%   Currently PASCAL VOC 2010 layout challenge style evalutation, i.e.,
%   there is one layout allowed per aeroplane. Part detections in a layout
%   inherit the score of the layout, and are scored in the detection
%   challenge, but constrained to match parts from the same plane.
%
%   

imageIds = jarReadImageSet(anno, layout.imageSet);
nImages = length(imageIds);
nObjects = length(layout.objTypes);

ap = zeros(nObjects,1);
prec = cell(nObjects,1);
recall = cell(nObjects,1);


for o = 1:nObjects,
    %loop over images and evaluate each detection
    objType = layout.objTypes{o};
    clear matches;
    for i = 1:nImages
        planeId = anno.aeroplane.id(anno.aeroplane.imageId == imageIds(i));
        if (length(planeId) > 1)
            matches(i).labels = [];
            matches(i).scores = [];
            if o == 1
                fprintf('[[ warning ]] imageid: %i has more than one plane, ignoring.\n', imageIds(i));
            end
            continue; 
        end
        layoutId = find(layout.imageId == imageIds(i));
        if ~isempty(layoutId)
            detInds = layout.objTypeId == o & layout.id == layoutId;
            detBoxes = layout.box(:, detInds);
            detScores = ones(1,sum(detInds))*layout.score(layoutId);

            % Get ground-truth boxes corresponding to the planeid
            if strcmp(objType,'aeroplane')
                gtInds = find(anno.(objType).id == planeId);
            else
                gtInds = find(anno.(objType).parentId == planeId);
            end
            gtBoxes = anno.(objType).box(:, gtInds);
            gtDifficult = anno.(objType).isDifficult(gtInds);
            matches(i) = evalDetections(gtBoxes, gtDifficult, detBoxes, detScores);
        end
    end
    labels = cat(2, matches(:).labels);
    scores = cat(2, matches(:).scores);
    [recall{o}, prec{o}, info] = vl_pr(labels, scores);
    % PASCAL >= 2010 style AP: area under the tighest 
    % piecewise constant upperbound on the actual AP
    ap(o) = averagePrecision(recall{o}, prec{o});
end

