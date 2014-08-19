function ap = test_jarEval(anno)

imageSet = 'aeroplane-typical-test';
objType = 'aeroplane';

imageIds = jarReadImageSet(anno, imageSet);

aeroInds = find(ismember(anno.aeroplane.imageId, imageIds));

det.box = anno.aeroplane.box(:, aeroInds);
%det.score = rand(1, size(det.box, 2));
det.box(:,2:2:end) = 0;
det.score = size(det.box, 2):-1:1;
det.imageId = anno.aeroplane.imageId(aeroInds);

[ap, prec, recall] = jarEval(det, anno, objType);
plot(recall, prec);
