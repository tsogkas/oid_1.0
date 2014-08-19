function [imageIds, tag] = jarReadImageSet(anno, objectClass, ...
                                           imageSets, onlyTypical)
% function [imageIds, tag] = jarReadImageSet(anno, objectClass, imageSets, onlyTypical)

conf = jarConf();

% Universe of imageIds containing the target object class
imageIds = unique(anno.(objectClass).parentId);
[tf, imageIdx] = ismember(imageIds, anno.image.id);
assert(all(tf == true));

% Look up numeric ids for the image sets
[tf, imageSetIds] = ismember(imageSets, anno.meta.image.set);
assert(all(tf == true));

% Find the image indexes for members of the sets
tf = ismember(anno.image.set, imageSetIds);
imageIdxInSets = find(tf == true);

% Image ids for the target object class and image sets
imageIdx = intersect(imageIdx, imageIdxInSets);
imageIds = anno.image.id(imageIdx);

% Restrict to typical
if onlyTypical
  I = find(anno.image.typical(imageIdx) == true);
  imageIdx = imageIdx(I);
  imageIds = anno.image.id(imageIdx);
end

% Validate the selection
assert(all(ismember(unique(anno.image.set(imageIdx)), imageSetIds) == true));
if onlyTypical
  assert(all(anno.image.typical(imageIdx) == true));
end

if nargout == 2
  tag = objectClass;
  if onlyTypical
    tag = [tag '_typical'];
  else
    tag = [tag '_diverse'];
  end
  if ~iscell(imageSets)
    imageSets = {imageSets};
  end
  tag = [tag '_' imageSets{:}];
end
