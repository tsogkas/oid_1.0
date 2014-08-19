function id = jar_get_image_id(anno, obj_type, index)

parent_type = obj_type;
ind = index;
while ~strcmp(parent_type, 'image')
  parent_id = anno.(parent_type).parentId(ind);
  parent_type = anno.get_type(parent_id);
  ind = vl_binsearch(anno.(parent_type).id, parent_id);
end
id = anno.(parent_type).id(ind);
