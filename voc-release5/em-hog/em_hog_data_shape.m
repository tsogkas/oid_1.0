function [X, num_xforms] = em_hog_data_shape(pos, anno)

conf = voc_config();

obj_type = pos(1).obj_type;
is_phrase = ~isempty(regexp(lower(obj_type), 'phrase$'));

param.TPT_SIZE = conf.em_hog.(obj_type).tpt_size;
param.CELL_SIZE = 8;
param.THRESH = 0.3;
param.IS_PHRASE = is_phrase;
param.TRAIN_SCALE = conf.jar.train_scale;

num_pos = length(pos);
dim = conf.features.dim*param.TPT_SIZE^2;

xspace = conf.em_hog.(obj_type).x_space;
yspace = conf.em_hog.(obj_type).y_space;
sspace = conf.em_hog.(obj_type).s_space;
num_xforms = length(xspace)*length(yspace)*length(sspace);

X = sparse(dim, num_xforms*num_pos);
pos_ind = 1;
for i = 1:num_xforms:num_xforms*num_pos
  diary off; diary on;
  j = 0;
  for ds = sspace
    for dx = xspace
      for dy = yspace
        param.DX = dx;
        param.DY = dy;
        param.BASE_SCALE = ds;
        tic_toc_print('%d/%d\n', pos_ind, num_pos);
        f = extract_feature(pos(pos_ind), anno, param);
        X(:,i+j) = f(:);
        j = j + 1;
      end
    end
  end
  pos_ind = pos_ind + 1;
end


function f = extract_feature(pos, anno, param)

box = pos.boxes;
shapes = get_shapes(pos, anno, param);
width = box(3)-box(1)+1;
height = box(4)-box(2)+1;
scale = param.BASE_SCALE*param.CELL_SIZE*param.TPT_SIZE/max(width,height);
box = (box-1)*scale+1;
box([1 3]) = box([1 3]) + param.DX*param.CELL_SIZE;
box([2 4]) = box([2 4]) + param.DY*param.CELL_SIZE;

mu = mean(cat(2, shapes{:}), 2);
for i = 1:length(shapes)
  shape = shapes{i};
  shape = bsxfun(@minus, shape, mu);
  shape = (shape-1)*scale+1;
  shape = bsxfun(@plus, shape, mu*scale);
  shapes{i} = shape;
end

HALF_SIZE = param.TPT_SIZE*param.CELL_SIZE/2;
width = box(3)-box(1)+1;
height = box(4)-box(2)+1;
cx = box(1) + width/2;
cy = box(2) + height/2;
outer_box = [cx-HALF_SIZE cy-HALF_SIZE cx+HALF_SIZE cy+HALF_SIZE];
outer_box = round(outer_box);
box = round(box);

box(3:4) = box(3:4) - outer_box(1:2) + 1;
box(1:2) = 1 + box(1:2) - outer_box(1:2);
for i = 1:length(shapes)
  shape = shapes{i};
  shape(1,:) = shape(1,:) - outer_box(1) + 1;
  shape(2,:) = shape(2,:) - outer_box(2) + 1;
  shapes{i} = shape;
end
outer_box(3:4) = outer_box(3:4) - outer_box(1:2) + 1;
outer_box(1:2) = 1;

im_sz = [outer_box(4) outer_box(3)];
boundary_im = zeros(im_sz);

for i = 1:length(shapes)
  shape = shapes{i}';
  boundary = poly2boundary(shape);
  out_of_image = boundary(:,1) < 1 | boundary(:,1) > im_sz(2) | boundary(:,2) < 1 | boundary(:,2) > im_sz(1);
  boundary = boundary(~out_of_image,:);
  boundary_idx = sub2ind(im_sz, boundary(:,2), boundary(:,1));
  boundary_im(boundary_idx) = 1;
end

%view_jar_data(pos);
%clf; imagesc(boundary_im); keyboard;

boundary_im = repmat(boundary_im, [1 1 3]);
f = features2(boundary_im, param.CELL_SIZE);
f = f > param.THRESH;

%subplot(2,1,1);
%fprintf('%d %d\n', round(box(3)-box(1)+1), round(box(4)-box(2)+1));
%showboxes(boundary_im, cat(1, box, outer_box));
%subplot(2,1,2);
%visualizeHOG(max(0, f));
%pause;


function shapes = get_shapes(pos, anno, param)

if ~param.IS_PHRASE
  shapes = {pos.polygon};
else
  obj_type = pos.obj_type;
  member_type = anno.meta.(obj_type).memberType;
  members = anno.(obj_type).members{pos.anno_ind};
  shapes = anno.(member_type).polygon(members);

  image_id = anno.(obj_type).imageId(pos.anno_ind);
  image_ind = vl_binsearch(anno.image.id, image_id);
  sz = round(anno.image.size(:,image_ind) * param.TRAIN_SCALE);
  for i = 1:length(shapes)
    shape = shapes{i};
    shape = (shape-1)*param.TRAIN_SCALE + 1;
    if pos.flip
      shape(1,:) = sz(2) - shape(1,:) + 1;
    end
    shapes{i} = shape;
  end
end
