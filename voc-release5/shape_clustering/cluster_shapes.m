function clusters = cluster_shapes(shapes, numclusters, param)
    % clusters a set of shapes represented as polygons into numclusters
    % param is the set of parameters for shape alignment

    DT_SIZE = param.DT_SIZE;
    DT_CENTER = round(param.DT_SIZE([2 1])/2);
    SCALE = param.SCALE;

    numsample = min(100, length(shapes)); %average the DT of these elements
    idx = randperm(length(shapes));
    centers = shapes(idx(1:numsample));
    DT = zeros([DT_SIZE numsample]);
    for i = 1:numsample
        shape = centers{i};
        cx = mean(shape,1);
        scale = max(shape(:,2)) - min(shape(:,2));
        pts = (shape - repmat(cx, size(shape,1),1))/scale*SCALE + repmat(DT_CENTER, size(shape,1),1);
        boundary = poly2boundary(pts);
        out_of_image = boundary(:,1) < 1 | boundary(:,1) > DT_SIZE(2) | boundary(:,2) < 1 | boundary(:,2) > DT_SIZE(1);
        boundary = boundary(~out_of_image,:);
        boundaryIDX = sub2ind(DT_SIZE, boundary(:,2), boundary(:,1));
        boundaryIm = zeros(DT_SIZE);
        boundaryIm(boundaryIDX) = 1;
        DT(:,:,i) = bwdist(boundaryIm);
    end
    mDT = mean(DT,3); %the mean distance transform
    DT_OUT_OF_IMAGE = (mean(mDT(1,:))+mean(mDT(end,:)) + mean(mDT(:,1)) + mean(mDT(:,end)))/4;
    


    % initialize the initial search space
    stepsize = param.STEP_SIZE;
    n = param.TRANSLATION_STEPS;
    [offx,offy] = meshgrid(-n:n,-n:n);
    offx = offx(:)*stepsize; offy = offy(:)*stepsize;
    offs = param.SCALES;
    offr = param.ROTATIONS;
  
    %estimate initial transforms by aligning it to the mean
    init_transforms = zeros(5, length(shapes));
    fprintf('Estimating initial tranforms [%i total]\n', length(shapes));
    fprintf('Search space : offx [%.1f %.1f] step:%.1f, offy [%.1f %.1f] step:%.1f, offs [%.1f %.1f]\n', ...
                min(offx), max(offx), stepsize, min(offy),max(offy), stepsize, min(offs), max(offs));
    
    for i = 1:length(shapes)
        if mod(i,100) == 0
            fprintf('%i..', i);
        end
        shape = shapes{i};
        cx = mean(shape,1);
        scale = max(shape(:,2)) - min(shape(:,2));
        if ~param.INIT_ALIGNMENT
          init_transforms(:,i) = [cx scale 0];
        else
          scores = zeros(5, length(offr)*length(offx)*length(offs));
          count = 1;
          for ridx = 1:length(offr)
            curr_rot = offr(ridx);
            rshape = rot_shape(shape, curr_rot);
            for sidx = 1:length(offs),
                for tidx = 1:length(offx),
                    curr_cx = cx + [offx(tidx) offy(tidx)];
                    curr_scale = scale*offs(sidx);
                    pts = (rshape - repmat(curr_cx, size(rshape,1),1))/curr_scale*SCALE + repmat(DT_CENTER, size(rshape,1),1);
                    boundary = poly2boundary(pts);
                    out_of_image = boundary(:,1) < 1 | boundary(:,1) > DT_SIZE(2) | boundary(:,2) < 1 | boundary(:,2) > DT_SIZE(1);
                    boundary = boundary(~out_of_image,:);
                    boundaryIDX = sub2ind(DT_SIZE, boundary(:,2), boundary(:,1));
                    distance = sum(mDT(boundaryIDX)) + sum(out_of_image)*DT_OUT_OF_IMAGE;
                    scores(:,count) = [curr_cx curr_scale curr_rot distance];
                    count = count + 1;
                end
            end
          end
          [~,minidx] = min(scores(5,:));
          init_transforms(:,i) = scores(:,minidx);
        end
    end
    fprintf('[done]\n');

    %% initialize centers
    NUM_CLUSTERS = numclusters;
    fprintf('Clustering shapes to %i clusters.\n', NUM_CLUSTERS);
    
    %random initialization of shapes
    idx = randperm(length(shapes));

    %create DT out of each shapes
    DT = zeros([DT_SIZE NUM_CLUSTERS]);
    for i = 1:NUM_CLUSTERS
        shape = shapes{idx(i)};
        cx = init_transforms(1:2, idx(i))';
        scale = init_transforms(3, idx(i));
        rot = init_transforms(4, idx(i));
        shape = rot_shape(shape, rot);
        pts = (shape - repmat(cx, size(shape,1),1))/scale*SCALE + repmat(DT_CENTER, size(shape,1),1);
        boundary = poly2boundary(pts);
        out_of_image = boundary(:,1) < 1 | boundary(:,1) > DT_SIZE(2) | boundary(:,2) < 1 | boundary(:,2) > DT_SIZE(1);
        boundary = boundary(~out_of_image,:);
        boundaryIDX = sub2ind(DT_SIZE, boundary(:,2), boundary(:,1));
        boundaryIm = zeros(DT_SIZE);
        boundaryIm(boundaryIDX) = 1;
        DT(:,:,i) = bwdist(boundaryIm);
    end

    % K-Means clustering
    distance = zeros(length(shapes), NUM_CLUSTERS);
    ktransforms = init_transforms;
    kDT = DT; clear DT;
    
    numtoavg = 10;
    tic;

    %fine grained search around current position
    stepsize = param.STEP_SIZE;
    n = param.TRANSLATION_STEPS;
    [offx,offy] = meshgrid(-n:n,-n:n);
    offx = offx(:)*stepsize; offy = offy(:)*stepsize;
    offs = param.SCALES;

    % Cache boundaries
    fprintf('Caching boundary data\n');
    for i = 1:length(shapes)
      if mod(i,100) == 0
        fprintf('%i..', i);
      end

      shape = shapes{i};
      cx = init_transforms(1:2, i)';
      scale = init_transforms(3, i);
      scores = zeros(5, length(offr)*length(offx)*length(offs));
      count = 1;
      for ridx = 1:length(offr)
        curr_rot = offr(ridx);
        rshape = rot_shape(shape, curr_rot);
        for sidx = 1:length(offs),
          for tidx = 1:length(offx),
            curr_cx = cx + [offx(tidx) offy(tidx)];
            curr_scale = scale*offs(sidx);
            pts = (rshape - repmat(curr_cx, size(rshape,1),1))/curr_scale*SCALE + repmat(DT_CENTER, size(rshape,1),1);
            boundary = poly2boundary(pts);
            out_of_image = boundary(:,1) < 1 | boundary(:,1) > DT_SIZE(2) | boundary(:,2) < 1 | boundary(:,2) > DT_SIZE(1);
            boundary = boundary(~out_of_image,:);
            boundaryIDX = sub2ind(DT_SIZE, boundary(:,2), boundary(:,1));
            cache{i,ridx,sidx,tidx}.boundaryIDX = boundaryIDX;
            cache{i,ridx,sidx,tidx}.out_of_image_sum = sum(out_of_image);
          end
        end
      end
    end
    fprintf('[done]\n');


    for iter = 1:param.NUM_ITER,
        fprintf('iter %01i:[m..',iter);
        %UPDATE memeberships
        for i = 1:length(shapes)
            shape = shapes{i};
            cx = ktransforms(1:2, i)';
            scale = ktransforms(3, i);
            rot = ktransforms(4, i);
            shape = rot_shape(shape, rot);
            pts = (shape - repmat(cx, size(shape,1),1))/scale*SCALE + repmat(DT_CENTER, size(shape,1),1);
            boundary = poly2boundary(pts);
            out_of_image = boundary(:,1) < 1 | boundary(:,1) > DT_SIZE(2) | boundary(:,2) < 1 | boundary(:,2) > DT_SIZE(1);
            boundary = boundary(~out_of_image,:);
            boundaryIDX = sub2ind(DT_SIZE, boundary(:,2), boundary(:,1));
            for j = 1:NUM_CLUSTERS
                distance(i,j) = sum(kDT(boundaryIDX + prod(DT_SIZE)*(j-1))) + sum(out_of_image)*DT_OUT_OF_IMAGE;
            end
        end
        [~, kIDX] = min(distance, [], 2);

        fprintf('dt..');
        %UPDATE kDT
        for i = 1:NUM_CLUSTERS,
            idx = find(kIDX == i);
            
            rp = randperm(length(idx));
            idx = idx(rp(1:min(length(idx), numtoavg)));
%            % average DTs for several shapes closest to the
%            % cluster average DT
%            [~, ord] = sort(distance(idx,i));
%            idx = idx(ord(1:min(length(idx), numtoavg)));
            
            ADT = zeros([DT_SIZE length(idx)]);

            for j = 1:length(idx),
                shape = shapes{idx(j)};
                cx = ktransforms(1:2, idx(j))';
                scale = ktransforms(3, idx(j));
                rot = ktransforms(4, idx(j));
                shape = rot_shape(shape, rot);
                pts = (shape - repmat(cx, size(shape,1),1))/scale*SCALE + repmat(DT_CENTER, size(shape,1),1);
                boundary = poly2boundary(pts);
                out_of_image = boundary(:,1) < 1 | boundary(:,1) > DT_SIZE(2) | boundary(:,2) < 1 | boundary(:,2) > DT_SIZE(1);
                boundary = boundary(~out_of_image,:);
                boundaryIDX = sub2ind(DT_SIZE, boundary(:,2), boundary(:,1));
                boundaryIm = zeros(DT_SIZE);
                boundaryIm(boundaryIDX) = 1;
                ADT(:,:,j) = bwdist(boundaryIm);
            end
            kDT(:,:,i) = mean(ADT,3);
        end
        fprintf('t..');
        %update tranformation
        for i = 1:length(shapes)
            shape = shapes{i};
            cx = init_transforms(1:2, i)';
            scale = init_transforms(3, i);
            scores = zeros(5, length(offr)*length(offx)*length(offs));
            count = 1;
            for ridx = 1:length(offr)
              curr_rot = offr(ridx);
              %rshape = rot_shape(shape, curr_rot);
              for sidx = 1:length(offs),
                  for tidx = 1:length(offx),
                      curr_cx = cx + [offx(tidx) offy(tidx)];
                      curr_scale = scale*offs(sidx);
                      %pts = (rshape - repmat(curr_cx, size(rshape,1),1))/curr_scale*SCALE + repmat(DT_CENTER, size(rshape,1),1);
                      %boundary = poly2boundary(pts);
                      %out_of_image = boundary(:,1) < 1 | boundary(:,1) > DT_SIZE(2) | boundary(:,2) < 1 | boundary(:,2) > DT_SIZE(1);
                      %boundary = boundary(~out_of_image,:);
                      %boundaryIDX = sub2ind(DT_SIZE, boundary(:,2), boundary(:,1)) + (kIDX(i)-1)*prod(DT_SIZE);
                      boundaryIDX = cache{i,ridx,sidx,tidx}.boundaryIDX + (kIDX(i)-1)*prod(DT_SIZE);
                      %distance = sum(kDT(boundaryIDX)) + sum(out_of_image)*DT_OUT_OF_IMAGE;
                      distance = sum(kDT(boundaryIDX)) + cache{i,ridx,sidx,tidx}.out_of_image_sum*DT_OUT_OF_IMAGE;
                      scores(:,count) = [curr_cx curr_scale curr_rot distance];
                      count = count + 1;
                  end
              end
            end
            [~,minidx] = min(scores(5,:));
            ktransforms(:,i) = scores(:,minidx);
        end
        fprintf('] %.2fs elapsed.\n',toc);
    end

    
    clusters.mDT=mDT; %initial mean
    clusters.kDT=kDT; %cluster mean
    clusters.kIDX=kIDX; %cluster assignment
    
    transforms = zeros(4, length(shapes));
    for i = 1:length(shapes)
        cx = ktransforms(1:2, i)';
        scale = ktransforms(3, i);
        transforms(1:2,i) = -cx'/scale*SCALE + DT_CENTER';
        transforms(3,i) = SCALE/scale;     
        transforms(4,i) = ktransforms(4, i);
        transforms(5,i) = ktransforms(5, i);
    end
    clusters.transforms=transforms; %transform to the cluter center
end
