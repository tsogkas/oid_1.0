function view_clusters(clusters, shapes)
    %% ESTIMATE initial scaling and transformation
    figure; clf;
    imagesc(clusters.mDT); colormap hot;
    title('Mean Distance Transform');

    NUM_CLUSTERS = size(clusters.kDT,3);

    % draw the clusters    
    figure; clf;
    nx = 4;
    ny = ceil(NUM_CLUSTERS/nx);
    for i = 1:NUM_CLUSTERS,
        idx = find(clusters.kIDX == i);
        subplot(ny,nx,i); hold on; axis ij; 
        for j = 1:length(idx),
            shape = shapes{idx(j)};
            cx = clusters.transforms(1:2, idx(j))';
            scale = clusters.transforms(3, idx(j));
            rot = clusters.transforms(4, idx(j));
            shape = rot_shape(shape, rot);
            normshape = shape*scale + repmat(cx,size(shape,1),1);
            plot(normshape([1:end 1], 1), normshape([1:end 1],2),'-','Color',rand(3,1));
        end
        axis tight;       
        axis off;
        axis equal;
        title(sprintf('%i', length(idx)));
    end
    
    % draw the average distance transforms
    kDT = reshape(clusters.kDT, size(clusters.kDT,1), size(clusters.kDT,2), 1, size(clusters.kDT,3));
    figure; clf;
    maxval = max(kDT(:));
    montage(kDT/maxval); colormap hot;
    title('Cluster Mean DT');
end
