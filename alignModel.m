% alignModel    Aligns the filters in a dpm so that left-facing filters 
% correspond to odd indexes and right-facing filters correspond to even 
% indexes. The hierarchical filter cascade code works only with models with 
% left-facing filters as the default and right-facing filters as flipped.
% 
% For models that have already been trained using the lrsplit function from
% voc-release5, the filters are not necessarily arranged in this way, so we
% have to rearrange them manually. This can be avoided if we use the
% viewpoint annotations that come with the OID dataset, to split training
% examples to left/right facing (call jar_train with usevp = 1).
% 
%       model = alignModel(model)
% 
% Stavros Tsogkas <stavros.tsogkas@ecp.fr>
% Last update: August 2014


function model = alignModel(model)

if isfield(model, 'isAligned') && model.isAligned
    warning('Model is already aligned. Leaving model unchanged');
    return
else
    switch model.class
        case 'nose'
            switch model.note
                case 'unsup left-right with k=20'
                    componentsToFlip = [1,6,8,10,13,14,16,18,19,20];
                case 'unsup left-right with k=40'
                    componentsToFlip = [2,3,4,6,8,9,11,13,15,16,19,22,24,25,26,27,28,30,34,37,40];
                otherwise
                    warning('Training configuration unknown. Leaving model unchanged')
                    componentsToFlip = [];
            end
        case 'verticalStabilizer'
            switch model.note
                case 'unsup left-right with k=20'
                    componentsToFlip = [1,5,6,8,12,13,14,16,17];
                case 'unsup left-right with k=40'
                otherwise
                    warning('Training configuration unknown. Leaving model unchanged')
                    componentsToFlip = [];
            end
        otherwise
            warning('Aeroplane part not supported. Leaving model unchanged')
            componentsToFlip = [];
    end
    for iComp = componentsToFlip
        % flip weight vector
        ind = 2*iComp - 1;
        blocklabel = model.filters(ind).blocklabel;
        shape = model.blocks(blocklabel).shape;
        w = flipfeat(reshape(model.blocks(blocklabel).w, shape));
        model.blocks(blocklabel).w = w(:);
        assert(all(isinf(model.blocks(blocklabel).lb)))
    end
    assert(all([model.filters(1:2:end).flip] == 0))
    assert(all([model.filters(2:2:end).flip] == 1))
    model.isAligned = true;
end
