% visualizeFilterTree
% Visualize hierarchical tree of merged filters.
% If stop == true, visualization pauses after showing each filter.
% 
%   visualizeFilterTree(tree, stop)
% 
% Stavros Tsogkas <stavros.tsogkas@ecp.fr>
% Last update: October 2013


function visualizeFilterTree(tree, stop)

if nargin < 2
    stop = true;
end

%  Find root position
indRoot = getRootPosition(tree);

% Find nodes at each level
colors = {'r','g','b','m','c','y','n','o','p','l'};
atLevel{1} = indRoot;
colorAtLevel{1} = 'k';
parentColorAtLevel{1} = 'k';

i = 1;
ic = 1;
while ~isempty(atLevel{i})
    atLevel{i+1} = [];
    colorAtLevel{i+1} = [];
    parentColorAtLevel{i+1} = [];
%     colors = colors(randperm(length(colors)));
    for j=1:numel(atLevel{i})
        atLevel{i+1} = [atLevel{i+1}, tree(atLevel{i}(j)).children];
        for k=1:length(tree(atLevel{i}(j)).children)
            parentColorAtLevel{i+1} = [parentColorAtLevel{i+1}, colorAtLevel{i}(j)];
%             colorAtLevel{i+1} = [colorAtLevel{i+1}, colors{1+mod(2*j+k,length(colors))}];
            if isempty(tree(tree(atLevel{i}(j)).children(k)).children)
                colorAtLevel{i+1} = [colorAtLevel{i+1}, 'w'];
            else
                colorAtLevel{i+1} = [colorAtLevel{i+1}, colors{mod(ic,length(colors))+1}];
            end
            ic = ic+1;
        end
    end
    i = i + 1;
end

% Find max tree width
nLevels = i-1;
maxWidth = 0;
for i=1:nLevels
    maxWidth = max(maxWidth,length(atLevel{i}));
end

% Visualize tree 
figure;
figHeight = 1/nLevels;
for i=1:nLevels
    nNodes   = length(atLevel{i});
    figWidth = 1/(nNodes+1);
    figStart = (1-nNodes*figWidth)/2;
    for j=1:nNodes
        left   = figStart+(j-1)*figWidth;
        bottom = 1-figHeight*i;
        subplot('Position',[left,bottom,figWidth,figHeight*0.9]);
        pcolor = 255*color2vector(parentColorAtLevel{i}(j)); 
        chcolor = 255*color2vector(colorAtLevel{i}(j));
        visualizeHOG(max(tree(atLevel{i}(j)).filter,0),pcolor,chcolor);
        if stop
            pause
        end
    end
end


% -------------------------------------------------------------------------
function visualizeHOG(w,parentColor,childColor)
% Visualize HOG features/weights.
%   visualizeHOG(w)

% Make pictures of positive and negative weights
bs = 20;
w = w(:,:,1:9);
scale = max(max(w(:)),max(-w(:)));
pos = HOGpicture(w, bs) * 255/scale;
neg = HOGpicture(-w, bs) * 255/scale;

% Put pictures together and draw
buff = 10;
pos = padarray(pos, [buff buff], 128, 'both');
if min(w(:)) < 0
  neg = padarray(neg, [buff buff], 128, 'both');
  im = uint8([pos; neg]);
else
  im = uint8(pos);
end

if nargin>1
    im = repmat(im,[1 1 3]);
    for i=1:size(im,3)
        im(1:buff,:,i) = parentColor(i);
        im(end-buff:end,:,i) = childColor(i);
    end
end
imagesc(im); 
colormap gray;
axis equal;
axis off;


function im = HOGpicture(w, bs)
% Make picture of positive HOG weights.
%   im = HOGpicture(w, bs)

% construct a "glyph" for each orientaion
bim1 = zeros(bs, bs);
bim1(:,round(bs/2):round(bs/2)+1) = 1;
bim = zeros([size(bim1) 9]);
bim(:,:,1) = bim1;
for i = 2:9,
  bim(:,:,i) = imrotate(bim1, -(i-1)*20, 'crop');
end

% make pictures of positive weights bs adding up weighted glyphs
s = size(w);    
w(w < 0) = 0;    
im = zeros(bs*s(1), bs*s(2));
for i = 1:s(1),
  iis = (i-1)*bs+1:i*bs;
  for j = 1:s(2),
    jjs = (j-1)*bs+1:j*bs;          
    for k = 1:9,
      im(iis,jjs) = im(iis,jjs) + bim(:,:,k) * w(i,j,k);
    end
  end
end

% --- Turns string input into corresponding rgb color vector --------------
% (supports basic colors).
function v = color2vector(color)

assert(ischar(color),'input must be a string or a single char');
switch color
    case {'r','red'}
        v = [1 0 0];
    case {'b','blue'}
        v = [0 0 1];
    case {'g','green'}
        v = [0 1 0];
    case {'y','yellow'}
        v = [1 1 0];
    case {'m','magenta'}
        v = [1 0 1];
    case {'c','cyan'}
        v = [0 1 1];
    case {'w','white'}
        v = [1 1 1];
    case {'k','black'}
        v = [0 0 0];
    case {'n','navy'}
        v = [0 0 .5];
    case {'ol','olive'}
        v = [.5 .5 0];
    case {'o','orange'}
        v = [1 .647 0];
    case {'maroon'}
        v = [.5 0 0];
    case {'p','purple'}
        v = [.5 0 .5];
    case {'s','silver'}
        v = [.75 .75 .75];
    case {'l','gold'}
        v = [1 .843 0];
    otherwise
        error('Color not supported yet')
end
