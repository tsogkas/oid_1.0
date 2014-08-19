load /export/ws12/tduosn/data/rbg/rel5-dev/2007/jar_aeroplane_aeroplane-typical-train.mat;
pos = pos(1:2:end);
p = randperm(length(pos));
try
  load('/export/ws12/tduosn/data/rbg/rel5-dev/2007/color_words');
catch
  [C, A] = learn_color_words(pos(p(1:600)), 30);
  save('/export/ws12/tduosn/data/rbg/rel5-dev/2007/color_words', 'C', 'A');
end

%mex -g -outdir bin color/color_word_hists.cc;
mex -O -outdir bin color/color_word_hists.cc;

[h, i1, i2] = vq_pixels(imreadx(pos(1)), C, 8);

load /export/ws12/tduosn/data/rbg/rel5-dev/2007/aeroplane_facing_comp_1.mat;
model.features.use_color = true;
model.features.color_ctrs = C;

load /export/ws12/tduosn/data/rbg/rel5-dev/2007/jar_aeroplane_aeroplane-typical-train.mat;
spos = jar_cluster_facing(pos);

%init_color_templates(model, spos{1}(1:400), 10);
init_color_templates(model, spos{1}, 10);
