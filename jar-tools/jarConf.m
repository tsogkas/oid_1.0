function conf = jarConf()
% conf = jarConf()
%
% Simple jar configuration structure.

%conf.path.anno = '/export/ws12/tduosn/data/jar/annotations/anno_av.mat';
%conf.path.anno = '/export/ws12/tduosn/data/jar/annotations/anno.mat';

% workshop config
%conf.path.anno = '/users/vedaldi/d/jar/annotations/rbg-stable/anno_jul_20.mat';
%conf.path.imageSet = '/users/vedaldi/d/jar/sets/';
%conf.path.image = '/users/vedaldi/d/jar/images/aeroplane/';

% post-workshop / CVPR config
oidconf = oidConfig();
conf.path.anno = fullfile(oidconf.paths.dataDirectory, 'anno.mat');
conf.path.imageSet = fullfile(oidconf.paths.setsDirectory, 'aeroplane-all.txt');
conf.path.image = fullfile(oidconf.paths.imageDirectory, 'aeroplane/');
