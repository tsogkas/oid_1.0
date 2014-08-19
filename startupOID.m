function startupOID()

if ~exist(['cpp/cascadeHFT.' mexext], 'file') || ~exist(['cpp/fconvsse_ST.' mexext], 'file')
    compileMex('cpp/');
end
addpath(genpath('cpp/'))
addpath(genpath('data/'))
addpath(genpath('jar-tools/'))
addpath(genpath('voc-release5/'))

end