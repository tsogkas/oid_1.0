#!/bin/bash
# qsub -l cpu_arch=x86_64 -pe smp 4 -N global_facing \
#  -o /home/rbg/rel5-dev/sge/logs/global_facing.log \
#  -j y jar-train-global-facing.sh 2>&1



cd /home/rbg/rel5-dev

/export/ws12/tduosn/software/matlab2011a/bin/matlab -nodesktop << EOF
matlabpool open 6;

anno = jarLoadAnno();
load /export/ws12/tduosn/data/rbg/rel5-dev/2007/aeroplane_final.mat;
[ap, recall, prec] = jar_test(model, anno, 'aeroplane-typical-test');
EOF

exit 0
