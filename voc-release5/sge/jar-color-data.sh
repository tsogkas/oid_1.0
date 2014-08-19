#!/bin/bash
# qsub -l cpu_arch=x86_64 -pe smp 4 -N global_facing \
#  -o /home/rbg/rel5-dev/sge/logs/global_facing.log \
#  -j y jar-train-global-facing.sh 2>&1

cd /home/rbg/rel5-dev

/export/ws12/tduosn/software/matlab2011a/bin/matlab -nodesktop << EOF
matlabpool open 4;
color_debug;
EOF

exit 0
