#!/usr/bin/perl

use Cwd;
$currWorkDir = &Cwd::cwd();

#@clss = ('aeroplane', 'aeroplane_body', 'aeroplane_nose', 'aeroplane_stabilizer');
@clss = ('aeroplane_body', 'aeroplane_nose', 'aeroplane_stabilizer');
foreach $cls (@clss) {
  $cmd = "qsub -l cpu_arch=x86_64 -pe smp 2 -N $cls " .
         "-o $currWorkDir/logs/$cls.log -j y jar-train.sh $cls 2>&1";

# print "Command:\n $cmd\n";

  @L = `$cmd`;
  warn @L;
  @jnk = split(" ", pop @L);
  push @ID, $jnk[2];
}
