
#!/bin/bash
#
#$ -cwd
#$ -j y
#$ -S /bin/bash


mpirun --mca btl vader,self,tcp -n 95 ~/anaconda3/envs/bunker2/bin/python /home/gabriel/codes/Anticorruption/dbscan_for_anchorage_new_version_cluster_adjusted_terrestrial.py
