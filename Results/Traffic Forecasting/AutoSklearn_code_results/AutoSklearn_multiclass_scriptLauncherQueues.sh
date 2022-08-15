#!/bin/bash
listTimes = "900 3600 9000"
for t in $listTimes;do
	listVar="1 2"
	for i in $listVar; 	do
		sbatch --partition=normal AutoSklearn_multiclass_scriptPython.sh $t $i
	done;
done
