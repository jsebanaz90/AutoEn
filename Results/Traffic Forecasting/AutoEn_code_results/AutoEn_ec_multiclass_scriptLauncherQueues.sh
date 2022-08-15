#!/bin/bash
listVar="1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18"
for i in $listVar; do
	sbatch --partition=normal AutoEn_ec_multiclass_scriptPython.sh $i
done