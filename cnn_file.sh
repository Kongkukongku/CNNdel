#!/bin/bash


if [ $# -lt 1 ]
then
	echo "Usage: . /path/cnn_file.sh $sample_chr"
	return
fi


sample_chr=$1
benchmark=/home/wj/DL/benchmark_vcf/${sample_chr}."benchmark.vcf"
val=out.cnn.$sample_chr
tools_integrated=/home/wj/DL/integrated/out.integrated.$sample_chr.format     # results of tools
filename=${sample_chr}.predict.res
paste $tools_integrated $filename | awk '{if($5==1) print $1"\t"$2"\t"$3}' > ${val}.format # the input with label 1 CNNdel has detected
verify-deletion -e 0.3 -m 1000 ${val}.format $benchmark ${val}.cmp >> res.cnn.txt # find from the benchmark to see how many 


echo -n "validated:" >> stats.res.txt
awk '{a+=$2; b+=$3; c+=$4; d+=$5} END{printf("\t%d\t%d\t%d|%d\t%.4f\t%.4f\n",a,b,c,d, c/a, d/b)}'  res.cnn.txt >> stats.res.txt	


