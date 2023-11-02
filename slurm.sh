#!/bin/bash
#SBATCH --job-name=python-cpu-test ##作业名称
#SBATCH --partition=airs-cpu ##作业申请的分区名称
#SBATCH --nodes=1 ##作业申请的节点数
#SBATCH --ntasks-per-node=1 ##作业申请的每个节点使用的核心数#SBATCH --error=%j.err
#SBATCH --output=%j.out
CURDIR=`pwd`
rm -rf $CURDIR/nodelist.$SLURM_JOB_ID
NODES=`scontrol show hostnames $SLURM_JOB_NODELIST`
for i in $NODES
do
echo "$i:$SLURM_NTASKS_PER_NODE" >> $CURDIR/nodelist.$SLURM_JOB_IDdone
echo $SLURM_NPROCS
echo "process will start at : "
date
echo "++++++++++++++++++++++++++++++++++++++++"
##以下按照python在CPU单核串行计算的需求，添加所需要用到的python软件环境变量#setting environment for python3.6
export PATH=/home/software/python/python-3.6/bin:$PATH
python /home/223100001/PotraitNet/train.py ##该命令为python软件基于CPU跑单核串行模式的命令格式。echo "++++++++++++++++++++++++++++++++++++++++"
echo "processs will sleep 30s"
sleep 30
echo "process end at : "
date
rm -rf $CURDIR/nodelist.$SLURM_JOB_ID
