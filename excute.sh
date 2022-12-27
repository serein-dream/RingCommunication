world_size=$1
cluster_name=$2
master_ip=$3
home_path="/home/RunRing"
env_path="${home_path}/virtualenvs/py_pytorch/bin/activate"
code_path="/home/MAXVYANG/fl/ring-reduce"

machines=( 03 04 06 07 08 )

rank=0

for i in "${machines[@]}"; do
  dir_path="${home_path}/world_size${world_size}/"
  log_path=${dir_path}${cluster_name}$i"_"$world_size".log"
  node_name=${cluster_name}$i
  if [[ $i == ${machines[0]} ]]
	then
	  rm -rf $dir_path
	  mkdir -p $dir_path
	fi
	# master and slaves share the same volume, do not need to rm and mkdir.
	echo $node_name
  ssh $node_name "source $env_path; cd $code_path; nohup python3 -u train.py --init-method tcp://$master_ip --rank $rank --world-size $world_size"
  rank=$(($rank+1))
  [ "$rank" -ge "$world_size" ] && echo "enough clients" && exit 1
done
