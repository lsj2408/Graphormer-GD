# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
ulimit -c unlimited
[ -z "${n_gpu}" ] && n_gpu=$(nvidia-smi -L | wc -l)
[ -z "${lr}" ] && lr=2e-4
[ -z "${end_lr}" ] && end_lr=1e-9
[ -z "${warmup_steps}" ] && warmup_steps=60000
[ -z "${total_steps}" ] && total_steps=400000
[ -z "${layers}" ] && layers=12
[ -z "${hidden_size}" ] && hidden_size=80
[ -z "${ffn_size}" ] && ffn_size=80
[ -z "${num_head}" ] && num_head=8
[ -z "${batch_size}" ] && batch_size=64
[ -z "${update_freq}" ] && update_freq=1
[ -z "${seed}" ] && seed=1
[ -z "${clip_norm}" ] && clip_norm=5
[ -z "${data_path}" ] && data_path="./"
[ -z "${save_path}" ] && save_path="./"
[ -z "${dropout}" ] && dropout=0.0
[ -z "${act_dropout}" ] && act_dropout=0.1
[ -z "${attn_dropout}" ] && attn_dropout=0.1
[ -z "${weight_decay}" ] && weight_decay=0.01

[ -z "${droppath_prob}" ] && droppath_prob=0.0

[ -z "${MASTER_PORT}" ] && MASTER_PORT=10086
[ -z "${OMPI_COMM_WORLD_SIZE}" ] && OMPI_COMM_WORLD_SIZE=1

[ -z "$save_prefix" ] && save_prefix=''
[ -z "${dataset_name}" ] && dataset_name="zinc-rd-subset"

[ -z "${relu_mul_bias}" ] && relu_mul_bias="false"
[ -z "${one_init_mul_bias}" ] && one_init_mul_bias="false"
[ -z "${mul_bias_with_edge_feature}" ] && mul_bias_with_edge_feature="false"

hyperparams=lr-$lr-end_lr-$end_lr-tsteps-$total_steps-wsteps-$warmup_steps-L$layers-D$hidden_size-H$num_head-BS$((batch_size*update_freq*n_gpu))-CLIP$clip_norm-dp$dropout-attn_dp$attn_dropout-act_dp$act_dropout-wd$weight_decay-dpp$droppath_prob/SEED$seed-o-$one_init_mul_bias-r-$relu_mul_bias-e-$mul_bias_with_edge_feature
save_dir=$save_path/$dataset_name-$save_prefix-$hyperparams
tsb_dir=$save_dir/tsb

mkdir -p $save_dir

echo -e "\n\n"
echo "=====================================ARGS======================================"
echo "arg0: $0"
echo "seed: ${seed}"
echo "batch_size: $((batch_size*update_freq*n_gpu))"
echo "n_layers: ${layers}"
echo "lr: ${lr}"
echo "warmup_steps: ${warmup_steps}"
echo "total_steps: ${total_steps}"
echo "clip_norm: ${clip_norm}"
echo "hidden_size: ${hidden_size}"
echo "ffn_size: ${ffn_size}"
echo "num_head: ${num_head}"
echo "update_freq: ${update_freq}"
echo "dropout: ${dropout}"
echo "attn_dropout: ${attn_dropout}"
echo "act_dropout: ${act_dropout}"
echo "weight_decay: ${weight_decay}"
echo "droppath_prob: ${droppath_prob}"
echo "relu_mul_bias: ${relu_mul_bias}"
echo "one_init_mul_bias: ${one_init_mul_bias}"
echo "mul_bias_with_edge_feature: ${mul_bias_with_edge_feature}"
echo "dataset_name: ${dataset_name}"
echo "save_dir: ${save_dir}"
echo "tsb_dir: ${tsb_dir}"
echo "data_dir: ${data_path}"
echo "==============================================================================="

# ENV
echo -e "\n\n"
echo "======================================ENV======================================"
echo 'Environment'
ulimit -c unlimited;
echo '\n\nhostname'
hostname
echo '\n\nnvidia-smi'
nvidia-smi
echo '\n\nls -alh'
ls -alh
echo -e '\n\nls ~ -alh'
ls ~ -alh
echo "torch version"
python -c "import torch; print(torch.__version__)"
echo "==============================================================================="

echo -e "\n\n"
echo "==================================MP==========================================="
echo "MASTER_ADDR: ${MASTER_ADDR}"
echo "MASTER_PORT: ${MASTER_PORT}"
echo "NCCL_SOCKET_IFNAME: ${NCCL_SOCKET_IFNAME}"
echo "OMPI_COMM_WORLD_RANK: ${OMPI_COMM_WORLD_RANK}"
echo "OMPI_COMM_WORLD_SIZE: ${OMPI_COMM_WORLD_SIZE}"


if [[ -z "${OMPI_COMM_WORLD_SIZE}" ]]
then
  ddp_options=""
else
  if (( $OMPI_COMM_WORLD_SIZE == 1))
  then
	ddp_options=""
  else
    ddp_options="--nnodes=$OMPI_COMM_WORLD_SIZE --node_rank=$OMPI_COMM_WORLD_RANK --master_addr=$MASTER_ADDR"
  fi
fi
echo "ddp_options: ${ddp_options}"
echo "==============================================================================="

# ENV
echo -e "\n\n"
echo "======================================ACTIONS======================================"
if ( $one_init_mul_bias == "true" )
then
  one_init_mul_bias_args="--one-init-mul-bias"
else
  one_init_mul_bias_args=""
fi
echo "one_init_mul_bias_args: ${one_init_mul_bias_args}"

if ( $relu_mul_bias == "true" )
then
  relu_mul_bias_args="--relu-mul-bias"
else
  relu_mul_bias_args=""
fi
echo "relu_mul_bias_args: ${relu_mul_bias_args}"

if ( $mul_bias_with_edge_feature == "true" )
then
  mul_bias_with_edge_feature_args="--mul-bias-with-edge-feature"
else
  mul_bias_with_edge_feature_args=""
fi
echo "mul_bias_with_edge_feature_args: ${mul_bias_with_edge_feature_args}"

echo "==============================================================================="

export NCCL_ASYNC_ERROR_HADNLING=1
export OMP_NUM_THREADS=1

python -m torch.distributed.launch --nproc_per_node=$n_gpu --master_port=$MASTER_PORT $ddp_options train.py \
--user-dir ./graphormer \
--num-workers 16 \
--ddp-backend=legacy_ddp \
--dataset-name $dataset_name \
--dataset-source pyg --with-resistance-distance \
--task graph_prediction --valid-subset valid,test \
--criterion l1_loss \
--arch graphormer_slim $one_init_mul_bias_args $relu_mul_bias_args \
--num-classes 1 --seed $seed $mul_bias_with_edge_feature_args \
--attention-dropout $attn_dropout --act-dropout $act_dropout --dropout $dropout --droppath-prob $droppath_prob \
--optimizer adam --adam-betas '(0.9, 0.999)' --adam-eps 1e-8 --clip-norm $clip_norm --weight-decay $weight_decay \
--lr-scheduler polynomial_decay --power 1 \
--warmup-updates $warmup_steps --total-num-update $total_steps --max-update $total_steps \
--lr $lr --end-learning-rate $end_lr \
--batch-size $batch_size \
--data-buffer-size 20 \
--encoder-layers $layers \
--encoder-embed-dim $hidden_size \
--encoder-ffn-embed-dim $ffn_size \
--encoder-attention-heads $num_head \
--no-epoch-checkpoints \
--max-epoch 10000 \
--save-interval-updates 10000 \
--save-dir $save_dir --tensorboard-logdir $tsb_dir
