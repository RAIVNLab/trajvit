#!/bin/bash

# Default values for arguments
exp_name="debug"
train_corpus="panda_short"
test_corpus="activitynet_test"
exp_dir=${SL_EXP_DIR}
ngpus=1
model="trajvit"
batch_size=8
task="pretrain"
num_latent=1
nnode=1
train_num_frames=16
test_num_frames=16
vit_name="vit-large"
image_ckpt=None
ckpt=None
resume=false
mask_down_factor=4
lr=1e-4
save_freq=1
log_wandb=false

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --exp_name) exp_name="$2"; shift ;;
        --train_corpus) train_corpus="$2"; shift ;;
        --test_corpus) test_corpus="$2"; shift ;;
        --ngpus) ngpus="$2"; shift ;;
        --model) model="$2"; shift ;;
        --batch_size) batch_size="$2"; shift ;;
        --task) task="$2"; shift ;;
		--num_latent) num_latent="$2"; shift ;;
		--nnode) nnode="$2"; shift ;;
        --train_num_frames) train_num_frames="$2"; shift ;;
        --test_num_frames) test_num_frames="$2"; shift ;;
        --vit_name) vit_name="$2"; shift ;;
        --image_ckpt) image_ckpt="$2"; shift ;;
        --mask_down_factor) mask_down_factor="$2"; shift ;;
        --ckpt) ckpt="$2"; shift ;;
        --lr) lr="$2"; shift ;;
        --save_freq) save_freq="$2"; shift ;;
        --log_wandb) log_wandb=true ;;  # Toggle true if present
        --resume) resume=true ;;  # Toggle true if present
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

if [[ $train_num_frames -gt 16 || $test_num_frames -gt 16 ]]; then
    use_test_64=true
else
    use_test_64=false
fi


mode="local"

# Derived variables
output_dir="${exp_dir}/pt_${train_corpus}/${train_corpus}_${exp_name}"
config_path="./configs/pretrain.yaml"
echo "output dir >> ${output_dir}"

project_dir=$PWD

# if [[ -d ${output_dir} ]] && [[ ${exp_name} != "debug" ]]; then
# 	echo "Dir ${output_dir} already exist. Exit."
# 	exit 1
# fi

mkdir -p ${output_dir}

if [[ ${exp_name} != "debug" ]]; then
	### save code copy 
	cd .. 
	code_dir=${output_dir}/code
	project_dirname=entry
	rsync -ar ${project_dirname} ${code_dir} --exclude='*.out'  # --exclude='.git'
	cd ${code_dir}/${project_dirname}
	echo "Copied source files to '${PWD}' and launch from this dir"
fi

if [[ ${ckpt} != "None" ]]; then
    resume=true
fi


############### ======> Your training scripts [START]
common_args="
    -m tasks.${task}
    ${config_path}
    output_dir=${output_dir}
    train_corpus=${train_corpus}
    vit_type=${model}
    batch_size.video=${batch_size}
    traj_pos.use_bounding_box=true
    optimizer.max_grad_norm=10
    perceiver.num_latent=${num_latent}
    video_input.num_frames_test=${test_num_frames}
    video_input.num_frames=${train_num_frames}
    use_test_64=${use_test_64}
    traj_model.model_name=${vit_name}
    optimizer.lr=${lr}
    image_pretrained_path=${image_ckpt}
    mask_down_factor=${mask_down_factor}
    pretrained_path=${ckpt}
    save_freq=${save_freq}
    resume=${resume}
"

if [[ ${exp_name} != "debug" ]]; then
    if [[ ${nnode} != 1 ]]; then
        echo "beaker replica rank ${BEAKER_REPLICA_RANK}; master address ${BEAKER_LEADER_REPLICA_HOSTNAME}"

        torchrun --nnodes=${nnode} \
            --nproc_per_node=${ngpus} \
            --node_rank ${BEAKER_REPLICA_RANK} \
            --master_addr ${BEAKER_LEADER_REPLICA_HOSTNAME} \
            --master_port 29406 \
            ${common_args} \
            wandb.project=video_encoder_${corpus} \
            wandb.enable=${log_wandb}

    else
		PYTHONPATH=.:${PYTHONPATH} \
		torchrun --nnodes=1 \
			--nproc_per_node=${ngpus} \
			${common_args} \
			wandb.project=video_encoder_${corpus} \
			wandb.enable=${log_wandb}
	fi
else
	PYTHONPATH=.:${PYTHONPATH} \
	torchrun --nnodes=1 \
		--nproc_per_node=${ngpus} \
		${common_args} \
		wandb.enable=False
fi


############### ======> Your training scripts [END]

### cd back
cd ${project_dir}
echo "Finish at dir: ${PWD}"
echo "back to dir: ${project_dir}"
echo "output dir >> ${output_dir}"
