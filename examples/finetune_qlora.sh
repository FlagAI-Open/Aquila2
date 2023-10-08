# *** Please Modify the following parameters according to your own case. ***
# Path to Aquila2 project
AQUILA2_HOME=/data2/yzd/git/Aquila2

# Location and name of the checkpoint file
CKPT_INPUT=$AQUILA2_HOME/checkpoints
MODEL_NAME_INPUT=aquila2chat-hf

# Path and name of dataset file
DATASETS=/data2/20230907
DATA_FILE=sft_v0.9.12_train.jsonl



# *** The following parameters can be modified according to your own needs. ***
# Epochs
EPOCHS=5

# Conversation template, chosen from "aquila-v1","aquila-chat" and "aquila"
CONVO_TEMPLATE=aquila-v1

# Self-defined experiment name
EXPNAME=aquila_experiment

# Path to the experiment logs
EXPNAME_PATH=${AQUILA2_HOME}/output/logs/$EXPNAME

# Path to the output checkpoints 
CKPT_OUTPUT=$AQUILA2_HOME/output/checkpoints

# Name of the output checkpoint
MODEL_NAME_OUTPUT=$MODEL_NAME_INPUT-sft-$EXPNAME

# Path to the deepspeed config 
DEEPSPEED_CONFIG=$AQUILA2_HOME/examples/ds_zero2.config

# Path to the hostfile
HOSTFILE=$AQUILA2_HOME/examples/hostfile




# *** Training Process **
NNodes=`wc -l ${HOSTFILE} | cut -d " " -f1`
MASTER_ADDR=`head -n 1 ${HOSTFILE} | cut -d " " -f1`
echo "Master node: ${MASTER_ADDR}"


i=0
for ip in `cat ${HOSTFILE} | cut -d " " -f1`
do
    echo "Starting node ${i}/${NNodes}: ${ip}"
    ssh $ip eval \
    	"cd ${PWD} && \
         export PYTHONPATH=${PYTHONPATH}:. && \
         export WANDB_MODE=offline && \
         
         torchrun \
             --nnodes=${NNodes} \
             --node_rank=${i} \
             --nproc_per_node=8 \
             --master_addr=${MASTER_ADDR} \
             --master_port=20001 \
	         $AQUILA2_HOME/examples/finetune.py \
             --model_dir $CKPT_INPUT \
             --model_name $MODEL_NAME_INPUT \
             --data_path $DATASETS/$DATA_FILE \
             --use_lora True \
             --q_lora True \
             --lora_r 8 \
             --lora_alpha 16 \
             --lora_dropout 0.05 \
             --convo_template $CONVO_TEMPLATE \
             --fp16 \
             --model_max_length 2048 \
             --output_dir $CKPT_OUTPUT/$MODEL_NAME_OUTPUT \
             --num_train_epochs $EPOCHS \
             --per_device_train_batch_size 4 \
             --per_device_eval_batch_size 1 \
             --gradient_accumulation_steps 1 \
             --evaluation_strategy no \
             --eval_steps 1500 \
             --save_strategy 'epoch' \
             --save_steps 2000 \
             --save_total_limit 10 \
             --learning_rate 9.65e-6 \
             --weight_decay 0.1 \
             --seed 42 \
             --warmup_ratio 0.1 \
             --lr_scheduler_type 'linear' \
             --logging_steps 1 \
             --deepspeed $DEEPSPEED_CONFIG \
             --gradient_checkpointing True \
             --flash_attn True \
             --lazy_preprocess True" 
    i=`expr $i + 1`
done

wait