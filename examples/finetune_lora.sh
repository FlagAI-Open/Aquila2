COMMON_PATH=/data2/yzd/git/Aquila2/
FASTCHAT_HOME=/data2/yzd/git/Aquila2/
export PYTHONPATH=$FASTCHAT_HOME:$PYTHONPATH

set -u
  EXPNAME=$1
set +u
EXPNAME_PATH=${COMMON_PATH}/output/logs/$EXPNAME
mkdir -p $EXPNAME_PATH
cp $0 $EXPNAME_PATH/

# 7B 
CKPT_INPUT=$COMMON_PATH/examples/checkpoints
MODEL_NAME_INPUT=aquila2chat-hf

# # 34B
CKPT_INPUT=/data2/20230907
MODEL_NAME_INPUT=/iter_0205000_hf

DATASETS=/data2/20230907
CKPT_OUTPUT=$COMMON_PATH/output/checkpoints
LOGFILE=$EXPNAME_PATH/log.txt
LOGFILE=$COMMON_PATH/output/logs/log.txt.$EXPNAME
DEEPSPEED_CONFIG=ds_zero2.config
HOSTFILE=hostfile

DATA_VERSION=v0.9.12

DATA_FILE=sft_${DATA_VERSION}_train.jsonl
MODEL_NAME_OUTPUT=$MODEL_NAME_INPUT-sft-$DATA_VERSION-$EXPNAME

EPOCHS=5
CONVO_TEMPLATE=aquila
CONVO_TEMPLATE=aquila-v1

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
	         /data2/yzd/git/Aquila2/examples/finetune.py \
             --model_name_or_path $CKPT_INPUT/$MODEL_NAME_INPUT \
             --data_path $DATASETS/$DATA_FILE \
             --use_lora True \
             --q_lora False \
             --lora_r 8 \
             --lora_alpha 16 \
             --lora_dropout 0.05 \
             --convo_template $CONVO_TEMPLATE \
             --bf16 \
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
