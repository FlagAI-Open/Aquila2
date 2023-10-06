# Initialization
AQUILA2_HOME=/data2/yzd/git/Aquila2/
export PYTHONPATH=$AQUILA2_HOME:$PYTHONPATH

set -u
  EXPNAME=$1
set +u
EXPNAME_PATH=${AQUILA2_HOME}/output/logs/$EXPNAME
mkdir -p $EXPNAME_PATH
cp $0 $EXPNAME_PATH/

# 7B 
CKPT_INPUT=$AQUILA2_HOME/examples/checkpoints
MODEL_NAME_INPUT=aquila2chat-hf

DATASETS=/data2/20230907
CKPT_OUTPUT=$AQUILA2_HOME/output/checkpoints
LOGFILE=$AQUILA2_HOME/log.txt
LOGFILE=$AQUILA2_HOME/output/logs/log.txt.$EXPNAME
DEEPSPEED_CONFIG=ds_zero2.config
HOSTFILE=hostfile

DATA_FILE=sft_v0.9.12_train.jsonl
MODEL_NAME_OUTPUT=$MODEL_NAME_INPUT-sft-$EXPNAME

EPOCHS=5
CONVO_TEMPLATE=aquila-v1



# Training Proces


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
