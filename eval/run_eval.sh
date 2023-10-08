CKPT='./checkpoints-in/Aquila2-33B'

TASK=mmlu
python $TASK.py -d 0 -c $CKPT > $TASK.log

TASK=cmmlu
python $TASK.py -d 0 -c $CKPT > $TASK.log

TASK=ceval
python $TASK.py -d 0 -c $CKPT > $TASK.log

TASK=gsm8k
python $TASK.py -d 0 -c $CKPT > $TASK.log
