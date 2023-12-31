### 调整参数
下面是微调过程中一些可调整的重要参数:

|   参数名          |  类型   | 描述  |                                  
| :---------------- | :------- | :-- |   
| per_device_train_batch_size         | int  |   每次迭代训练时，从数据集中抽取的样本数。一般来说，它越大，处理速度越快，但会占用更多的内存。   |
| gradient_accumulation_steps          |int  |    在更新模型权重之前，要对多个小批次进行梯度计算的次数。主要应用于GPU显存较小的情况下，可以使用小的batch_size，通过梯度累积达到与大batch_size相同的效果。    |
| learning_rate          |float  |    指控制模型更新参数时的步长或速率。学习率过高可能导致模型不收敛，而学习率过低则可能导致训练时间过长或者陷入局部最优解。   |  
| gradient_checkpointing           |bool |    一种内存优化技术，用于减少神经网络训练过程中的 GPU 或其他计算设备的内存使用量。这种技术特别有用对于那些有限的硬件资源，但需要训练大型神经网络的情况。 | 
| warmup_ratio           |float |   初始学习率与原始学习率的比例。     | 
| save_strategy          | str  |    保存模型权重的策略，当训练时间较长时，保存间隔可以避免因突然中断或出现错误导致训练成果全部丢失; 可选项有: 1.'epoch'代表在每一轮训练结束时保存权重 2. 'steps'代表每隔一定步数保存一次模型，具体的步数在`save_steps`参数里指定。   |   
| logging_steps           |int  |    日志输出的间隔，即每训练多少个iteration输出一次日志信息。    | 
| use_lora           |bool  |    是否启用lora微调。   | 
| q_lora           |bool  |    是否启用qlora微调, 需要`use_lora`为true时，此参数才会生效。   | 
| lora_r          |int  |    `lora_r`是低秩适应的秩。这个参数控制了低秩适应的复杂性。通过调整`lora_r`的值，可以控制降维的程度，从而影响模型的性能和效率。较小的 lora_r 值可能导致更简单、更快的模型，但可能牺牲一些性能。    |  
| lora_alpha           |int  |    在 LoRA 中，`lora_alpha`和`lora_r`的比率通常用来调整低秩适应层的学习率。具体来说，该比率将决定低秩适应层的学习率相对于原始模型其他部分的学习率的倍数。通过调整这个比率，你可以控制 LoRA 层参数更新的速度。   | 
| lora_dropout           |float |    `lora_dropout`是dropout率。在深度学习和神经网络中，dropout是一种正则化技术，通过随机关闭一部分神经元来防止过拟合。   | 
