from flagai.auto_model.auto_loader import AutoLoader


state_dict = "./checkpoints/"
model_name = 'Aquila2Chat-hf'

state_dict = "/data2/20230907/"
model_name = 'iter_0205000_hf'


autoloader = AutoLoader("aquila2",
                    model_dir=state_dict,
                    model_name=model_name,
                    # lora_dir='/data2/yzd/git/Aquila2/examples/checkpoints/lora/aquila2chat-hf')
                    qlora_dir='/data2/yzd/FastChat/checkpoints_out/30bhf_save/checkpoint-6000')

model = autoloader.get_model()
tokenizer = autoloader.get_tokenizer()
# 

test_data = [
        "北京的十大景点是什么?请将回答翻译成英文和日语",
    "写一首中秋主题的五言绝句并翻译成英文和韩语",
]

for text in test_data:
    print(model.predict(text, tokenizer=tokenizer))
