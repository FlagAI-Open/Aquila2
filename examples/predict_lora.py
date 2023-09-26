from flagai.auto_model.auto_loader import AutoLoader


state_dict = "./checkpoints/"
model_name = 'Aquila2Chat-hf'

autoloader = AutoLoader("aquila2",
                    model_dir=state_dict,
                    model_name=model_name,
                    lora_dir='/data2/yzd/git/Aquila2/examples/checkpoints/lora/aquila2chat-hf')

model = autoloader.get_model()
tokenizer = autoloader.get_tokenizer()
# 

test_data = [
    "请介绍下北京有哪些景点。",
    "唾面自干是什么意思",
    "'我'字有几个笔划",
]

for text in test_data:
    print(model.predict(text, tokenizer=tokenizer))
