from flagai.auto_model.auto_loader import AutoLoader


model_name = 'AquilaChat2-7B'

autoloader = AutoLoader("aquila2",model_name=model_name,
                    lora_dir='/data2/yzd/git/Aquila2/examples/checkpoints/lora/aquila2chat-hf')
                    # qlora_dir='/data2/yzd/FastChat/checkpoints_out/30bhf_save/checkpoint-6000')

model = autoloader.get_model()
tokenizer = autoloader.get_tokenizer()

test_data = [
    "北京的十大景点是什么?请将回答翻译成英文和日语",
    "写一首中秋主题的五言绝句",
]

for text in test_data:
    print(model.predict(text, tokenizer=tokenizer))
