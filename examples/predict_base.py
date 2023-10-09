from flagai.auto_model.auto_loader import AutoLoader


# 模型名称
model_name = 'AquilaChat2-7B'

# 加载模型以及tokenizer
autoloader = AutoLoader("aquila2", model_name=model_name)

model = autoloader.get_model()
tokenizer = autoloader.get_tokenizer()

# 对话测试样例
test_data = [
    "北京的十大景点是什么?请将回答翻译成英文和日语",
    "写一首中秋主题的五言绝句",
]

for text in test_data:
    print(model.predict(text, tokenizer=tokenizer, sft=False))
