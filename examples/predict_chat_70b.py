from flagai.auto_model.auto_loader import AutoLoader


model_name = 'AquilaChat2-70B-Expr'

autoloader = AutoLoader("aquila2", model_name=model_name, all_devices=True)

model = autoloader.get_model()
tokenizer = autoloader.get_tokenizer()

test_data = [
    "北京的十大景点是什么?",
    "写一首中秋主题的五言绝句",
    "Write a tongue twister that's extremely difficult to pronounce.",
]

for text in test_data:
    print(model.predict(text, tokenizer=tokenizer, model_name=model_name, top_p=0.9, seed=123, topk=15, temperature=1.0))
