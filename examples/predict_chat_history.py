from flagai.auto_model.auto_loader import AutoLoader

model_name = 'AquilaChat2-34B'

autoloader = AutoLoader("aquila2", model_name=model_name)

model = autoloader.get_model()
tokenizer = autoloader.get_tokenizer()

test_data = [
    "怎样才能在有限的时间内学习新技能？",
    "我正在学习编程，有没有一些针对初学者的在线教程推荐？",
    "我想学习Python编程语言，哪个在线教育平台提供的课程最适合我？",
]

history = []
for text in test_data:
    print(f"=============================================================================")
    print(model.predict(text, tokenizer=tokenizer, model_name=model_name, history=history, top_p=0.9, seed=123, topk=15, temperature=1.0))
