import torch

from flagai.auto_model.auto_loader import AutoLoader
from flagai.model.aquila2.utils import covert_prompt_to_input_ids_with_history

from threading import Thread
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer

model_name = 'AquilaChat2-34B'
convo_template = 'aquila-legacy'

#model_name = 'AquilaChat2-7B'
#convo_template = 'aquila-v1'

autoloader = AutoLoader("aquila2", model_name=model_name, all_devices=True)

model = autoloader.get_model()
tokenizer = autoloader.get_tokenizer()

test_data = [
    "北京的十大景点是什么?",
    "写一首中秋主题的五言绝句",
    "Write a tongue twister that's extremely difficult to pronounce.",
]

history = None
device = 'cuda'
for text in test_data:
    print(f"=======================================================\n")
    inputs = tokenizer.encode_plus(text)
    tokens = covert_prompt_to_input_ids_with_history(text, history=history, tokenizer=tokenizer, max_token=20480, convo_template=convo_template)
    del inputs['attention_mask']
    inputs['input_ids'] = torch.tensor(tokens)[None,].to(device)
    streamer = TextIteratorStreamer(tokenizer=tokenizer)

    # Arguments
    inputs.update({"streamer": streamer, "max_new_tokens": 50})

    # Generation
    thread = Thread(target=model.generate, kwargs=inputs)
    thread.start()

    for token in streamer:
        print(token, end="")

    thread.join()
