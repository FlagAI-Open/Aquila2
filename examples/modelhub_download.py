from flagai.model.file_utils import _get_model_id, _get_checkpoint_path, _get_vocab_path, _get_model_files

model_name = 'AquilaChat2-34B-AWQ'
download_path = f'Aquila2/checkpoints/{model_name}'

try:
    model_id = _get_model_id(model_name)
except:
    raise FileNotFoundError("Model name not found in local path and ModelHub")

if model_id and model_id != "null":
    model_files = eval(_get_model_files(model_name))
    print("model files:" + str(model_files))
    for file_name in model_files:
        if not file_name.endswith("bin"):
            _get_vocab_path(download_path, file_name, model_id)
        elif 'pytorch_model.bin' in model_files:
            checkpoint_path = _get_checkpoint_path(download_path, 'pytorch_model.bin', model_id)
        else:
            checkpoint_merge = {}
            for file_to_load in model_files:
                if "pytorch_model-0" in file_to_load:
                    _get_checkpoint_path(download_path, file_to_load, model_id)
