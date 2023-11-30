import torch.cuda
import torch.backends
import os
import logging
import uuid

LOG_FORMAT = "%(levelname) -5s %(asctime)s" "-1d: %(message)s"
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logging.basicConfig(format=LOG_FORMAT)

llm_model_dict = {
    "aquilachat2-34b": {
        "name": "aquilachat2-34b",
        "pretrained_model_name": "aquilachat2-34b",
        "local_model_path": None,
        "provides": "Aquila"
    },
}

# LLM 名称
LLM_MODEL = ""aquilachat2-34b""

# LLM running device
LLM_DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# 基于上下文的prompt模版，请务必保留"{question}"和"{context}"
PROMPT_TEMPLATE = """The following information is known:
{context}

Based on the above known information, answer the user's questions concisely and professionally. If you can't get an answer from it, please say "the question cannot be answered based on known information" or "not enough relevant information is provided". Adding fabricated components to the answer is not allowed, and the answer should be in English. The question is：{question}"""

FLAG_USER_NAME = uuid.uuid4().hex

