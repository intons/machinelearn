from transformers import AutoTokenizer
import warnings
from transformers.utils.generic import is_tf_tensor

# 临时覆盖 TensorFlow 检查函数
def is_tf_tensor_override(obj):
    return False

is_tf_tensor.__code__ = is_tf_tensor_override.__code__

sentence = "Hello World!"
tokenizer = AutoTokenizer.from_pretrained("Xenova/gpt-4")
token_ids = tokenizer(sentence).input_ids
print("Token IDs:", token_ids)

# 正确解码
decoded_sentence = tokenizer.decode(token_ids)
print("Decoded sentence:", decoded_sentence)

# 查看每个 token 的解码结果
print("\n单个 token 的解码结果:")
for token_id in token_ids:
    token = tokenizer.convert_ids_to_tokens([token_id])[0]
    print(f"Token ID: {token_id}, Token: {token}")