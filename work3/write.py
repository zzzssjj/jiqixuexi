import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline

pipe = pipeline("text-generation", model="uer/gpt2-chinese-cluecorpussmall")
model_name = "uer/gpt2-chinese-cluecorpussmall"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

def generate_text(prompt, max_length=100, num_return_sequences=1):
    # 对输入进行编码
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # 生成文本
    outputs = model.generate(
        inputs.input_ids,
        max_length=max_length,
        num_return_sequences=num_return_sequences,
        no_repeat_ngram_size=2,
        repetition_penalty=1.5,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
    )
    
    # 解码并返回结果
    generated_texts = []
    for output in outputs:
        text = tokenizer.decode(output, skip_special_tokens=True)
        generated_texts.append(text)
    
    return generated_texts


prompt = "假如我能隐身一天，我会"
result = generate_text(prompt, max_length=50)
print(result[0])
