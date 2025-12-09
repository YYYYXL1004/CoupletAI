import os
import sys
import torch
from flask import Flask, request, render_template
import argparse
from module import Tokenizer, init_model_by_key, is_seq2seq_model

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str)
parser.add_argument("--device", default="cuda:4", type=str)
args = parser.parse_args()

MODEL_PATH = args.model_path


class Context(object):
    def __init__(self, path, device_name):
        print(f"loading pretrained model from {path}")
        self.device = torch.device(device_name)
        model_info = torch.load(path, map_location=self.device)
        self.tokenizer = model_info['tokenzier']
        self.model = init_model_by_key(model_info['args'], self.tokenizer)
        
        # 处理多卡训练保存的模型（去掉module.前缀）
        state_dict = model_info['model']
        if any(k.startswith('module.') for k in state_dict.keys()):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        
        # 检测是否为Seq2Seq模型
        self.use_seq2seq = is_seq2seq_model(self.model)
        self.punct_ids = self.tokenizer.get_punctuation_ids() if self.use_seq2seq else None
        print(f"Model: {self.model.__class__.__name__}, Seq2Seq: {self.use_seq2seq}")

    def predict(self, s):
        input_ids = torch.tensor(self.tokenizer.encode(s)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            if self.use_seq2seq:
                # Seq2Seq模型：带重复惩罚 + 标点对齐 + 强制等长
                logits = self.model.generate(
                    input_ids, 
                    repetition_penalty=1.5,
                    punctuation_ids=self.punct_ids,
                    force_same_length=True
                ).squeeze(0)
            else:
                # 原始模型：序列标注
                logits = self.model(input_ids).squeeze(0)
        pred = logits.argmax(dim=-1).tolist()
        pred = self.tokenizer.decode(pred)
        return pred
        
app = Flask(__name__)
ctx = Context(MODEL_PATH, args.device)

@app.route('/<coupletup>')
def api(coupletup):
    return ctx.predict(coupletup)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        return render_template("index.html")
    coupletup = request.form.get("coupletup")
    coupletdown = ctx.predict(coupletup)
    return render_template("index.html", coupletdown=coupletdown)


if __name__ == '__main__':
    app.run(host='0.0.0.0')
