from lime import lime_text
from lime.lime_text import LimeTextExplainer
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import torch
import numpy as np

# setting
'''
첫 실행 시 다운로드가 필요해 미리 할당
'''
model = AutoModelForSequenceClassification.from_pretrained('JminJ/kcElectra_base_Bad_Sentence_Classifier')
tokenizer = AutoTokenizer.from_pretrained('JminJ/kcElectra_base_Bad_Sentence_Classifier')
batch_size = 256
classifier = pipeline("text-classification", model=model,tokenizer=tokenizer,batch_size=batch_size)

def mk_predlist(inps):
    outputs = classifier(inps)
    result = []
    for output in outputs:
        if output['label'] == 'bad_sen':
            result.append([output['score'], 1-output['score']])
        else:
            result.append([1-output['score'], output['score']])
    return np.array(result)

explainer = LimeTextExplainer(class_names=['bad','ok'])

# 실행
test_text = "" # 테스트할 텍스트 입력

if classifier(test_text)[0]['label'] == 'bad_sen':
    print('비속어 있음')
    exp = explainer.explain_instance(test_text, mk_predlist,top_labels=1)

    badword = exp.as_list(label=0)[0][0]

else:
    print('비속어 없음')