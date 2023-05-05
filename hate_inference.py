from transformers import AutoTokenizer, BertForSequenceClassification, TextClassificationPipeline

# model_output 폴더 path
PATH = './model_output'

model = BertForSequenceClassification.from_pretrained(
    PATH
)

'''pretrained model 불러오기'''
tokenizer = AutoTokenizer.from_pretrained(PATH)

pipe = TextClassificationPipeline(
    model = model,
    tokenizer = tokenizer,
    device="cpu",
    return_all_scores=True,
    function_to_apply='sigmoid'
    )

def get_predicated_label(output_labels, min_score):
    labels = []
    for label in output_labels:
        if label['score'] > min_score:
            labels.append(1)
        else:
            labels.append(0)
    return labels


'''사용자 입력 inference(사용자 텍스트 입력 시 이 부분만 반복실행)'''
text = '테스트 중이야'

out = get_predicated_label(pipe(text)[0], 0.5)
if out[9] == 1:
    result = 'clean'
else:
    result = 'notClean'
print(result)
