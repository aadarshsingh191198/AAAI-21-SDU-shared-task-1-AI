import json
import pandas as pd
from tqdm import tqdm

def fixed_tags(tags):
    fixed = []
    cont = None
    for tag in tags:
        if tag == 'O':
            fixed.append(tag)
            cont = None
        else:
            if cont == tag:
                fixed.append(tag.replace("U","I"))
            else: 
                fixed.append(tag.replace("U","B"))
                cont = tag
    assert len(list(filter(lambda x: 'long' in x,fixed)))== len(list(filter(lambda x:'long' in x,tags)))
    assert len(list(filter(lambda x: 'short' in x,fixed)))== len(list(filter(lambda x:'short' in x,tags)))

    assert len(fixed) == len(tags)
    return fixed

def naive_fixed_tags(tags):
    fixed_tags_dict = {"U-short":"B-short","U-long":"B-long","L-short":"I-short","L-long":"I-long"}
    return [fixed_tags_dict.get(tag, tag) for tag in tags]

if __name__ == '__main__':
    f = open('predictions\\scibert_finetune\\finetune_predict.txt')
    predictions = f.readlines()
    preds = []
    for i,prediction in tqdm(enumerate(predictions)):
        prediction = json.loads(prediction)
        preds.append({"id":f"TS-{i}", "predictions":fixed_tags(prediction['tags'])})
        # preds.append({"id":f"TS-{i}", "predictions":naive_fixed_tags(prediction['tags'])})

    f = open('predictions\\scibert_finetune\\output.json','w')
    f.write(str(preds).replace("'",'"'))
    f.close()