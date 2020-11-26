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

if __name__ == '__main__':
    f = open('predict.txt')
    predictions = f.readlines()
    preds = []
    for i,prediction in tqdm(enumerate(predictions)):
        prediction = json.loads(prediction)
        preds.append({"id":f"TS-{i}", "tags":fixed_tags(prediction['tags'])})

    f = open('output.json','w')
    f.write(str(preds).replace('\'',))
    f.close()