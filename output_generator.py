import json
import pandas as pd
import numpy as np
from tqdm import tqdm
import sys
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

def softmax(x):
    return np.exp(x)/sum(np.exp(x))

def logit_to_preds(logits,mask,labels=["O","U-long","U-short"]):
    soft_logit = [softmax(logit) for logit in logits]
    max_logit = [np.argmax(logit) for logit in soft_logit]
    masked_logit = max_logit[:sum(mask)]
    logit_predictions = [labels[i] for i in masked_logit]
    return logit_predictions

def naive_fixed_tags(tags):
    fixed_tags_dict = {"U-short":"B-short","U-long":"B-long","L-short":"I-short","L-long":"I-long"}
    return [fixed_tags_dict.get(tag, tag) for tag in tags]

if __name__ == '__main__':
    #python output_generator.py <input_file> <output_file>
    f = open(sys.argv[1]) #predict.txt
    predictions = f.readlines()
    f.close()
    preds = []
    for i,prediction in tqdm(enumerate(predictions)):
        prediction = json.loads(prediction)
        logit_preds = logit_to_preds(prediction['logits'], prediction['mask'])
        preds.append({"id":f"TS-{i}", "predictions":fixed_tags(logit_preds)})
        # preds.append({"id":f"TS-{i}", "predictions":naive_fixed_tags(prediction['tags'])})
        # preds.append({"id":f"TS-{i}", "predictions":fixed_tags(prediction['tags'])})

        
    f = open(sys.argv[2],'w') #output.json
    f.write(str(preds).replace("'",'"'))
    f.close()
