import nltk
import pandas as pd

def reformat_test(x, test = False):  
  tok_text = nltk.word_tokenize(x['sentence'])
  word_pos = nltk.pos_tag(tok_text)
  return '\n'.join([f'{word} {pos} O O' for (word,pos) in word_pos])

def reformat(x):  
  tok_text = nltk.word_tokenize(x['sentence'])
  tags = x['labels'].split()
  word_pos = nltk.pos_tag(tok_text)
  return '\n'.join([f'{word} {pos} O {i}' for (word,pos),i in zip(word_pos,tags)]) #Not very sure as to why the third element is O.

def make_data(filename):
    df = pd.read_csv(f'./dataset/{filename}.csv')
    # train = pd.read_csv('/content/AAAI-21-SDU-shared-task-1-AI/dataset/train.csv')

    if filename == 'test':
        df['reformatted_data'] = df[['sentence']].apply(reformat_test,axis=1)
    else:
        df['reformatted_data'] = df[['sentence','labels']].apply(reformat,axis=1)
    # train['reformatted_data'] = train[['sentence','labels']].apply(reformat,axis=1)
    
    with open(f'dataset/scibert_sduai/{filename}.txt','w', encoding='utf-8') as f:
        f.write('\n\n'.join(df['reformatted_data'].tolist()))

    print(f'{filename} data reformatted and stored in scibert_sduai/{filename}.txt..')

if __name__ == '__main__':
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    make_data('train')
    make_data('dev')
    make_data('test')
