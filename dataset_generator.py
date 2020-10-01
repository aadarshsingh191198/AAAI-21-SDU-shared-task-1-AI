import pandas as pd
import json
import os

class Dataset():
    def __init__(self, filename):
        self.filename = filename
        self.output_file = self.filename.split('.')[0]+'.csv'
        self.convert_to_df()
        self.write_to_csv()
        self.show_csv()

    def convert_to_df(self):
        with open(self.filename) as file:
            data = json.load(file)
            dataset = [[sample['id'],' '.join(sample['tokens']), ' '.join(sample['labels'])] for sample in data]
        
        self.df = pd.DataFrame(dataset, columns = ['id','sentence','labels'])
       
    def write_to_csv(self):
        self.df.to_csv(self.output_file,index=False)

    def show_csv(self):
        print(pd.read_csv(self.output_file).head())

dev_data = Dataset('dataset\\dev.json')
train_data = Dataset('dataset\\train.json')
