# SDU@AAAI-21 - Shared Task 1: Acronym Identification

This repository contains the following:
* acronym identification training and development set along with the evaluation scripts for the [acronym identification task at SDU@AAAI-21](https://sites.google.com/view/sdu-aaai21/shared-task)[*]
* scripts to generate data as per usecase[+]
* notebooks containing training, validation and prediction code[+]
* scripts to generate output as per the required format[+]

[*] - Already present in the original repository
[+] - Added by me

# Dataset

The dataset folder contains three files:

- **train.json**: The training samples for acronym identification task. Each sample has three attributes:
  - tokens: The list of words (tokens) of the sample
  - labels: The short-form and long-form labels of the words in BIO format. The labels `B-short` and `B-long` identifies the beginning of a short-form and long-form phrase, respectively. The labels `I-short` and `I-long` indicates the words inside the short-form or long-form phrases. Finally, the label `O` shows the word is not part of any short-form or long-form phrase. 
  - id: The unique ID of the sample
- **dev.json**: The development set for acronym identification task. The samples in `dev.json` have the same attributes as the samples in `train.json`.
- **predictions.json**: A sample prediction file created from `dev.json` to test the scoring script. The participants should submit the final test predictions of their model in the same format as the `predictions.json` file. Each prediction should have two attributes:
  - id: The ID of the sample (i.e., the same IDs used in the train/dev/test samples provided in `train.json`, `dev.json` and `test.json`) 
  - predictions: The labels of the words of the sample in BIO format. The labels `B-short` and `B-long` identifies the beginning of a short-form and long-form phrase, respectively. The labels `I-short` and `I-long` indicates the words inside the short-form or long-form phrases. Finally, the label `O` shows the word is not part of any short-form or long-form phrase.

# Code
In order to familiarize the participants with this task, we provide a rule-based model in the `code` directory. This baseline implements the method proposed by [Schwartz and Hearst](http://psb.stanford.edu/psb-online/proceedings/psb03/schwartz.pdf) [1]. To identify acronyms, if more than 60% of the characters of a word are uppercased, this model recognizes it as acronym (i.e., short-form). To identify the long-form, it compares the characters of the acronym with the characters of the words that are before or after the acronym up to a certain window size. If the characters of these words could form the acronym, they are labeled as long-form. To run this model, use the following command:

`python code/character_match.py -input <path/to/input.json> -output <path/to/output.json>`

Please replace the `<path/to/input.json>` and `<path/to/output.json>` with the real paths to the input file (e..g, `dataset/dev.json`) and output file. The output file contains the predictions and can be evaluated by the scorer using the command described in the next section. The official scores for this baseline are: *Precision: 93.22%, Recall: 78.90%, F1: 85.46%*

# Scripts

- **dataset_generator.py**: Convert the json files to csv
- **dataset_reformatter.py**: Convert the generated csvs to CoNLL-2003 format
- **output_generator.py**: Convert the prediction files generated by Scibert to the required format.

# Evaluation

To evaluate the predictions (in the format provided in `dataset/predictions.json` file), run the following command:

`python scorer.py -g path/to/gold.json -p path/to/predictions.json`

The `path/to/gold.json` and `path/to/predictions.json` should be replaced with the real paths to the gold file (e.g., `dataset/dev.json` for evaluation on development set) and predictions file (i.e., the predictions generated by your system in the same format as `dataset/predictions.json` file). The official evaluation metrics are the macro-averaged precision, recall and F1 for short form and long form predictions. For verbose evaluation (including the micro-averaged precision, recall and F1 and also short form and long form scores seperatedly), use the following command:

`python scorer.py -g path/to/gold.json -p path/to/predictions.json -v`

**The CodaLab competition** - [Acronym Identification](https://competitions.codalab.org/competitions/26609).

**Parent repo** - https://github.com/amirveyseh/AAAI-21-SDU-shared-task-1-AI)

# References
[1] Schwartz AS, Hearst MA. A simple algorithm for identifying abbreviation definitions in biomedical text. Pac Symp Biocomput. 2003:451-62. PMID: 12603049.
[2] Scibert: https://github.com/allenai/scibert
[3] Bert-sklearn: https://github.com/charles9n/bert-sklearn
