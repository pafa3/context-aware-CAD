**Introduction**
This repository contains the code and resources for my master's thesis project, "Context-Aware Hate Speech Detection using BERT: An Investigation with the Contextual Abuse Dataset," submitted for the degree of MA Linguistics (Text Mining) at Vrije Universiteit Amsterdam.
Superviosrs:
Isa maks
Ilia Markov

**Project Overview**
The prevalence of toxic language and hate speech on online platforms has become a significant concern, necessitating the development of effective automatic detection methods. This project explored three experimental conditions:

Using only the comment text
Combining the comment with its preceding parent comment
Incorporating the entire conversation thread

**Repository Contents**
**bert**
Bulding upon Vidgen et al. (2021) work, contains BERT model implementations and related scripts.
**Code**
analysis.ipynb  - provides an analysis of the dataset
cad_v1_1.ipynb - contains the  code for creating the new dataset that contains parents comments
error_analysis.ipynb - Creates error analysis files and provides confusion matrix and classification report

**Data**
cad_v1_1.tsv: The original dataset by Vidgen et al. (2021) (https://aclanthology.org/2021.naacl-main.182.pdf)
cad_v1_1_parents.csv: The transofmred dataset that was developed for this project which includes parent comments.

**error_analysis**
filtered_error_analysis.csv: Dataset filtered to include only rows where there's a disagreement between labels_info and prediction_level_3
sampled_error_analysis.csv: Random samples from filtered_error_analysis with error analysis performed on the first 100 samples
sampled_error_analysis3.csv: Disagreement between predictions with and without context
sampled_error_analysis3_2.csv: Disagreement with additional error analysis

**Result**
contains the output of the BERT models
