# event detection

## Dataset

- ACE2005

- ACE2005+

- ERE

- MAVEN

## Baselines

The source codes for baselines, including [DMCNN](DMCNN),[BiLSTM,BiLSTM+CRF](BiLSTM),[BERT,BERT+CRF](BERT), [DMBERT](DMBERT).

## Results

### Data Split


| Dataset    | #Class | #Train | #Dev | #Test |
| ----------- | ----------- | ----------- | ----------- | ----------- |
| ACE2005      | 33   | 17172   | 923      | 832 |
| ACE2005+   | 33    |  19216    | 901 | 676    |
| MAVEN   | 168     | 32378       |  4047| 4047     |
| ERC  | 38     | 14722       |  1209| 1163    |

For ACE2005, we follow [Entity, relation, and event extraction with contextualized span representations.](https://arxiv.org/abs/1909.03546)

For ACE2005+, we follow [A Joint Neural Model for Information Extraction with Global Features.](https://aclanthology.org/2020.acl-main.713/)

For [MAVEN](https://github.com/THU-KEG/MAVEN-dataset/blob/main/DataFormat.md), we merged the training and validation sets due to the hide of golden labels, and we split train/dev/test set according to the ratio of 0.8/0.1/0.1.

For ERC, we follow [DEGREE: A Data-Efficient Generation-Based Event Extraction Model.](https://arxiv.org/abs/2108.12724)


### Trigger Classification (Precision,Recall,micro-F1)

| Model      | ACE2005 | ACE2005+ | Duee1.0 |
| ----------- | ----------- | ----------- | ----------- |
| BERT      | (69.63, 75.25, 72.33)    | (72.41, **74.29**,73.34)     | (85.42,	**87.04**,	**86.22**)       |
| BERT+CRF   | (68.11,**75.51**,71.62)     | (71.85, 74.06,	72.94)       | (84.53, 86.70, 85.60)      |
| BiLSTM   | (73.64, 72.36,	72.99)       | (73.55, 71.30, 72.41)      |  (80.40, 69.88, 74.77)       |
| BiLSTM+CRF   | (74.47, 71.46, 72.94)      | (**79.89**, 70.28, **74.78**)      | (83.70, 75.44, 79.36)      |
| DMCNN   | (74.33, 66.84, 70.38)      | (67.02, 73.54, 70.13)       |  (81.96, 65.98, 73.11)      |
| DMBERT   | (**76.47**, 71.92, **74.12**)      | (74.93, 73.54, 74.23)       | (**88.96**, 81.72, 85.18)      |
| EDPRC   | (76.14, 71.58, 73.79)     | (74.63, 73.26, 73.94)         | (83.77, 83.68, 83.73)      |
