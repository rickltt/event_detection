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
| ERE  | 38     | 14722       |  1209| 1163    |

For ACE2005, we follow [Entity, relation, and event extraction with contextualized span representations.](https://arxiv.org/abs/1909.03546)

For ACE2005+, we follow [A Joint Neural Model for Information Extraction with Global Features.](https://aclanthology.org/2020.acl-main.713/)

For [MAVEN](https://github.com/THU-KEG/MAVEN-dataset/blob/main/DataFormat.md), we merged the training and validation sets due to the hide of golden labels, and we split train/dev/test set according to the ratio of 0.8/0.1/0.1.

For ERE, we follow [DEGREE: A Data-Efficient Generation-Based Event Extraction Model.](https://arxiv.org/abs/2108.12724)


### Trigger Classification (Precision,Recall,micro-F1)


| Model      | ACE2005     | ACE2005+    | ERE         | MAVEN       |
| -----------| ----------- | ----------- | ----------- | ----------- |
| BERT       | (67.63, 71.21, 69.37)  |  (68.40, 71.46, 69.90)  | (54.94, 59.53, 57.14)  | (63.70, 68.51, 66.02) |
| BERT+CRF   | (68.20, **74.75**, **71.33**)  |  (71.88, **74.76**, 73.29)  | (55.74, **59.89**, 57.74)  | (63.77, 68.42, 66.02) |
| BiLSTM     | (72.92, 61.87, 66.94)  |  (76.36, 56.37, 64.86)  | (54.29, 41.38, 46.96)  | (70.31, 56.09, 62.40) |
| BiLSTM+CRF | (71.79, 57.83, 64.06)  |  (76.01, 57.55, 65.50)  | (57.77, 40.47, 47.60)  | (69.67, 57.83, 63.20) |
| DMCNN      | (69.72, 50.00, 58.24)  |  (62.93, 52.44, 57.21)  | (45.53, 30.14, 36.27)  | (70.18, 46.06, 55.62) |
| DMBERT     | (61.95, 64.14, 63.03)  |  (64.19, 63.33, 63.76)  | (55.99, 54.53, 55.25)  | (65.18, **68.86**, 66.97) |
| EDPRC      | (**79.87**, 62.28, 69.99)  |  (**81.29**, 67.15, **73.54**)  | (**61.85**, 54.63, **58.01**)  | (**73.15**, 64.77, **68.71**) | 
