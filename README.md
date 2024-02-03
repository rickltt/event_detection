# event detection

## Dataset

- ACE2005

- ACE2005+

- ERE

- MAVEN

## Baselines

The source codes for baselines, including [DMCNN](DMCNN),[BiLSTM,BiLSTM+CRF](BiLSTM),[BERT,BERT+CRF](BERT), [DMBERT](DMBERT).

## Data Split


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


