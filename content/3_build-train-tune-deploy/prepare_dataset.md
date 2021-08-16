---
title: "3.1 Prepare your dataset and upload it to Amazon S3"
weight: 1
---

{{% notice tip %}}
Watch the livestream to follow along with the presenter
{{% /notice %}}

## Open the following notebook to follow along

Notebook: `1_prepare_dataset.ipynb`

![](/images/setup/setup14.png)

{{% notice info %}}
A copy of the code from the notebook is also available below, if you prefer building your notebook from scratch by copy pasting each code cell and then running them.
{{% /notice %}}

## Preparing your dataset


```python
import sagemaker
import boto3
import pandas as pd
import numpy as np
```


```python
sess = boto3.Session()
sagemaker_session = sagemaker.Session()
role = sagemaker.get_execution_role()

bucket = sagemaker_session.default_bucket()
prefix = "sagemaker_huggingface_workshop"

print(f"sagemaker role arn: {role}")
print(f"sagemaker bucket: {sagemaker_session.default_bucket()}")
print(f"sagemaker session region: {sagemaker_session.boto_region_name}")
```

```python
df = pd.read_csv('./data/Womens Clothing E-Commerce Reviews.csv')
df = df[['Review Text',	'Rating']]
df.columns = ['text', 'label']
df['label'] = df['label'] - 1

df = df.dropna()

train, validate, test = \
              np.split(df.sample(frac=1, random_state=42),
                       [int(.6*len(df)), int(.8*len(df))])

train.shape, validate.shape, test.shape
```


```python
train.head(10)
```


```python
train.to_csv(   './data/train.csv'   , index=False)
validate.to_csv('./data/validate.csv', index=False)
test.to_csv(    './data/test.csv'    , index=False)
```


```python
dataset_path = sagemaker_session.upload_data(path='data', key_prefix=f'{prefix}/data')
print(f'Dataset location: {dataset_path}')
```
