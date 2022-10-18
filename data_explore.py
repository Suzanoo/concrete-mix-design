import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

CURR = os.getcwd()
df = pd.read_csv(os.path.join(CURR, 'dataset', 'concrete_data.csv'))
df.columns = ['cement', 'slag', 'flyash', 'water', 'superplasticizer',
             'coarse_agg', 'fine_agg', 'age', 'strength']
df.dtypes
df.shape
df.isnull().sum()
df.describe().T

# ========================
# Visualization
## pairplot 
sns.pairplot(df, diag_kind='kde')

## correlation plot
plt.figure(figsize=(15,8))
sns.heatmap(df.corr(),vmax=1, square=True, annot=True, cmap='viridis')
plt.title('Correlation between different attributes')
plt.show()

## boxplot
df.boxplot(figsize=(15,8))
plt.show()

# ========================
## See outliers
def outliers(item):
    X = df[item]
    N = X[np.abs(((X - np.mean(X))/np.std(X))) > 3].count()
    print(f"Outliers in {item} : {N}")

for item in df.columns:
    outliers(item)

## replace outliers by median
for cols in df.columns[:-1]:
  Q1 = df[cols].quantile(0.25)
  Q3 = df[cols].quantile(0.75)
  iqr = Q3 - Q1
  low = Q1-1.5*iqr
  high = Q3+1.5*iqr
  df.loc[(df[cols] < low) | (df[cols] > high), cols] = df[cols].median()

df.boxplot(figsize=(15,8))
plt.show()

df.to_csv(os.path.join(CURR, 'dataset/Concrete_Data_edit.csv'), encoding='utf-8', index=False)

# ========================

