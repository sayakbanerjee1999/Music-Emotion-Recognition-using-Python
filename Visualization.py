import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv('Dataset/Emotion_data.csv')
feature = data.loc[:, 'tempo':]
target = data['label']
targetName = data['class']
featureName = list(feature)
color = ['red' if l==1 else 'green' if l==2 else 'blue' if l==3 else 'orange' for l in data['label']]

for name in featureName:
    feature[name] = (feature[name]-feature[name].min())/(feature[name].max()-feature[name].min())

feature['class'] = data['class']

feature_mean = feature.loc[:, 'chroma_stft_mean'::3]

feature_std = feature.loc[:, 'chroma_stft_var'::3]

feature_var = feature.loc[:, 'chroma_stft_std'::3]

fig, ax = plt.subplots(figsize=(12, 12))
sns.heatmap(feature[feature['class']=='sad'].iloc[:, :-1], linewidths=0.5, ax=ax, cmap='Greens')
plt.show()