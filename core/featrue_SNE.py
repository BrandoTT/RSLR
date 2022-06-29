import torch
import numpy as np
from sklearn.manifold import TSNE
from pandas import DataFrame
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns

features = '/home/algroup/czt/DANN_DivideMix_0915/DANN_DivideMix_New/DANN_DivideMix/DANN_py3-master/Record/office31/A_W/features/RSLR_features_target_office31_office_a_w.npy'
features = np.load(features)
# Art
target_file = '/home/algroup/czt/DANN_DivideMix_0915/DANN_DivideMix_New/DANN_DivideMix/DANN_py3-master/dataset/office31/webcam.txt'
with open(target_file, 'r') as f:
    file_dir, true_labels = [], []
    for i in f.read().splitlines():
        file_dir.append(i.split(' ')[0])
        true_labels.append(int(i.split(' ')[1]))
Y = np.array(true_labels)
X_embedded = TSNE(n_components=2).fit_transform(features)
data = np.column_stack((X_embedded, Y))
df = DataFrame(data, columns=['DIM_1', 'DIM_2', 'Label'])
df = df.astype({'Label':'int'})
df.dtypes
sns.set_context('notebook', font_scale=2.)
sns.set_style("darkgrid")
fig, (ax1, ax, ax3, ax4) = plt.subplots(ncols=4,nrows=1,figsize=(46,10))
# fig 1
sns.scatterplot(
    x='DIM_1',
    y='DIM_2',
    hue='Label',
    data=df,
    palette='viridis',
    ax=ax
)
# fig 2
sns.scatterplot(
    x='DIM_1',
    y='DIM_2',
    data=df,
    hue='Label',
    palette='viridis',
    ax=ax1
)
norm = plt.Normalize(0, 64)
sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
sm.set_array([])
ax.get_legend().remove()
ax1.get_legend().remove()
ax.figure.colorbar(sm)
plt.xlabel('Dimension 1').set_fontsize('20')
plt.ylabel('Dimension 2').set_fontsize('20')
plt.show()
plt.savefig("RSLR_feature_office31.jpg")









features_webcam = '/home/algroup/czt/DANN_DivideMix_0915/DANN_DivideMix_New/DANN_DivideMix/DANN_py3-master/Record/office31/A_W/features/RSLR_features_target_office31_office_a_w.npy'
#features_amacon = '/home/algroup/czt/DANN_DivideMix_0915/DANN_DivideMix_New/DANN_DivideMix/DANN_py3-master/Record/office31/W_A/features/RSLR_features_target_office31_office_w_a.npy'
features_webcam = np.load(features_webcam)
#features_amacon = np.load(features_amacon)
# Art
target_file_webcam = '/home/algroup/czt/DANN_DivideMix_0915/DANN_DivideMix_New/DANN_DivideMix/DANN_py3-master/dataset/office31/webcam.txt'
target_file_amazon = '/home/algroup/czt/DANN_DivideMix_0915/DANN_DivideMix_New/DANN_DivideMix/DANN_py3-master/dataset/office31/amazon.txt'

with open(target_file_webcam, 'r') as f:
    file_dir, true_labels = [], []
    for i in f.read().splitlines():
        file_dir.append(i.split(' ')[0])
        true_labels.append(int(i.split(' ')[1]))
Y = np.array(true_labels)
X_embedded = TSNE(n_components=2).fit_transform(target_file_webcam)
data = np.column_stack((X_embedded, Y))
df_a_w = DataFrame(data, columns=['DIM_1', 'DIM_2', 'Label'])
df_a_w = df_a_w.astype({'Label':'int'})
df_a_w.dtypes

with open(target_file_amazon, 'r') as f:
    file_dir, true_labels = [], []
    for i in f.read().splitlines():
        file_dir.append(i.split(' ')[0])
        true_labels.append(int(i.split(' ')[1]))
Y = np.array(true_labels)
X_embedded = TSNE(n_components=2).fit_transform(target_file_amazon)
data = np.column_stack((X_embedded, Y))
df_w_a = DataFrame(data, columns=['DIM_1', 'DIM_2', 'Label'])
df_w_a = df_w_a.astype({'Label':'int'})
df_w_a.dtypes



