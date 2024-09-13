import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd

rcParams['font.family'] = 'Times New Roman'

def categorize_platform(df):
    df['PlatformCategory'] = df['PlatformType'].apply(lambda x: 'PC' if x in ['win', 'steam'] else 'Mobile')
    return df

exclude_columns = ['PlayerID', 'MatchID', 'PlatformType', 'PlatformCategory', 'HeroId']

def load_and_prepare_data(file_path):
    data = pd.read_csv(file_path)
    data = categorize_platform(data)
    features = data.drop(columns=exclude_columns)
    feature_names = features.columns.tolist()
    features_scaled = StandardScaler().fit_transform(features.fillna(0))
    return data, features_scaled, feature_names

data_cheaters, features_scaled_cheaters, column_names_cheaters = load_and_prepare_data('BlackEigenvalue_matchBehavior.csv')
data_normal, features_scaled_normal, column_names_normal = load_and_prepare_data('WhiteEigenvalue_matchBehavior.csv')

cornflowerblue_rgb = (130/255, 176/255, 210/255)
indianred_rgb = (250/255, 127/255, 111/255)

# def pca_and_plot(features_scaled_subset, feature_names, title, ax):
#     pca = PCA().fit(features_scaled_subset)

#     loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

#     circle = plt.Circle((0, 0), 1, facecolor='none', edgecolor=cornflowerblue_rgb, linestyle='--', lw=2)
#     ax.add_artist(circle)
#     ax.set_xlim(-1.1, 1.1)
#     ax.set_ylim(-1.1, 1.1)

#     ax.axhline(0, color='grey', lw=1, linestyle='-.', alpha=0.7)
#     ax.axvline(0, color='grey', lw=1, linestyle='-.', alpha=0.7)

#     for j in range(loadings.shape[0]):
#         ax.arrow(0, 0, loadings[j, 0], loadings[j, 1], color=indianred_rgb, 
#                  head_width=0.03, head_length=0.04, alpha=0.85)
#         x_adjust = 0.065 if loadings[j, 0] > 0 else -0.05
#         y_adjust = 0.03 if loadings[j, 1] > 0 else 0.0
#         ax.text(loadings[j, 0] + x_adjust, loadings[j, 1] + y_adjust, feature_names[j], 
#                 color='black', ha='center', va='center', fontsize=20)

#     ax.spines['top'].set_linewidth(1.5)
#     ax.spines['right'].set_linewidth(1.5)
#     ax.spines['left'].set_linewidth(1.5)
#     ax.spines['bottom'].set_linewidth(1.5)

#     ax.set_xlabel("Principal Component 1", fontsize=24, weight='bold')
#     ax.set_ylabel("Principal Component 2", fontsize=24, weight='bold')
#     ax.set_title(title, fontsize=28, weight='bold')
#     ax.grid(True, linestyle=':', alpha=0.5)

#     ax.tick_params(axis='both', which='major', labelsize=16)

# platforms = ['PC', 'Mobile']
# cheater_conditions = ['Cheaters', 'Normal Players']

# fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(24, 24))

# for i, platform in enumerate(platforms):
#     for j, condition in enumerate(cheater_conditions):
#         if condition == 'Cheaters':
#             subset_data = data_cheaters[data_cheaters['PlatformCategory'] == platform]
#         else:
#             subset_data = data_normal[data_normal['PlatformCategory'] == platform]

#         features_subset = subset_data.drop(columns=exclude_columns)
#         features_scaled_subset = StandardScaler().fit_transform(features_subset.fillna(0))

#         title = f'{platform} {condition}'
#         pca_and_plot(features_scaled_subset, column_names_cheaters, title, axs[i, j])

# plt.tight_layout()
# plt.savefig('PCA_evaluation.pdf')


def pca_and_plot(features_scaled_subset, feature_names, title, ax):
    pca = PCA().fit(features_scaled_subset)

    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

    circle = plt.Circle((0, 0), 1, facecolor='none', edgecolor=cornflowerblue_rgb, linestyle='--', lw=2)
    ax.add_artist(circle)
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)

    ax.axhline(0, color='grey', lw=1, linestyle='-.', alpha=0.7)
    ax.axvline(0, color='grey', lw=1, linestyle='-.', alpha=0.7)

    for j in range(loadings.shape[0]):
        ax.arrow(0, 0, loadings[j, 0], loadings[j, 1], color=indianred_rgb, 
                 head_width=0.03, head_length=0.04, alpha=0.85)
        x_adjust = 0.065 if loadings[j, 0] > 0 else -0.05
        y_adjust = 0.03 if loadings[j, 1] > 0 else 0.0
        ax.text(loadings[j, 0] + x_adjust, loadings[j, 1] + y_adjust, feature_names[j], 
                color='black', ha='center', va='center', fontsize=20)

    ax.spines['top'].set_linewidth(1.5)
    ax.spines['right'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)

    ax.set_xlabel("Principal Component 1", fontsize=34, weight='bold')
    ax.set_ylabel("Principal Component 2", fontsize=34, weight='bold')
    ax.set_title(title, fontsize=38, weight='bold')
    ax.grid(True, linestyle=':', alpha=0.5)

    ax.tick_params(axis='both', which='major', labelsize=25)

platforms = ['PC', 'Mobile']
cheater_conditions = ['Cheaters', 'Normal Players']

fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(24, 24))

for i, platform in enumerate(platforms):
    for j, condition in enumerate(cheater_conditions):
        if condition == 'Cheaters':
            subset_data = data_cheaters[data_cheaters['PlatformCategory'] == platform]
        else:
            subset_data = data_normal[data_normal['PlatformCategory'] == platform]

        features_subset = subset_data.drop(columns=exclude_columns)
        features_scaled_subset = StandardScaler().fit_transform(features_subset.fillna(0))

        title = f'{platform} {condition}'
        pca_and_plot(features_scaled_subset, column_names_cheaters, title, axs[i, j])

plt.tight_layout()
plt.savefig('PCA_evaluation.pdf')