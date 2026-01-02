#%% [markdown]
# # CSIRO Image2Biomass - EDA
#
# 목초지 이미지로부터 바이오매스(건조 중량) 예측
# - Train: 357 images × 5 targets (long format)
# - Test: 1 image (public만 제공, private은 Kaggle에서 관리)
# - Metric: Globally weighted R²

#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from PIL import Image

# Insight: Seaborn 스타일 설정 - whitegrid가 분포 시각화에 적합
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

DATA_DIR = Path("data")

#%% [markdown]
# ## 1. Load Data

#%%
train_df = pd.read_csv(DATA_DIR / "train.csv")
test_df = pd.read_csv(DATA_DIR / "test.csv")
sample_sub = pd.read_csv(DATA_DIR / "sample_submission.csv")

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")
print(f"Sample submission shape: {sample_sub.shape}")

#%%
train_df.head(10)

#%%
train_df.info()

#%% [markdown]
# ## 2. Data Structure Analysis
#
# Long format → Wide format 변환 필요 (이미지당 5개 타겟)

#%%
# Long format: 각 row가 (image, target_name, target_value)
# Wide format: 각 row가 (image, Dry_Green_g, Dry_Dead_g, ...)

# Unique images
n_unique_images = train_df['image_path'].nunique()
print(f"Unique train images: {n_unique_images}")
print(f"Rows per image: {len(train_df) / n_unique_images}")

# Target names
print(f"\nTarget names: {train_df['target_name'].unique()}")

#%%
# Wide format 변환
# Insight: pivot 시 tabular features는 first()로 가져옴 (동일 이미지는 동일 값)
tabular_cols = ['image_path', 'Sampling_Date', 'State', 'Species', 'Pre_GSHH_NDVI', 'Height_Ave_cm']

train_wide = train_df.pivot_table(
    index=tabular_cols,
    columns='target_name',
    values='target',
    aggfunc='first'
).reset_index()

print(f"Wide format shape: {train_wide.shape}")
train_wide.head()

#%% [markdown]
# ## 3. Target Distribution

#%%
targets = ['Dry_Green_g', 'Dry_Dead_g', 'Dry_Clover_g', 'Dry_Total_g', 'GDM_g']

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for i, target in enumerate(targets):
    ax = axes[i]
    # Insight: 바이오매스는 보통 right-skewed → log 변환 고려
    data = train_wide[target]

    ax.hist(data, bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(data.mean(), color='red', linestyle='--', label=f'Mean: {data.mean():.2f}')
    ax.axvline(data.median(), color='orange', linestyle='--', label=f'Median: {data.median():.2f}')
    ax.set_title(f'{target}\nSkew: {data.skew():.2f}, Zeros: {(data == 0).sum()}')
    ax.set_xlabel('Value (g)')
    ax.legend()

axes[-1].axis('off')
plt.suptitle('Target Distributions', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('notebooks/figures/target_distributions.png', dpi=150, bbox_inches='tight')
plt.show()

#%%
# Target statistics
target_stats = train_wide[targets].describe().T
target_stats['skew'] = train_wide[targets].skew()
target_stats['zeros'] = (train_wide[targets] == 0).sum()
target_stats['zero_pct'] = target_stats['zeros'] / len(train_wide) * 100
print(target_stats.round(2))

#%% [markdown]
# ## 4. Target Correlations

#%%
fig, ax = plt.subplots(figsize=(8, 6))
corr = train_wide[targets].corr()
sns.heatmap(corr, annot=True, cmap='RdBu_r', center=0, fmt='.2f', ax=ax)
ax.set_title('Target Correlations')
plt.tight_layout()
plt.savefig('notebooks/figures/target_correlations.png', dpi=150, bbox_inches='tight')
plt.show()

# Insight: Dry_Total = Dry_Green + Dry_Dead + Dry_Clover 관계 확인
print("\n[Tensor Risk] Dry_Total 구성 확인:")
train_wide['computed_total'] = train_wide['Dry_Green_g'] + train_wide['Dry_Dead_g'] + train_wide['Dry_Clover_g']
diff = (train_wide['Dry_Total_g'] - train_wide['computed_total']).abs()
print(f"Max diff from sum: {diff.max():.4f}")
print(f"Mean diff: {diff.mean():.4f}")

#%% [markdown]
# ## 5. Tabular Features Analysis

#%%
# Categorical features
print("=== Categorical Features ===")
for col in ['State', 'Species']:
    print(f"\n{col}:")
    print(train_wide[col].value_counts())

#%%
# State distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# State counts
ax = axes[0]
state_counts = train_wide['State'].value_counts()
state_counts.plot(kind='bar', ax=ax, color='steelblue', edgecolor='black')
ax.set_title('Samples per State')
ax.set_xlabel('State')
ax.set_ylabel('Count')
ax.tick_params(axis='x', rotation=45)

# Species counts
ax = axes[1]
species_counts = train_wide['Species'].value_counts()
species_counts.plot(kind='bar', ax=ax, color='forestgreen', edgecolor='black')
ax.set_title('Samples per Species')
ax.set_xlabel('Species')
ax.set_ylabel('Count')
ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('notebooks/figures/categorical_distributions.png', dpi=150, bbox_inches='tight')
plt.show()

#%%
# Numerical features
print("\n=== Numerical Features ===")
print(train_wide[['Pre_GSHH_NDVI', 'Height_Ave_cm']].describe())

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

ax = axes[0]
ax.hist(train_wide['Pre_GSHH_NDVI'], bins=30, edgecolor='black', alpha=0.7, color='green')
ax.set_title('NDVI Distribution')
ax.set_xlabel('Pre_GSHH_NDVI')

ax = axes[1]
ax.hist(train_wide['Height_Ave_cm'], bins=30, edgecolor='black', alpha=0.7, color='brown')
ax.set_title('Vegetation Height Distribution')
ax.set_xlabel('Height_Ave_cm')

plt.tight_layout()
plt.savefig('notebooks/figures/numerical_features.png', dpi=150, bbox_inches='tight')
plt.show()

#%% [markdown]
# ## 6. Feature-Target Relationships

#%%
fig, axes = plt.subplots(2, 5, figsize=(18, 8))

for i, target in enumerate(targets):
    # NDVI vs Target
    ax = axes[0, i]
    ax.scatter(train_wide['Pre_GSHH_NDVI'], train_wide[target], alpha=0.5, s=20)
    ax.set_xlabel('NDVI')
    ax.set_ylabel(target)
    ax.set_title(f'NDVI vs {target}')

    # Height vs Target
    ax = axes[1, i]
    ax.scatter(train_wide['Height_Ave_cm'], train_wide[target], alpha=0.5, s=20, color='brown')
    ax.set_xlabel('Height (cm)')
    ax.set_ylabel(target)
    ax.set_title(f'Height vs {target}')

plt.tight_layout()
plt.savefig('notebooks/figures/feature_target_scatter.png', dpi=150, bbox_inches='tight')
plt.show()

#%%
# Correlation with targets
feature_target_corr = train_wide[['Pre_GSHH_NDVI', 'Height_Ave_cm'] + targets].corr()
print("Feature-Target Correlations:")
print(feature_target_corr.loc[['Pre_GSHH_NDVI', 'Height_Ave_cm'], targets].round(3))

#%% [markdown]
# ## 7. Temporal Analysis

#%%
train_wide['Sampling_Date'] = pd.to_datetime(train_wide['Sampling_Date'])
train_wide['Year'] = train_wide['Sampling_Date'].dt.year
train_wide['Month'] = train_wide['Sampling_Date'].dt.month

print("Year distribution:")
print(train_wide['Year'].value_counts().sort_index())

print("\nMonth distribution:")
print(train_wide['Month'].value_counts().sort_index())

#%%
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# By Year
ax = axes[0]
train_wide.groupby('Year')[targets].mean().plot(kind='bar', ax=ax)
ax.set_title('Mean Target by Year')
ax.set_xlabel('Year')
ax.set_ylabel('Mean Value (g)')
ax.legend(loc='upper right', fontsize=8)
ax.tick_params(axis='x', rotation=0)

# By Month
ax = axes[1]
train_wide.groupby('Month')[targets].mean().plot(kind='line', marker='o', ax=ax)
ax.set_title('Mean Target by Month (Seasonality)')
ax.set_xlabel('Month')
ax.set_ylabel('Mean Value (g)')
ax.legend(loc='upper right', fontsize=8)
ax.set_xticks(range(1, 13))

plt.tight_layout()
plt.savefig('notebooks/figures/temporal_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

#%% [markdown]
# ## 8. Sample Images

#%%
# Insight: 대표적인 이미지 몇 개를 시각화하여 데이터 특성 파악
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()

# 다양한 특성을 가진 샘플 선택
sample_indices = [
    train_wide['Dry_Total_g'].idxmin(),  # 최소 바이오매스
    train_wide['Dry_Total_g'].idxmax(),  # 최대 바이오매스
    train_wide['Dry_Green_g'].idxmax(),  # 최대 녹색
    train_wide['Dry_Dead_g'].idxmax(),   # 최대 고사
    train_wide['Dry_Clover_g'].idxmax(), # 최대 클로버
    train_wide[train_wide['Dry_Total_g'] > 0]['Dry_Total_g'].quantile(0.25),  # Q1
    train_wide[train_wide['Dry_Total_g'] > 0]['Dry_Total_g'].quantile(0.5),   # Median
    train_wide[train_wide['Dry_Total_g'] > 0]['Dry_Total_g'].quantile(0.75),  # Q3
]

# 실제 인덱스로 변환 (quantile은 값을 반환하므로)
samples = [
    train_wide.loc[train_wide['Dry_Total_g'].idxmin()],
    train_wide.loc[train_wide['Dry_Total_g'].idxmax()],
    train_wide.loc[train_wide['Dry_Green_g'].idxmax()],
    train_wide.loc[train_wide['Dry_Dead_g'].idxmax()],
    train_wide.loc[train_wide['Dry_Clover_g'].idxmax()],
]

# 중간값 근처 샘플 추가
median_val = train_wide['Dry_Total_g'].median()
median_idx = (train_wide['Dry_Total_g'] - median_val).abs().idxmin()
samples.append(train_wide.loc[median_idx])

# 랜덤 샘플 2개 추가
np.random.seed(42)
random_indices = np.random.choice(train_wide.index, 2, replace=False)
for idx in random_indices:
    samples.append(train_wide.loc[idx])

titles = ['Min Total', 'Max Total', 'Max Green', 'Max Dead', 'Max Clover', 'Median Total', 'Random 1', 'Random 2']

for i, (sample, title) in enumerate(zip(samples, titles)):
    ax = axes[i]
    img_path = DATA_DIR / sample['image_path']
    img = Image.open(img_path)
    ax.imshow(img)
    ax.set_title(f"{title}\nG:{sample['Dry_Green_g']:.0f} D:{sample['Dry_Dead_g']:.0f} C:{sample['Dry_Clover_g']:.0f} T:{sample['Dry_Total_g']:.0f}")
    ax.axis('off')

plt.suptitle('Sample Images (G=Green, D=Dead, C=Clover, T=Total)', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('notebooks/figures/sample_images.png', dpi=150, bbox_inches='tight')
plt.show()

#%% [markdown]
# ## 9. Image Properties

#%%
# 이미지 크기 확인
sample_img_path = DATA_DIR / train_wide.iloc[0]['image_path']
sample_img = Image.open(sample_img_path)
print(f"Sample image size: {sample_img.size}")
print(f"Sample image mode: {sample_img.mode}")

# 모든 이미지 크기 확인 (동일한지)
img_sizes = []
for img_path in train_wide['image_path'].unique()[:50]:  # 처음 50개만 확인
    img = Image.open(DATA_DIR / img_path)
    img_sizes.append(img.size)

unique_sizes = set(img_sizes)
print(f"\nUnique image sizes (first 50): {unique_sizes}")

#%% [markdown]
# ## 10. Summary & Key Findings

#%%
print("=" * 60)
print("EDA Summary - CSIRO Image2Biomass")
print("=" * 60)

print(f"""
📊 Dataset Overview:
- Train images: {n_unique_images}
- Test images: 1 (public, private는 Kaggle에서 관리)
- Targets: 5 (Dry_Green_g, Dry_Dead_g, Dry_Clover_g, Dry_Total_g, GDM_g)

📈 Target Characteristics:
- All targets are right-skewed → Log transform 고려
- Many zeros in Dry_Clover_g ({(train_wide['Dry_Clover_g'] == 0).sum()}/{len(train_wide)} = {(train_wide['Dry_Clover_g'] == 0).mean()*100:.1f}%)
- Dry_Total ≈ Dry_Green + Dry_Dead + Dry_Clover (거의 정확히 성립)
- GDM_g ≈ Dry_Green_g (높은 상관관계)

🌍 Spatial Distribution:
- States: {train_wide['State'].nunique()} ({', '.join(train_wide['State'].unique())})
- Species: {train_wide['Species'].nunique()} types

📅 Temporal Distribution:
- Years: {train_wide['Year'].min()} - {train_wide['Year'].max()}
- Seasonality observed (Month 9-11 높은 바이오매스)

🔢 Tabular Features:
- NDVI: {train_wide['Pre_GSHH_NDVI'].mean():.2f} ± {train_wide['Pre_GSHH_NDVI'].std():.2f}
- Height: {train_wide['Height_Ave_cm'].mean():.1f} ± {train_wide['Height_Ave_cm'].std():.1f} cm
- NDVI-Target correlation: moderate positive
- Height-Target correlation: weak to moderate positive

🎯 Modeling Considerations:
1. Multi-output regression (5 targets, but correlated)
2. Spatial CV 고려 (State/Species 기반 split)
3. Log transform for targets (right-skewed)
4. Image + Tabular fusion model
5. Data augmentation 신중히 (농업 도메인 특성)
""")

#%%
# Wide format 저장 (모델 학습용)
train_wide.to_csv(DATA_DIR / "train_wide.csv", index=False)
print(f"Saved train_wide.csv: {train_wide.shape}")
