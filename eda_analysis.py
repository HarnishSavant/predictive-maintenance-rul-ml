"""
EDA - Exploratory Data Analysis
Dataset: NASA C-MAPSS FD001 (Predictive Maintenance)
Run: python eda_analysis.py
"""

import matplotlib
matplotlib.use('Agg')

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

print("=" * 50)
print("   Exploratory Data Analysis - NASA C-MAPSS")
print("=" * 50)

# --- Step 1: Load the dataset ---
columns = ['unit_id', 'cycle', 'op1', 'op2', 'op3']
for i in range(1, 22):
    columns.append('sensor_' + str(i))

train = pd.read_csv('Data/train_FD001.txt', sep=r'\s+', header=None, names=columns)
test = pd.read_csv('Data/test_FD001.txt', sep=r'\s+', header=None, names=columns)
rul = pd.read_csv('Data/RUL_FD001.txt', sep=r'\s+', header=None, names=['RUL'])

# --- Step 2: Basic info ---
print("\n--- Dataset Shape ---")
print(f"Training data : {train.shape[0]} rows, {train.shape[1]} columns")
print(f"Testing data  : {test.shape[0]} rows, {test.shape[1]} columns")
print(f"RUL labels    : {rul.shape[0]} engines")

print("\n--- First 5 Rows ---")
print(train.head())

print("\n--- Data Types ---")
print(train.dtypes.value_counts())

print("\n--- Missing Values ---")
print(f"Train missing: {train.isnull().sum().sum()}")
print(f"Test missing : {test.isnull().sum().sum()}")

print("\n--- Basic Statistics ---")
print(train.describe().round(2))

# --- Step 3: Add RUL column to training data ---
max_cycles = train.groupby('unit_id')['cycle'].max().reset_index()
max_cycles.columns = ['unit_id', 'max_cycle']
train = train.merge(max_cycles, on='unit_id')
train['RUL'] = train['max_cycle'] - train['cycle']
train.drop('max_cycle', axis=1, inplace=True)

# ============================================
# PLOT 1: Engine Lifetime (Bar Chart)
# ============================================
engine_life = train.groupby('unit_id')['cycle'].max()

plt.figure(figsize=(12, 4))
plt.bar(engine_life.index, engine_life.values, color='steelblue', edgecolor='black', linewidth=0.3)
plt.xlabel('Engine ID')
plt.ylabel('Total Cycles (Lifetime)')
plt.title('Plot 1: How Long Each Engine Ran Before Failure')
plt.tight_layout()
plt.savefig('eda_plot1_engine_lifetime.png', dpi=100)
plt.close()
print("\n[Saved] eda_plot1_engine_lifetime.png")

print(f"  Average lifetime : {engine_life.mean():.0f} cycles")
print(f"  Shortest engine  : Engine {engine_life.idxmin()} ({engine_life.min()} cycles)")
print(f"  Longest engine   : Engine {engine_life.idxmax()} ({engine_life.max()} cycles)")

# ============================================
# PLOT 2: RUL Distribution (Histogram)
# ============================================
plt.figure(figsize=(8, 4))
plt.hist(train['RUL'], bins=40, color='salmon', edgecolor='black')
plt.xlabel('Remaining Useful Life (RUL)')
plt.ylabel('Count')
plt.title('Plot 2: Distribution of RUL Values')
plt.axvline(train['RUL'].mean(), color='red', linestyle='--', label=f"Mean = {train['RUL'].mean():.0f}")
plt.legend()
plt.tight_layout()
plt.savefig('eda_plot2_rul_distribution.png', dpi=100)
plt.close()
print("[Saved] eda_plot2_rul_distribution.png")

# ============================================
# PLOT 3: Correlation Heatmap
# ============================================
sensor_cols = ['sensor_' + str(i) for i in range(1, 22)]
corr = train[sensor_cols + ['RUL']].corr()

plt.figure(figsize=(14, 10))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0, linewidths=0.5)
plt.title('Plot 3: Correlation Heatmap (Sensors + RUL)')
plt.tight_layout()
plt.savefig('eda_plot3_correlation_heatmap.png', dpi=100)
plt.close()
print("[Saved] eda_plot3_correlation_heatmap.png")

# Print top 5 sensors most correlated with RUL
rul_corr = corr['RUL'].drop('RUL').abs().sort_values(ascending=False)
print("\n  Top 5 sensors correlated with RUL:")
for i, (name, val) in enumerate(rul_corr.head(5).items()):
    print(f"    {i+1}. {name} = {val:.3f}")

# ============================================
# PLOT 4: Sensor Trends for 1 Engine
# ============================================
top_sensors = rul_corr.head(4).index.tolist()
engine1 = train[train['unit_id'] == 1]

fig, axes = plt.subplots(2, 2, figsize=(12, 7))
fig.suptitle('Plot 4: Sensor Trends Over Time (Engine 1)', fontsize=13)

for i, sensor in enumerate(top_sensors):
    ax = axes[i // 2][i % 2]
    ax.plot(engine1['cycle'], engine1[sensor], color='teal', linewidth=0.9)
    ax.set_title(sensor)
    ax.set_xlabel('Cycle')
    ax.set_ylabel('Value')

plt.tight_layout()
plt.savefig('eda_plot4_sensor_trends.png', dpi=100)
plt.close()
print("[Saved] eda_plot4_sensor_trends.png")

# ============================================
# PLOT 5: Boxplots of Important Sensors
# ============================================
top6 = rul_corr.head(6).index.tolist()

fig, axes = plt.subplots(2, 3, figsize=(14, 7))
fig.suptitle('Plot 5: Boxplots of Top Sensors', fontsize=13)

for i, sensor in enumerate(top6):
    ax = axes[i // 3][i % 3]
    ax.boxplot(train[sensor], patch_artist=True,
               boxprops=dict(facecolor='lightblue'))
    ax.set_title(sensor)

plt.tight_layout()
plt.savefig('eda_plot5_boxplots.png', dpi=100)
plt.close()
print("[Saved] eda_plot5_boxplots.png")

# ============================================
# Summary
# ============================================
print("\n" + "=" * 50)
print("   EDA COMPLETE")
print("=" * 50)
print(f"  Total Engines  : {train['unit_id'].nunique()}")
print(f"  Total Features : 3 settings + 21 sensors")
print(f"  Missing Values : None")
print(f"  Best Sensors   : {top_sensors}")
print(f"  Plots Saved    : 5")
print("=" * 50)
