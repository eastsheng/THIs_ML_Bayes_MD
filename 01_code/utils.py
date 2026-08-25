
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import fastdataing.fastdataing as fd
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Define the function to calculate evaluation metrics: R^2, RMSE, MAE, ARD, AARD
def calculate_metrics(y_true, y_pred):
	# R²
	r2 = r2_score(y_true, y_pred)
	# RMSE
	rmse = np.sqrt(mean_squared_error(y_true, y_pred))
	# MAE
	mae = mean_absolute_error(y_true, y_pred)
	# ARD (Average Relative Difference)
	ard = np.mean((y_true - y_pred) / y_true) * 100
	# AARD (Average Absolute Relative Difference)
	aard = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
	return r2, rmse, mae, ard, aard

def _read_data(f,select_columns,n=161,random_state=42):
	df = pd.read_excel(f)
	df = df[select_columns]
	range_dff = pd.DataFrame({
	    'Min': df.apply(lambda col: col[col != 0].min()),
		'Max': df.apply(lambda col: col[col != 0].max()),
		'Mean': df.apply(lambda col: col[col != 0].mean()),
	})
	print("数据集特征范围：")
	print(range_dff)
	# Randomly select 161 records as the additional validation set
	random_samples = df.sample(n=n,random_state=random_state)
	dff = df.drop(random_samples.index)
	X_other = random_samples[select_columns[:-1]]
	y_other = random_samples[select_columns[-1]]

	X = dff[select_columns[:-1]]
	y = dff[select_columns[-1]]

	return X,y,X_other,y_other

def __read_data(f,select_columns,n=161,random_state=42):
	df = pd.read_excel(f)
	df = df[select_columns]
	range_dff = pd.DataFrame({
	    'Min': df.apply(lambda col: col[col != 0].min()),
		'Max': df.apply(lambda col: col[col != 0].max()),
		'Mean': df.apply(lambda col: col[col != 0].mean()),
	})
	print("数据集特征范围：")
	print(range_dff)
	# Select salts:
	df_salts = df.loc[[i for i in range(23,84)]]
	dff = df.drop(df_salts.index)
	X_salts = df_salts[select_columns[:-1]]
	y_salts = df_salts[select_columns[-1]]
	# Select alcohols:
	df_alcohols = dff.loc[[i for i in range(1122,1154)]]
	dff = dff.drop(df_alcohols.index)
	X_alcohols = df_alcohols[select_columns[:-1]]
	y_alcohols = df_alcohols[select_columns[-1]]
	# Select ionic liquids:
	df_ionliquids = dff.loc[[i for i in range(1382,1408)]]
	dff = dff.drop(df_ionliquids.index)
	X_ionliquids = df_ionliquids[select_columns[:-1]]
	y_ionliquids = df_ionliquids[select_columns[-1]]
	# Select amino acids:
	df_aa = dff.loc[[i for i in range(1551,1583)]]
	dff = dff.drop(df_aa.index)
	X_aa = df_aa[select_columns[:-1]]
	y_aa = df_aa[select_columns[-1]]

	select_samples = pd.concat([df_salts, df_alcohols, df_ionliquids, df_aa], ignore_index=True)
	X_other = select_samples[select_columns[:-1]]
	y_other = select_samples[select_columns[-1]]
	print(X_other)
	X = dff[select_columns[:-1]]
	y = dff[select_columns[-1]]

	return X,y,X_other,y_other


def read_data(f,select_columns,n=161,random_state=42):
	df = pd.read_excel(f)
	df = df[select_columns]
	range_dff = pd.DataFrame({
	    'Min': df.apply(lambda col: col[col != 0].min()),
		'Max': df.apply(lambda col: col[col != 0].max()),
		'Mean': df.apply(lambda col: col[col != 0].mean()),
	})
	# print("Dataset feature ranges:")
	# print(range_dff)
	class_1 = ['NaI', 'NaCl', 'NaBr', 'Na2SO4', 'HCOONa', 'KCl', 'KBr', 'K2CO3', 'HCOOK',
	           'MgCl2', 'MgBr2', 'CaCl2', 'CaBr2', 'ZnBr2', 'NH4Cl']
	class_2 = ['methanol', 'ethylene glycol', 'diethylene glycol', 'triethylene glycol']
	class_3 = ['[EMIM]-[Cl]', '[EMIM]-[Br]', '[EMIM]-[EtSO4]', '[EMIM]-[HSO4]', '[OH-EMIM]-[BF4]',
	           '[BMIM]-[BF4]', '[TMA]-[OH]', '[BMIM]-[MeSO4]', '[TEA]-[Cl]', '[OH-C2MIM]-[Cl]', '[BMIM]-[I]']
	class_4 = ['asparagine', 'arginine', 'alanine', 'glycine', 'lysine', 'proline',
	           'phenylalanine', 'serine', 'threonine', 'valine']
	c1 = (df[class_1]>0).any(axis=1)
	c2 = (df[class_2]>0).any(axis=1)
	c3 = (df[class_3]>0).any(axis=1)
	c4 = (df[class_4]>0).any(axis=1)
	
	# Select salts:
	c1 = c1.reindex(df.index, fill_value=False)
	df_salts = df[c1].sample(n=101,random_state=random_state)
	dff = df.drop(df_salts.index)
	X_salts = df_salts[select_columns[:-1]]
	y_salts = df_salts[select_columns[-1]]
	# print(df_salts.shape)
	# Select alcohols:
	c2 = c2.reindex(dff.index, fill_value=False)
	df_alcohols = dff[c2].sample(n=28,random_state=random_state)
	dff = dff.drop(df_alcohols.index)
	X_alcohols = df_alcohols[select_columns[:-1]]
	y_alcohols = df_alcohols[select_columns[-1]]
	# print(df_alcohols.shape)
	# Select ionic liquids:	
	c3 = c3.reindex(dff.index, fill_value=False)
	df_ionliquids = dff[c3].sample(n=19,random_state=random_state)
	dff = dff.drop(df_ionliquids.index)
	X_ionliquids = df_ionliquids[select_columns[:-1]]
	y_ionliquids = df_ionliquids[select_columns[-1]]
	# print(df_ionliquids.shape)
	# Select amino acids:
	c4 = c4.reindex(dff.index, fill_value=False)
	df_aa = dff[c4].sample(n=13,random_state=random_state)
	dff = dff.drop(df_aa.index)
	X_aa = df_aa[select_columns[:-1]]
	y_aa = df_aa[select_columns[-1]]
	# print(df_aa.shape)
	select_samples = pd.concat([df_salts, df_alcohols, df_ionliquids, df_aa], ignore_index=True)
	X_other = select_samples[select_columns[:-1]]
	y_other = select_samples[select_columns[-1]]
	X = dff[select_columns[:-1]]
	y = dff[select_columns[-1]]
	# print(X.shape,X_other.shape)

	return X,y,X_other,y_other

if __name__ == '__main__':
	# f = './Data/newest_thermo_inhibitors_hydrate_phase_equilibrium_20241227.xlsx'
	f = '../Data/newest_thermo_inhibitors_hydrate_phase_equilibrium_20250313.xlsx'
	random_state=42
	label = "Thermodynamics"
	# Define the selected columns and group them into four inhibitor classes
	class_1 = ['NaI', 'NaCl', 'NaBr', 'Na2SO4', 'HCOONa', 'KCl', 'KBr', 'K2CO3', 'HCOOK',
	           'MgCl2', 'MgBr2', 'CaCl2', 'CaBr2', 'ZnBr2', 'NH4Cl']
	class_2 = ['methanol', 'ethylene glycol', 'diethylene glycol', 'triethylene glycol']
	class_3 = ['[EMIM]-[Cl]', '[EMIM]-[Br]', '[EMIM]-[EtSO4]', '[EMIM]-[HSO4]', '[OH-EMIM]-[BF4]',
	           '[BMIM]-[BF4]', '[TMA]-[OH]', '[BMIM]-[MeSO4]', '[TEA]-[Cl]', '[OH-C2MIM]-[Cl]', '[BMIM]-[I]']
	class_4 = ['asparagine', 'arginine', 'alanine', 'glycine', 'lysine', 'proline',
	           'phenylalanine', 'serine', 'threonine', 'valine']
	select_columns = class_1 + class_2 + class_3 + class_4

	df = pd.read_excel(f)
	df = df[select_columns]
	# print(df)
	# Count positive entries in each column
	greater_than_zero_counts = (df > 0).sum()
	# sns.set_theme(style="whitegrid")
	# sns.set(style="whitegrid")
	fig = fd.add_fig(figsize=(8, 6))
	ax = fd.add_ax(fig)
	plt.subplots_adjust(bottom=0.15,left=0.15)
	# Aggregate counts by inhibitor class
	class_1_counts = greater_than_zero_counts[class_1].sum()
	class_2_counts = greater_than_zero_counts[class_2].sum()
	class_3_counts = greater_than_zero_counts[class_3].sum()
	class_4_counts = greater_than_zero_counts[class_4].sum()
	# Set bar positions and widths
	x = ["Salts","Alcohols","Ionic Liquids", "Amino Acids"]
	# x = ["Salts","Alcohols","Ionic liquids", "Amino acids"]
	y = [class_1_counts,class_2_counts,class_3_counts,class_4_counts]
	ax = sns.barplot(x=x, y=y, palette="dark:#5A9_r",alpha=0.5)
	# Display values above the bars
	for p in ax.patches:
	    ax.text(p.get_x() + p.get_width() / 2, p.get_height(), int(p.get_height()),
	            ha='center', va='bottom', fontsize=20)
	# Display the figure
	ax.set_ylabel("Count")
	# ax.set_xticks(x,rotate=15)
	ax.set_xticklabels(x, rotation=15)  # Set tick labels and rotate them by 15 degrees

	ax.set_ylim(0,1500)
	# plt.tight_layout()
	# print(greater_than_zero_counts)
	for i in range(4):
		print(round(y[i]/2112*161,1),y[i])
	# plt.savefig("./imgs/count_data_new.png",dpi=300,transparent=True)
	plt.show()

	select_columns = [
	   'LogP','MW',
	   'NaI', 'NaCl', 'NaBr', 'Na2SO4','HCOONa',  'KCl','KBr','K2CO3', 'HCOOK', 
	   'MgCl2', 'MgBr2','CaCl2','CaBr2', 'ZnBr2',  'NH4Cl', 
	   'methanol','ethylene glycol','diethylene glycol',  'triethylene glycol', 
	   '[EMIM]-[Cl]','[EMIM]-[Br]', '[EMIM]-[EtSO4]', '[EMIM]-[HSO4]','[OH-EMIM]-[BF4]', 
	   '[BMIM]-[BF4]', '[TMA]-[OH]', '[BMIM]-[MeSO4]', '[TEA]-[Cl]','[OH-C2MIM]-[Cl]', '[BMIM]-[I]', 
	   'asparagine', 'arginine', 'alanine', 'glycine','lysine', 'proline', 
	   'phenylalanine', 'serine', 'threonine', 'valine', 'P (MPa)', 'T (K)']

	# ---------------------------------------------------------------------
	X,y,X_other,y_other = read_data(f,select_columns,n=161,random_state=random_state)