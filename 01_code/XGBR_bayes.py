import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import fastdataing.fastdataing as fd
from tqdm import tqdm
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, RandomizedSearchCV
from xgboost import XGBRegressor

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from joblib import dump, load
from datetime import datetime
from pathlib import Path
import warnings
import sys
import logging
import optuna
from readlog import print_log
from utils import read_data, calculate_metrics


def objective(trial):

	# Define the hyperparameters to optimize
	params = {
	'n_estimators': trial.suggest_int('n_estimators', 10, 1000, step=1), # Number of trees
	'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, step=0.01),
	'max_depth': trial.suggest_int('max_depth', 3, 50, step=1), # Maximum tree depth
	'min_child_weight': trial.suggest_int('min_child_weight', 1, 20, step=1),
	'subsample': trial.suggest_float('subsample', 0.5, 1.0, step=0.05),
	'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0, step=0.05),
	'gamma': trial.suggest_float('gamma', 0, 0.5, step=0.05),
	'reg_lambda': trial.suggest_float('reg_lambda', 0, 10, step=0.1),
	'reg_alpha': trial.suggest_float('reg_alpha', 0, 1.0, step=0.1),
	# 'early_stopping_rounds': trial.suggest_int('early_stopping_rounds', 10, 50, step=10)
	'random_state': random_state,
	'n_jobs': n_jobs
	}
	# Use five-fold cross-validation
	kf = KFold(n_splits=cv, shuffle=True, random_state=random_state)
	r2_list, rmse_list, mae_list, ard_list, aard_list = [], [], [], [], []

	# Run five-fold cross-validation
	for train_index, test_index in kf.split(X_train):
		X_tr, X_te = X_train[train_index], X_train[test_index]
		y_tr, y_te = y_train[train_index], y_train[test_index]

		# Create the regression model
		model = XGBRegressor(**params)
		# Fit the model on the training fold
		model.fit(X_tr, y_tr,eval_set=[(X_te, y_te)], verbose=False)

		# Predict the validation fold
		y_pr = model.predict(X_te)
		
		# Calculate evaluation metrics
		r2, rmse, mae, ard, aard = calculate_metrics(y_te, y_pr)
		r2_list.append(r2)
		rmse_list.append(rmse)
		mae_list.append(mae)
		ard_list.append(ard)
		aard_list.append(aard)

	# Return RMSE as the objective value to be minimized
	return np.mean(rmse_list)

def extract_metrics_from_trials(study,X,y,random_state=42):
	# Extract the metrics for each trial
	for trial in study.trials:
		param = trial.params
		# print(trial)
		r2_list, rmse_list, mae_list, ard_list, aard_list = [], [], [], [], []
		
		for train_index, test_index in KFold(n_splits=cv, shuffle=True, random_state=random_state).split(X):
			X_train, X_test = X[train_index], X[test_index]
			y_train, y_test = y[train_index], y[test_index]
			
			model = XGBRegressor(**param)
			model.fit(X_train, y_train)
			y_pred = model.predict(X_test)
			
			r2, rmse, mae, ard, aard = calculate_metrics(y_test, y_pred)
			
			r2_list.append(r2)
			rmse_list.append(rmse)
			mae_list.append(mae)
			ard_list.append(ard)
			aard_list.append(aard)
		
		metrics['R^2'].append(np.mean(r2_list))
		metrics['RMSE'].append(np.mean(rmse_list))
		metrics['MAE'].append(np.mean(mae_list))
		metrics['ARD'].append(np.mean(ard_list))
		metrics['AARD'].append(np.mean(aard_list))

	# Convert metrics to a DataFrame for boxplot visualization
	df_metrics = pd.DataFrame(metrics)
	return df_metrics

if __name__ == '__main__':
	# ----------------------------- Variables -----------------------------
	label = "Thermodynamics" # purehydrate porehydrate
	model = "XGBR"
	search = "bayes_new_V5"
	test_size=0.2
	random_state=42
	cv = 5  # Number of cross-validation folds
	path="CV5_new_four_random1"
	n_jobs = -1
	n_trials = 500
	train = True
	start_time = datetime.now()
	folder_path = Path("save_model")
	folder_path.mkdir(exist_ok=True)
	folder_path = Path("imgs")
	folder_path.mkdir(exist_ok=True)
	folder_path = Path(f"./save_model/{path}")
	folder_path.mkdir(exist_ok=True)

	print_log(f'./save_model/{path}/{label}_best_{model}_model_{search}.log')
	f = '../Data/newest_thermo_inhibitors_hydrate_phase_equilibrium_20250313.xlsx'
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

	X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=test_size, random_state=random_state)
	print(f">>> 训练集 X_train: {X_train.shape}")
	print(f">>> 测试集 X_test: {X_test.shape}")
	print(f">>> 额外验证集 X_test: {X_other.shape}")

	y_train = y_train.to_numpy()
	y_test = y_test.to_numpy()
	y_other = y_other.to_numpy()

	scaler = StandardScaler() # StandardScaler MinMaxScaler
	X_train = scaler.fit_transform(X_train)
	X_test = scaler.transform(X_test)
	X_other = scaler.transform(X_other)

	if train:

		# Create an Optuna study that minimizes RMSE
		study = optuna.create_study(direction='minimize')  # Minimize RMSE
		study.optimize(objective, n_trials=n_trials)  # Run the optimization trials
		# Report the best hyperparameters and RMSE
		# print(f"Best trial: {study.best_trial}")
		print(f"最佳超参数: {study.best_trial.params}")
		print(f"最佳 RMSE: {study.best_value}")
		metrics = {
			'R^2': [],
			'RMSE': [],
			'MAE': [],
			'ARD': [],
			'AARD': []
		}
		# Extract the metrics for each trial
		df_metrics = extract_metrics_from_trials(study=study,X=X_train,y=y_train,random_state=random_state)
		df_metrics.to_excel(f'./save_model/{path}/metrics_results_{label}_best_{model}_model_{search}.xlsx', index=False)

		# Recreate the model using the best hyperparameters
		best_params = study.best_trial.params
		best_model = XGBRegressor(**best_params)
		# Refit the best model on the complete training set
		best_model.fit(X_train, y_train)

		dump(best_model, f'./save_model/{path}/{label}_best_{model}_model_{search}.pkl')
	else:
		best_model = load(f'./save_model/{path}/{label}_best_{model}_model_{search}.pkl')

	y_train_pred = best_model.predict(X_train)
	y_test_pred = best_model.predict(X_test)
	y_other_pred = best_model.predict(X_other)

	resi_train = y_train_pred - y_train
	r2_train, rmse_train, mae_train, ARD_train, AARD_train =  calculate_metrics(y_train,y_train_pred)
	print(f">>> R^2 (train) = {r2_train:.4f}")
	print(f">>> rmse (train) = {rmse_train:.4f}")
	print(f">>> MAE (train) = {mae_train:.4f}")
	print(f">>> ARD (train) = {ARD_train:.4f}")
	print(f">>> AARD (train) = {AARD_train:.4f}\n")

	resi_test = y_test_pred - y_test
	r2_test, rmse_test, mae_test, ARD_test, AARD_test =  calculate_metrics(y_test,y_test_pred)
	
	print(f">>> R^2 (test) = {r2_test:.4f}")
	print(f">>> rmse (test) = {rmse_test:.4f}")
	print(f">>> MAE (test) = {mae_test:.4f}")
	print(f">>> ARD (test) = {ARD_test:.4f}")
	print(f">>> AARD (test) = {AARD_test:.4f}\n")

	resi_val = y_other_pred - y_other
	r2_val, rmse_val, mae_val, ARD_val, AARD_val =  calculate_metrics(y_other,y_other_pred)
	
	print(f">>> R^2 (val) = {r2_val:.4f}")
	print(f">>> rmse (val) = {rmse_val:.4f}")
	print(f">>> MAE (val) = {mae_val:.4f}")
	print(f">>> ARD (val) = {ARD_val:.4f}")
	print(f">>> AARD (val) = {AARD_val:.4f}\n")

	# ------------------------------------------------------------------------------
	fig = fd.add_fig(figsize=(17,6))
	plt.subplots_adjust(bottom=0.15)
	ax = fd.add_ax(fig,subplot=(121))
	ay = fd.add_ax(fig,subplot=(122))
	ax.scatter(y_train,y_train_pred,edgecolors=(0, 0, 0),label="Train")
	ax.scatter(y_test,y_test_pred,edgecolors=(0, 0, 0),label="Test")
	# ax.scatter(y_val,y_val_pred,edgecolors=(0, 0, 0),label=r"$\regular Test_{\it new}$")
	ylims = [240,320]
	ax.plot(ylims,ylims,'r--',lw=2)
	ax.set_xlim(ylims[0],ylims[1])
	ax.set_ylim(ylims[0],ylims[1])
	ax.set_xticks([250,260,270,280,290,300,310,320])
	ax.set_yticks([250,260,270,280,290,300,310,320])

	ax.text(243,295,r"$\regular R^2$ (Train) = "+str(round(r2_train,4)))
	ax.text(243,288,r"$\regular R^2$ (Test) = "+str(round(r2_test,4)))
	# ax.text(237,268,r"$\regular R^2$ (Test$\regular_{\it new}$) = "+str(round(r2_test,4)))
	ax.text(275,257,r"$\regular RMSE$ (Train) = "+str(round(rmse_train,4)))
	ax.text(275,250,r"$\regular RMSE$ (Test) = "+str(round(rmse_test,4)))
	# ax.text(270,240,r"$\regular RMSE$ (Test$\regular_{\it new}$) = "+str(round(rmse_test,4)))

	ax.set_xlabel(r"$\regular \it T_{exp}$ (K)",fontsize=22)
	ax.set_ylabel(r"$\regular \it T_{pred}$ (K)",fontsize=22)
	ay.scatter(y_train,resi_train,edgecolors=(0, 0, 0),label="Train")
	ay.scatter(y_test,resi_test,edgecolors=(0, 0, 0),label="Test")
	# ay.scatter(y_val,resi_val,edgecolors=(0, 0, 0),label=r"$\regular Test_{\it new}$")
	ay.axhline(y=0,color="r",linestyle='--',lw=2)
	ay.set_xlabel(r"$\regular \it T_{exp}$ (K)",fontsize=22)
	ay.set_ylabel(r"$\regular \it Resi$ (K)",fontsize=22)
	ay.set_xlim(ylims[0],ylims[1])
	ay.set_xticks([250,260,270,280,290,300,310,320])
	ay.set_ylim(-20,20)
	
	fd.set_fig(ax,loc="upper left",ncols=1)
	fd.set_fig(ay,ncols=1)
	plt.savefig(f"./imgs/{model}_{label}_{search}.png",dpi=300,transparent=True)
	# plt.show()

	end_time = datetime.now()
	duration = end_time - start_time
	duration_minutes = duration.total_seconds() / 60
	print(f"程序运行时间: {duration_minutes:.2f} 分钟")