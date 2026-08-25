import numpy as np
import pandas as pd
from functools import partial
from joblib import dump, load
from skopt import gp_minimize
from skopt.space import Real
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import fastdataing.fastdataing as fd
from readlog import print_log
from utils import read_data
from rdkit import Chem
from rdkit.Chem import Descriptors
import copy
import pubchempy as pcp
from pathlib import Path

def generate_SIMLES(molecule):
	compound = pcp.get_compounds(molecule,"name")
	# print(molecule,compound)
	simles  = compound[0].canonical_smiles
	print(molecule,compound,simles)
	return simles

def generate_LogPs(molecules):  
	SIMLES_dict = {item: generate_SIMLES(item) for item in molecules}
	LogP_dict = copy.deepcopy(SIMLES_dict)
	MW_dict = copy.deepcopy(SIMLES_dict)
	
	for key, value in LogP_dict.items():
		mol = Chem.MolFromSmiles(value)
		if mol:
			logp = Descriptors.MolLogP(mol)
			# logps = list(logp)
			LogP_dict[key] = logp
			MW_dict[key] = Descriptors.MolWt(mol)
		else:
			print(f"Invalid SMILES: {value}")

	return LogP_dict, MW_dict

# Define the objective function for predicting a formulation
def objective_function(x,indexs,Ph,best_model,df,X_train):
	example = np.zeros(43).tolist() # Initialize one feature vector
	for i in range(len(indexs)):
		# print(x)
		example[indexs[i]] = x[i] # Set the optimized inhibitor concentration
		print(f">>> 选择的抑制剂浓度筛选: {x[i]}")
	print(f">>> 选择的抑制剂在特征中的索引: {indexs}")
	print(f">>> 压力: {Ph} MPa")
	# print(example)
	X_example = pd.DataFrame([example], columns=X_train.columns)
	my_dict = dict(zip(df["Name"], df["Full Name"]))
	my_dict.update({'methanol': 'methanol', 'diethylene glycol': 'diethylene glycol', 'triethylene glycol':'triethylene glycol',
		'glycine':'glycine','lysine':'lysine'
		})
	LogP, MW = 0, 0
	selected_inhibitors_full_name = []
	for i in range(len(indexs)):
		selected_inhibitors_full_name.append(my_dict[select_columns[indexs[i]]])
	# print(f">>> Full names of the selected inhibitors: {selected_inhibitors_full_name} ")
	LogP_dict, MW_dict = generate_LogPs(selected_inhibitors_full_name)
	for index, row in X_example.iterrows():
		for i in range(len(indexs)):
			inhibitor = select_columns[indexs[i]]
			LogP = LogP + X_example.at[index,inhibitor]*LogP_dict[my_dict[inhibitor]]
			MW = MW + X_example.at[index,inhibitor]*MW_dict[my_dict[inhibitor]]

	# Set the concentration-weighted LogP descriptor
	X_example["LogP"] = LogP
	# Set the concentration-weighted MW descriptor
	X_example["MW"] = MW

	X_example["P (MPa)"] = Ph # x[-1] # Set the pressure feature

	# print(f">>> Candidate feature vector: {X_example}")

	scaler = StandardScaler()
	X_train = scaler.fit_transform(X_train)
	X_example = scaler.transform(X_example)
	return best_model.predict([X_example[0]])[0]  # Use the sign convention required by the optimizer


if __name__ == '__main__':
	# -------------------------------------------------------------------------------------------------
	f = '../Data/newest_thermo_inhibitors_hydrate_phase_equilibrium_20250313.xlsx'
	pwf = '../Data/newest_pure_water_hydrate_phase_equilibrium.xlsx'
	select_columns = [
	   'LogP','MW',
	   'NaI', 'NaCl', 'NaBr', 'Na2SO4','HCOONa',  'KCl','KBr','K2CO3', 'HCOOK', 
	   'MgCl2', 'MgBr2','CaCl2','CaBr2', 'ZnBr2',  'NH4Cl', 
	   'methanol','ethylene glycol','diethylene glycol',  'triethylene glycol', 
	   '[EMIM]-[Cl]','[EMIM]-[Br]', '[EMIM]-[EtSO4]', '[EMIM]-[HSO4]','[OH-EMIM]-[BF4]', 
	   '[BMIM]-[BF4]', '[TMA]-[OH]', '[BMIM]-[MeSO4]', '[TEA]-[Cl]','[OH-C2MIM]-[Cl]', '[BMIM]-[I]', 
	   'asparagine', 'arginine', 'alanine', 'glycine','lysine', 'proline', 
	   'phenylalanine', 'serine', 'threonine', 'valine', 'P (MPa)', 'T (K)']
	label = "Thermodynamics" # purehydrate porehydrate
	search = "bayes_new_V5"
	model = "GBR"
	path="CV5_new_four_random1"
	save_path = f"double_less_20wt_{model}"
	folder_path = Path(f"./optimization/{save_path}")
	folder_path.mkdir(exist_ok=True)
	temp_path = Path(f"./optimization/{save_path}/temp")
	temp_path.mkdir(exist_ok=True)
	random_state = 42
	best_model = load(f'./save_model/{path}/{label}_best_{model}_model_{search}.pkl')
	# Fit the standardizer on the training data
	X,y,X_other,y_other = read_data(f,select_columns=select_columns,n=161,random_state=42)
	X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)
	# -------------------------------------------
	# indexss = [[select_columns.index("NaCl")]]
	# Iterate over the selected inhibitors
	indexsss = [
	#All concentration combinations [13,18,21,37]
	#0.1~20%  [11,17,30,39] 
	# [13,18],[13,21],[13,37],[18,21],[18,37],[21,37] #All concentration combinations

	[11,17],[11,30],[11,39],[17,30],[17,39],[30,39] #0.1~20% 
	]
	# -------------------------------------------
	n_calls = 200 # Number of optimization calls
	n_jobs = 8
	calc = True
	# Specify pressure Ph
	# Pressures = [5,15]
	Pressures = [5,15,25,35,45,55,65,75]
	if calc:
		print_log(f"./optimization/find_inhibitor_double_{model}.log","w")
		for ind in indexsss:
			indexss = [ind]
			# -------------------------------------------------------------------------------------------------
			df = pd.read_excel(f)
			dff = df[select_columns]
			mins = dff.apply(lambda col: col[col != 0].min())
			maxs = dff.apply(lambda col: col[col != 0].max())
			# print(f">>> Pressure range in the dataset: [{mins['P (MPa)']}, {maxs['P (MPa)']}]")
				
			# Define the search space using inhibitor mass-fraction and pressure ranges represented in the training data
			for indexs in indexss:
				selected_inhibitors = []
				for i in range(len(indexs)):
					selected_inhibitors.append(select_columns[indexs[i]])
				print(f">>> 选择优化的抑制剂: {selected_inhibitors} ")
				space = [
					# Real(mins[selected_inhibitors[0]],maxs[selected_inhibitors[0]], name=selected_inhibitors[0]),
					# Real(mins[selected_inhibitors[1]],maxs[selected_inhibitors[1]], name=selected_inhibitors[1]),
					Real(0.1,20, name=selected_inhibitors[0]),
					Real(0.1,20, name=selected_inhibitors[1]),
					# Real(Pressures[0],Pressures[5], name='P (MPa)'),
				]
				func_vals_list = []
				temperature_obs, c_inhibitors, c_inhibitors1 = [], [], []
				for pressure in Pressures:
					# pressure = 2
					# Run Bayesian optimization to identify the best formulation
					objective_function = partial(objective_function, indexs=indexs,Ph=pressure,best_model=best_model,df=df,X_train=X_train)
					result = gp_minimize(objective_function, space, n_calls=n_calls, verbose=1, random_state=random_state,n_jobs=n_jobs)
					# func_vals stores the objective value at each evaluation; reverse the sign when maximizing the target
					func_vals = result.func_vals
					func_vals = np.sort(func_vals)[::-1] # Sort in descending order
					func_vals_list.append(func_vals)
					temperature_obs.append(result.fun)
					c_inhibitors.append(result.x[0])
					c_inhibitors1.append(result.x[1])
					result_temp = np.array([result.fun,pressure,result.x[0],result.x[1]])
					np.savetxt(f"./optimization/{save_path}/temp/func_vals_{pressure}.temp",func_vals,fmt="%f")
					np.savetxt(f"./optimization/{save_path}/temp/results_{pressure}.temp",result_temp,fmt="%f")
				# Save the temperature optimization history
				func_vals_array = np.array(func_vals_list).T
				df_func_vals = pd.DataFrame(func_vals_array, columns=Pressures)
				# Save predicted temperature, pressure, and inhibitor concentrations
				df_func_vals.to_excel(f"./optimization/{save_path}/optimization_func_vals_{select_columns[indexs[0]]}_{select_columns[indexs[1]]}.xlsx", index=False)
				print(temperature_obs,Pressures,c_inhibitors,c_inhibitors1)
				ob = np.column_stack((temperature_obs,Pressures, c_inhibitors,c_inhibitors1)) #.reshape(1,-1)
				df_results = pd.DataFrame(ob, columns=["Objective T (K)", "Objective P (MPa)",
													 f"{select_columns[indexs[0]]}", f"{select_columns[indexs[1]]}"])
				# Save predicted temperature, pressure, and inhibitor concentrations
				df_results.to_excel(f"./optimization/{save_path}/optimization_results_{select_columns[indexs[0]]}_{select_columns[indexs[1]]}.xlsx", index=False)

		# # Report the optimization results
		# print(f">>> Optimal Parameters: {c_inhibitors}")
		# print(f">>> Minimum Target Value: {temperature_obs} K")  # Reverse the sign because the target is maximized
		# print(f">>> Objective Value: {func_vals_list}")

	# Visualize the Bayesian optimization process
	fig = fd.add_fig(figsize=(17.5, 6))
	ax = fd.add_ax(fig,subplot=(121))
	df_func_vals = pd.read_excel(f"./optimization/{save_path}/optimization_func_vals_{select_columns[indexss[0][0]]}_{select_columns[indexss[0][1]]}.xlsx")
	print(df_func_vals)
	for col in df_func_vals.columns:
		ax.plot(df_func_vals[col],label=f"{col} MPa")
	ax.set_xlabel("Number of Evaluations")
	ax.set_ylabel(fr"Objective $\regular \it T_H$")
	fd.set_fig(ax)
	plt.grid(True, which='both', color='gray', linestyle='--', linewidth=0.5)
	
	ax = fd.add_ax(fig,subplot=(122))
	pw = pd.read_excel(pwf)
	x, y = pw["T (K)"], pw["P (MPa)"]
	ax.scatter(x,y,s=50,edgecolors=(0, 0, 0),color="w",alpha=0.5,label="Pure water")
	xf,yf, fit = fd.polyfitting(x,y,degree=3)
	print(fit)
	ax.plot(xf,yf,color="r",label="Fitting")

	df_results = pd.read_excel(f"./optimization/{save_path}/optimization_results_{select_columns[indexss[0][0]]}_{select_columns[indexss[0][1]]}.xlsx")
	print(df_results)
	ax.scatter(df_results["Objective T (K)"],df_results["Objective P (MPa)"],s=100,edgecolors=(0, 0, 0),color="c",marker="*",label="Predicted point ")
	# # ax.text(273, 55, r'$y = 0.004336 x^3 - 3.626 x^2 + 1011 x - 9.4 \times 10^4$', fontsize=18, ha='left')
	ax.set_xlabel(r"$\regular \it T_H$ (K)")
	ax.set_ylabel(r"$\regular \it P_H$ (MPa)")
	ax.set_xlim(245,315)
	ax.set_ylim(-5,200)
	fd.set_fig(ax)
	plt.grid(True, which='both', color='gray', linestyle='--', linewidth=0.5)
	# plt.savefig(f"./imgs/GBR_ObjectiveValue_{label}_{search}_single.png",dpi=300,transparent=True)
	# plt.show()