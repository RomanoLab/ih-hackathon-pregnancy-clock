#!/usr/bin/env python3.9

#%%
import pandas as pd
import numpy as np
import os
import random
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn.inspection import permutation_importance
from itertools import combinations
import matplotlib.pyplot as plt
import seaborn as sns

# os.chdir("C:/Users/TRAMANH-LAPTOP/Desktop/I3H")
# %%
print("start of processing")
src = os.environ['INPUT_DIR']
dest = os.environ['OUTPUT_DIR']

meta=pd.read_csv(f'{src}/FlowRepository_FR-FCM-Z3FH_samplelog.csv')
immu=pd.read_csv(f'{src}/Immunome_Labor_Onset_BG-Immunome_Labor_Onset_BG20Statistics-241025_1628.csv')

#%%
immu[['Plate', 'ID', 'GA', 'Stims']] = immu['file'].str.split('_', n=3, expand=True)
immu=immu[['Plate', 'ID', 'GA', 'Stims'] + [col for col in immu.columns if col not in ['Plate', 'ID', 'GA', 'Stims']]]
immu['Stims']=immu['Stims'].str.split('_').str[0]
immu=immu[immu.Plate!="Plate14"]
immu = immu.drop(columns=["CD19|Eu151Di___151Eu_p38|median", "CD56 (NK)|Dy161Di___161Dy_cPARP|median", "CD8|Dy161Di___161Dy_cPARP|median"])
immu[['ID','GA']]=immu[['ID','GA']].apply(pd.to_numeric)

#%%
meta.columns = ['Plate', 'ID', 'GA', 'Stims', 'EGA', 'TL', 'Timepoint', 'Source Population', 'Source FCS File', 'Unique Population Name']

#%%
bigdf=immu.merge(meta[['Plate','ID','GA','Stims','EGA','TL','Timepoint']])
bigdf = bigdf[~bigdf['file'].str.contains(r'-1\.fcs$', regex=True)]
# %%
longdf=bigdf.drop(columns=['Plate','file','TL','EGA','Timepoint'])
stimulation=longdf.Stims.unique().tolist()

#%%
for sti in stimulation:
    input=longdf[longdf.Stims==sti]
    input = input.loc[:, input.nunique() > 1]
   
    random.seed(18399)
    patient=input.ID.unique().tolist()
    random.shuffle(patient)
    split_index = int(len(patient) * 0.85)
    train_patients = patient[:split_index]
    test_patients = patient[split_index:]

    train_df=input[input.ID.isin(train_patients)]
    test_df=input[input.ID.isin(test_patients)]

    X_train=train_df.drop(['ID','GA'],axis=1)
    X_test=test_df.drop(['ID','GA'],axis=1)
    y_train=train_df['GA']
    y_test=test_df['GA']

    rf = RandomForestRegressor(n_estimators=110, random_state=42, max_depth=30)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    
    rmse=np.sqrt(metrics.mean_squared_error(y_test, y_pred))

    print(sti)
    print(f"rmse: {rmse:.4f}")

    result = permutation_importance(rf, X_test, y_test, n_repeats=10, random_state=0, n_jobs=-1)
    perm_imp_df = pd.DataFrame({'Feature': X_test.columns, 'Permutation Importance': result.importances_mean}).sort_values('Permutation Importance', ascending=False)
    
    perm_imp_df.to_csv(f'{dest}/{sti}_perm_importance.csv')

    input=longdf[longdf.Stims==sti]
    input = input.loc[:, input.nunique() > 1]
    selectft=perm_imp_df[perm_imp_df['Permutation Importance']>10**(-2)].Feature.tolist()
    input=input[["ID", "GA"] + selectft]

    random.seed(18399)
    patient=input.ID.unique().tolist()
    random.shuffle(patient)
    split_index = int(len(patient) * 0.85)
    train_patients = patient[:split_index]
    test_patients = patient[split_index:]

    train_df=input[input.ID.isin(train_patients)]
    test_df=input[input.ID.isin(test_patients)]

    X_train=train_df.drop(['ID','GA'],axis=1)
    X_test=test_df.drop(['ID','GA'],axis=1)
    y_train=train_df['GA']
    y_test=test_df['GA']

    rf = RandomForestRegressor(n_estimators=110, random_state=42, max_depth=30)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    rmse=np.sqrt(metrics.mean_squared_error(y_test, y_pred))


    print(f"After fts select rmse: {rmse:.4f}")

#%%
# combinations_list = list(combinations(stimulation, 2))
# for sti_com in combinations_list:
#     sti1=sti_com[0]
#     sti2=sti_com[1]

#     input1=longdf[longdf.Stims==sti1]
#     input1 = input1.loc[:, input1.nunique() > 1]
#     input2=longdf[longdf.Stims==sti2]
#     input2 = input2.loc[:, input2.nunique() > 1]

#     input = input1.merge(input2, on=["ID", "GA"], suffixes=('_' + str(sti1), '_' + str(sti2)))
   
#     random.seed(18399)
#     patient=input.ID.unique().tolist()
#     random.shuffle(patient)
#     split_index = int(len(patient) * 0.85)
#     train_patients = patient[:split_index]
#     test_patients = patient[split_index:]

#     train_df=input[input.ID.isin(train_patients)]
#     test_df=input[input.ID.isin(test_patients)]

#     X_train=train_df.drop(['ID','GA'],axis=1)
#     X_test=test_df.drop(['ID','GA'],axis=1)
#     y_train=train_df['GA']
#     y_test=test_df['GA']

#     rf = RandomForestRegressor(n_estimators=110, random_state=42)
#     rf.fit(X_train, y_train)
#     y_pred = rf.predict(X_test)
#     rmse=np.sqrt(metrics.mean_squared_error(y_test, y_pred))

#     print(sti1 + "&" + sti2)
#     print(f"rmse: {rmse:.4f}")

#     result = permutation_importance(rf, X_test, y_test, n_repeats=10, random_state=0, n_jobs=-1)
#     perm_imp_df = pd.DataFrame({'Feature': X_test.columns, 'Permutation Importance': result.importances_mean}).sort_values('Permutation Importance', ascending=False)
#     selectft=perm_imp_df[perm_imp_df['Permutation Importance']>10**(-3)].Feature.tolist()

#     X_train=X_train[selectft]
#     X_test=X_test[selectft]

#     rf = RandomForestRegressor(n_estimators=110, random_state=42)
#     rf.fit(X_train, y_train)
#     y_pred = rf.predict(X_test)
#     rmse=np.sqrt(metrics.mean_squared_error(y_test, y_pred))

#     # print(sti1 + " & " + sti2)
#     print(f"After fts select rmse: {rmse:.4f}")

#%%
# Load data and label conditions
gmcsf = pd.read_csv(f'{dest}/GMCSF_perm_importance.csv')
gmcsf["Condition"] = "GMCSF"

ifna = pd.read_csv(f'{dest}/IFNa_perm_importance.csv')
ifna["Condition"] = "IFNa"

unstim = pd.read_csv(f'{dest}/unstim_perm_importance.csv')
unstim["Condition"] = "Unstim"

# Combine all data
df = pd.concat([gmcsf, ifna, unstim])

# Select top 10 features for each condition
top_gmcsf = gmcsf.nlargest(10, 'Permutation Importance')
top_ifna = ifna.nlargest(10, 'Permutation Importance')
top_unstim = unstim.nlargest(10, 'Permutation Importance')

# Combine into one dataframe
top_features_df = pd.concat([top_gmcsf, top_ifna, top_unstim])

# Define the color palette for each condition
condition_palette = {
    'GMCSF': 'green',
    'IFNa': 'red',
    'Unstim': 'blue'
}

# Create FacetGrid
plt.figure(figsize=(20, 50))
g = sns.FacetGrid(top_features_df, col="Condition", height=5, col_wrap=1, sharey=False, aspect=2)

# Custom plotting function to sort features by importance
def plot_sorted_bars(data, **kwargs):
    condition = data['Condition'].iloc[0]
    color = condition_palette[condition]
    sorted_data = data.sort_values('Permutation Importance', ascending=False)
    ax = plt.gca()
    sns.barplot(x='Permutation Importance', y='Feature', data=sorted_data, 
                ax=ax, color=color)

# Map the custom plotting function
g.map_dataframe(plot_sorted_bars)

# Formatting
g.set_axis_labels("Permutation Importance", "Feature")
g.set_titles("{col_name}")
g.fig.suptitle("Top 10 Features by Condition", fontsize=16)
g.fig.tight_layout()
g.fig.subplots_adjust(left=0.2, right=0.95, top=0.90)

plt.savefig(f'{dest}/feature_importance_facet.png')
plt.show()

# %%
combinations_list = list(combinations(stimulation, 3))
for sti_com in combinations_list:
    sti1=sti_com[0]
    sti2=sti_com[1]
    sti3=sti_com[2]

    input1=longdf[longdf.Stims==sti1]
    input1 = input1.loc[:, input1.nunique() > 1]
    input2=longdf[longdf.Stims==sti2]
    input2 = input2.loc[:, input2.nunique() > 1]
    input3=longdf[longdf.Stims==sti3]
    input3 = input3.loc[:, input3.nunique() > 1]
    
    input = input1.merge(input2, on=["ID", "GA"], suffixes=('_' + str(sti1), '_' + str(sti2))).merge(input3, on=["ID", "GA"])
   
    random.seed(18399)
    patient=input.ID.unique().tolist()
    random.shuffle(patient)
    split_index = int(len(patient) * 0.85)
    train_patients = patient[:split_index]
    test_patients = patient[split_index:]

    train_df=input[input.ID.isin(train_patients)]
    test_df=input[input.ID.isin(test_patients)]

    X_train=train_df.drop(['ID','GA'],axis=1)
    X_test=test_df.drop(['ID','GA'],axis=1)
    y_train=train_df['GA']
    y_test=test_df['GA']

    rf = RandomForestRegressor(n_estimators=110, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    rmse=np.sqrt(metrics.mean_squared_error(y_test, y_pred))

    print(sti1 + " & " + sti2 + " & " + sti3)
    print(f"rmse: {rmse:.4f}")

    result = permutation_importance(rf, X_test, y_test, n_repeats=10, random_state=0, n_jobs=-1)
    perm_imp_df = pd.DataFrame({'Feature': X_test.columns, 'Permutation Importance': result.importances_mean}).sort_values('Permutation Importance', ascending=False)
    perm_imp_df.to_csv(f'{dest}/{sti1}_{sti2}_{sti3}_perm_importance.csv',index=False)
    selectft=perm_imp_df[perm_imp_df['Permutation Importance']>10**(-3)].Feature.tolist()

    X_train=X_train[selectft]
    X_test=X_test[selectft]

    rf = RandomForestRegressor(n_estimators=110, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    
    rmse=np.sqrt(metrics.mean_squared_error(y_test, y_pred))

    print(f"After fts select rmse: {rmse:.4f}")

#%%
# combinations_list = list(combinations(stimulation, 4))
# for sti_com in combinations_list:
#     sti1=sti_com[0]
#     sti2=sti_com[1]
#     sti3=sti_com[2]
#     sti4=sti_com[3]

#     input1=longdf[longdf.Stims==sti1]
#     input1 = input1.loc[:, input1.nunique() > 1]
#     input2=longdf[longdf.Stims==sti2]
#     input2 = input2.loc[:, input2.nunique() > 1]
#     input3=longdf[longdf.Stims==sti3]
#     input3 = input3.loc[:, input3.nunique() > 1]
#     input4=longdf[longdf.Stims==sti4]
#     input4 = input4.loc[:, input4.nunique() > 1]
    
#     input = input1.merge(input2, on=["ID", "GA"], suffixes=('_' + str(sti1), '_' + str(sti2))).merge(input3, on=["ID", "GA"]).merge(input3, on=["ID", "GA"], suffixes=('_' + str(sti3), '_' + str(sti4)))
   
#     random.seed(18399)
#     patient=input.ID.unique().tolist()
#     random.shuffle(patient)
#     split_index = int(len(patient) * 0.85)
#     train_patients = patient[:split_index]
#     test_patients = patient[split_index:]

#     train_df=input[input.ID.isin(train_patients)]
#     test_df=input[input.ID.isin(test_patients)]

#     X_train=train_df.drop(['ID','GA'],axis=1)
#     X_test=test_df.drop(['ID','GA'],axis=1)
#     y_train=train_df['GA']
#     y_test=test_df['GA']

#     rf = RandomForestRegressor(n_estimators=120, random_state=42)
#     rf.fit(X_train, y_train)
#     y_pred = rf.predict(X_test)
#     rmse=np.sqrt(metrics.mean_squared_error(y_test, y_pred))

#     print(sti1 + " & " + sti2 + " & " + sti3+ " & " + sti4)
#     print(f"rmse: {rmse:.4f}")

#     result = permutation_importance(rf, X_test, y_test, n_repeats=10, random_state=0, n_jobs=-1)
#     perm_imp_df = pd.DataFrame({'Feature': X_test.columns, 'Permutation Importance': result.importances_mean}).sort_values('Permutation Importance', ascending=False)
#     selectft=perm_imp_df[perm_imp_df['Permutation Importance']>10**(-3)].Feature.tolist()

#     X_train=X_train[selectft]
#     X_test=X_test[selectft]

#     rf = RandomForestRegressor(n_estimators=120, random_state=42)
#     rf.fit(X_train, y_train)
#     y_pred = rf.predict(X_test)
#     rmse=np.sqrt(metrics.mean_squared_error(y_test, y_pred))

#     print(sti1 + " & " + sti2 + " & " + sti3+ " & " + sti4)
#     print(f"After fts select rmse: {rmse:.4f}")

# #%%
# combinations_list = list(combinations(stimulation, 5))
# for sti_com in combinations_list:
#     sti1=sti_com[0]
#     sti2=sti_com[1]
#     sti3=sti_com[2]
#     sti4=sti_com[3]
#     sti5=sti_com[4]

#     input1=longdf[longdf.Stims==sti1]
#     input1 = input1.loc[:, input1.nunique() > 1]
#     input2=longdf[longdf.Stims==sti2]
#     input2 = input2.loc[:, input2.nunique() > 1]
#     input3=longdf[longdf.Stims==sti3]
#     input3 = input3.loc[:, input3.nunique() > 1]
#     input4=longdf[longdf.Stims==sti4]
#     input4 = input4.loc[:, input4.nunique() > 1]
#     input5=longdf[longdf.Stims==sti5]
#     input5 = input5.loc[:, input5.nunique() > 1]
    
#     input = input1.merge(input2, on=["ID", "GA"], suffixes=('_' + str(sti1), '_' + str(sti2))).merge(input3, on=["ID", "GA"]).merge(input3, on=["ID", "GA"], suffixes=('_' + str(sti3), '_' + str(sti4))).merge(input5, on=["ID", "GA"])
   
#     random.seed(18399)
#     patient=input.ID.unique().tolist()
#     random.shuffle(patient)
#     split_index = int(len(patient) * 0.85)
#     train_patients = patient[:split_index]
#     test_patients = patient[split_index:]

#     train_df=input[input.ID.isin(train_patients)]
#     test_df=input[input.ID.isin(test_patients)]

#     X_train=train_df.drop(['ID','GA'],axis=1)
#     X_test=test_df.drop(['ID','GA'],axis=1)
#     y_train=train_df['GA']
#     y_test=test_df['GA']

#     rf = RandomForestRegressor(n_estimators=120, random_state=42)
#     rf.fit(X_train, y_train)
#     y_pred = rf.predict(X_test)
#     rmse=np.sqrt(metrics.mean_squared_error(y_test, y_pred))

#     print(f"rmse: {rmse:.4f}")

#     result = permutation_importance(rf, X_test, y_test, n_repeats=10, random_state=0, n_jobs=-1)
#     perm_imp_df = pd.DataFrame({'Feature': X_test.columns, 'Permutation Importance': result.importances_mean}).sort_values('Permutation Importance', ascending=False)
#     selectft=perm_imp_df[perm_imp_df['Permutation Importance']>10**(-3)].Feature.tolist()

#     X_train=X_train[selectft]
#     X_test=X_test[selectft]

#     rf = RandomForestRegressor(n_estimators=120, random_state=42)
#     rf.fit(X_train, y_train)
#     y_pred = rf.predict(X_test)
#     rmse=np.sqrt(metrics.mean_squared_error(y_test, y_pred))

#     print(f"After fts select rmse: {rmse:.4f}")

# #%%
# #All the good one contain GMCSF and IFNa. Unstim might help increase a bit.

#%%
sti1='GMCSF'
sti2='IFNa'
sti3='unstim'

input1=longdf[longdf.Stims==sti1]
input1 = input1.loc[:, input1.nunique() > 1]
input2=longdf[longdf.Stims==sti2]
input2 = input2.loc[:, input2.nunique() > 1]
input3=longdf[longdf.Stims==sti3]
input3 = input3.loc[:, input3.nunique() > 1]

input = input1.merge(input2, on=["ID", "GA"], suffixes=('_' + str(sti1), '_' + str(sti2))).merge(input3, on=["ID", "GA"])

perm_imp_df=pd.read_csv(f'{dest}/GMCSF_IFNa_unstim_perm_importance.csv')
selectft=perm_imp_df[perm_imp_df['Permutation Importance']>10**(-3)].Feature.tolist()
input=input[["ID", "GA"] + selectft]

#%%
random.seed(18399)
patient=input.ID.unique().tolist()
random.shuffle(patient)
split_index = int(len(patient) * 0.85)
train_patients = patient[:split_index]
test_patients = patient[split_index:]

train_df=input[input.ID.isin(train_patients)]
test_df=input[input.ID.isin(test_patients)]

X_train=train_df.drop(['ID','GA'],axis=1)
X_test=test_df.drop(['ID','GA'],axis=1)
y_train=train_df['GA']
y_test=test_df['GA']

#%%
rf = RandomForestRegressor(n_estimators=110, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

rmse=np.sqrt(metrics.mean_squared_error(y_test, y_pred))
r2=metrics.r2_score(y_test, y_pred)
mape = np.mean(np.abs((y_test - y_pred) / np.abs(y_test)))
print('r2:', r2)
# %%
import matplotlib.pyplot as plt
# Create scatter plot
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, alpha=0.5, label="Predictions vs Actual")

# Labels and title
plt.xlabel("Actual Values (y_test)")
plt.ylabel("Predicted Values (y_pred)")
plt.title("Actual vs Predicted Values")
plt.xlim(22, 40)
plt.ylim(22, 40)
plt.legend()
plt.grid(True)

# Show plot
plt.savefig(f'{dest}/best_model.png')
plt.show()