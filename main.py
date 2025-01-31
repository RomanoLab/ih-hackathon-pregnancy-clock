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
combinations_list = list(combinations(stimulation, 2))
for sti_com in combinations_list:
    sti1=sti_com[0]
    sti2=sti_com[1]

    input1=longdf[longdf.Stims==sti1]
    input1 = input1.loc[:, input1.nunique() > 1]
    input2=longdf[longdf.Stims==sti2]
    input2 = input2.loc[:, input2.nunique() > 1]

    input = input1.merge(input2, on=["ID", "GA"], suffixes=('_' + str(sti1), '_' + str(sti2)))
   
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

    print(sti1 + "&" + sti2)
    print(f"rmse: {rmse:.4f}")

    result = permutation_importance(rf, X_test, y_test, n_repeats=10, random_state=0, n_jobs=-1)
    perm_imp_df = pd.DataFrame({'Feature': X_test.columns, 'Permutation Importance': result.importances_mean}).sort_values('Permutation Importance', ascending=False)
    selectft=perm_imp_df[perm_imp_df['Permutation Importance']>10**(-3)].Feature.tolist()

    X_train=X_train[selectft]
    X_test=X_test[selectft]

    rf = RandomForestRegressor(n_estimators=110, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    rmse=np.sqrt(metrics.mean_squared_error(y_test, y_pred))

    # print(sti1 + " & " + sti2)
    print(f"After fts select rmse: {rmse:.4f}")

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
    perm_imp_df.to_csv(f'{dest}/permutation_importance.csv',index=False)
    selectft=perm_imp_df[perm_imp_df['Permutation Importance']>10**(-3)].Feature.tolist()

    X_train=X_train[selectft]
    X_test=X_test[selectft]

    rf = RandomForestRegressor(n_estimators=110, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    
    rmse=np.sqrt(metrics.mean_squared_error(y_test, y_pred))

    print(sti1 + " & " + sti2 + " & " + sti3)
    print(f"After fts select rmse: {rmse:.4f}")

#%%
combinations_list = list(combinations(stimulation, 4))
for sti_com in combinations_list:
    sti1=sti_com[0]
    sti2=sti_com[1]
    sti3=sti_com[2]
    sti4=sti_com[3]

    input1=longdf[longdf.Stims==sti1]
    input1 = input1.loc[:, input1.nunique() > 1]
    input2=longdf[longdf.Stims==sti2]
    input2 = input2.loc[:, input2.nunique() > 1]
    input3=longdf[longdf.Stims==sti3]
    input3 = input3.loc[:, input3.nunique() > 1]
    input4=longdf[longdf.Stims==sti4]
    input4 = input4.loc[:, input4.nunique() > 1]
    
    input = input1.merge(input2, on=["ID", "GA"], suffixes=('_' + str(sti1), '_' + str(sti2))).merge(input3, on=["ID", "GA"]).merge(input3, on=["ID", "GA"], suffixes=('_' + str(sti3), '_' + str(sti4)))
   
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

    rf = RandomForestRegressor(n_estimators=120, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    rmse=np.sqrt(metrics.mean_squared_error(y_test, y_pred))

    print(sti1 + " & " + sti2 + " & " + sti3+ " & " + sti4)
    print(f"rmse: {rmse:.4f}")

    result = permutation_importance(rf, X_test, y_test, n_repeats=10, random_state=0, n_jobs=-1)
    perm_imp_df = pd.DataFrame({'Feature': X_test.columns, 'Permutation Importance': result.importances_mean}).sort_values('Permutation Importance', ascending=False)
    selectft=perm_imp_df[perm_imp_df['Permutation Importance']>10**(-3)].Feature.tolist()

    X_train=X_train[selectft]
    X_test=X_test[selectft]

    rf = RandomForestRegressor(n_estimators=120, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    rmse=np.sqrt(metrics.mean_squared_error(y_test, y_pred))

    print(sti1 + " & " + sti2 + " & " + sti3+ " & " + sti4)
    print(f"After fts select rmse: {rmse:.4f}")

#%%
combinations_list = list(combinations(stimulation, 5))
for sti_com in combinations_list:
    sti1=sti_com[0]
    sti2=sti_com[1]
    sti3=sti_com[2]
    sti4=sti_com[3]
    sti5=sti_com[4]

    input1=longdf[longdf.Stims==sti1]
    input1 = input1.loc[:, input1.nunique() > 1]
    input2=longdf[longdf.Stims==sti2]
    input2 = input2.loc[:, input2.nunique() > 1]
    input3=longdf[longdf.Stims==sti3]
    input3 = input3.loc[:, input3.nunique() > 1]
    input4=longdf[longdf.Stims==sti4]
    input4 = input4.loc[:, input4.nunique() > 1]
    input5=longdf[longdf.Stims==sti5]
    input5 = input5.loc[:, input5.nunique() > 1]
    
    input = input1.merge(input2, on=["ID", "GA"], suffixes=('_' + str(sti1), '_' + str(sti2))).merge(input3, on=["ID", "GA"]).merge(input3, on=["ID", "GA"], suffixes=('_' + str(sti3), '_' + str(sti4))).merge(input5, on=["ID", "GA"])
   
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

    rf = RandomForestRegressor(n_estimators=120, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    rmse=np.sqrt(metrics.mean_squared_error(y_test, y_pred))

    print(f"rmse: {rmse:.4f}")

    result = permutation_importance(rf, X_test, y_test, n_repeats=10, random_state=0, n_jobs=-1)
    perm_imp_df = pd.DataFrame({'Feature': X_test.columns, 'Permutation Importance': result.importances_mean}).sort_values('Permutation Importance', ascending=False)
    selectft=perm_imp_df[perm_imp_df['Permutation Importance']>10**(-3)].Feature.tolist()

    X_train=X_train[selectft]
    X_test=X_test[selectft]

    rf = RandomForestRegressor(n_estimators=120, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    rmse=np.sqrt(metrics.mean_squared_error(y_test, y_pred))

    print(f"After fts select rmse: {rmse:.4f}")

#%%
#All the good one contain GMCSF and IFNa. Unstim might help increase a bit.