import numpy as np
import pandas as pd

def calculate_vif(df):
    df_numeric = df.select_dtypes(include=[np.number])
    vif_data = pd.DataFrame()
    vif_data["Feature"] = df_numeric.columns
    vif_data["VIF"] = [round(variance_inflation_factor(df_numeric.values, i), 2)
                       for i in range(df_numeric.shape[1])]
    return vif_data

    
def remove_intercolinarity(X_train, threshold):
    X_train_bis = X_train.copy()
    vif_results = calculate_vif(X_train_bis)
    vif_results = vif_results.set_index("Feature")
    name = vif_results.sort_values("VIF", ascending=False).iloc[0].name
    value = vif_results.sort_values("VIF", ascending=False).iloc[0].values[0]
    while value > threshold: 
        del X_train_bis[name]
        vif_results = calculate_vif(X_train_bis)
        vif_results = vif_results.set_index("Feature")
        name = vif_results.sort_values("VIF", ascending=False).iloc[0].name
        value = vif_results.sort_values("VIF", ascending=False).iloc[0].values[0]
    return vif_results



