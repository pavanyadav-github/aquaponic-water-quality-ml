# src/preprocessing.py
import pandas as pd
from sklearn.impute import KNNImputer

def load_and_preprocess(path):
    """Load dataset, impute missing values, create multiclass target."""
    df = pd.read_csv(path)

    # Handle missing values
    imputer = KNNImputer(n_neighbors=5)
    df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

    # Add multiclass columns
    df_imputed['Plant'] = df_imputed.apply(
        lambda r: 1 if (5.5 <= r['pH'] <= 7.5 and 16 <= r['Temperature'] <= 30 
                        and r['Dissolved Oxygen'] > 3 and r['Ammonia'] < 30 and r['Nitrite'] < 1) else 0, axis=1)
    
    df_imputed['Bacteria'] = df_imputed.apply(
        lambda r: 1 if (6 <= r['pH'] <= 8.5 and 14 <= r['Temperature'] <= 34 
                        and 4 <= r['Dissolved Oxygen'] <= 8 and r['Ammonia'] < 3 and r['Nitrite'] < 1) else 0, axis=1)
    
    df_imputed['Warm Water Fish'] = df_imputed.apply(
        lambda r: 1 if (6 <= r['pH'] <= 8.5 and 22 <= r['Temperature'] <= 32 
                        and 4 <= r['Dissolved Oxygen'] <= 6 and r['Ammonia'] < 3 
                        and r['Nitrate'] < 400 and r['Nitrite'] < 1) else 0, axis=1)
    
    df_imputed['Cold Water Fish'] = df_imputed.apply(
        lambda r: 1 if (6 <= r['pH'] <= 8.5 and 10 <= r['Temperature'] <= 21 
                        and 6 <= r['Dissolved Oxygen'] <= 8 and r['Ammonia'] < 1 
                        and r['Nitrate'] < 400 and r['Nitrite'] < 0.1) else 0, axis=1)

    # Compute Output column
    def calculate_output(row):
        binary_str = f"{int(row['Plant'])}{int(row['Bacteria'])}{int(row['Warm Water Fish'])}{int(row['Cold Water Fish'])}"
        return int(binary_str, 2)

    df_imputed['Output'] = df_imputed.apply(calculate_output, axis=1)

    # Features and target
    features = ['pH', 'Dissolved Oxygen', 'Temperature', 'Ammonia', 'Nitrite', 'Nitrate']
    X = df_imputed[features]
    y = df_imputed['Output']

    return X, y, df_imputed
