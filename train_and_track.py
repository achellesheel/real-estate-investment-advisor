import os
import time
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.preprocessing import StandardScaler, PowerTransformer
import mlflow
import mlflow.sklearn
import mlflow.xgboost

def main():
    # Resolve all paths relative to this script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 1. Initialize MLflow with a SQLite database backend to support Model Registry
    db_path = f"sqlite:///{os.path.join(script_dir, 'mlflow.db')}"
    mlflow.set_tracking_uri(db_path)
    
    experiment_name = "Real_Estate_Investment_Advisor"
    mlflow.set_experiment(experiment_name)
    
    print("Loading raw dataset for fitting preprocessing pipelines...")
    df_raw = pd.read_csv(os.path.join(script_dir, 'india_housing_prices.csv')).iloc[:, 0:23]
    df_raw = df_raw.drop_duplicates()
    
    # Fill missing values just in case
    for col in df_raw.select_dtypes(include=['number']).columns:
        df_raw[col] = df_raw[col].fillna(df_raw[col].median())
    for col in df_raw.select_dtypes(exclude=['number']).columns:
        df_raw[col] = df_raw[col].fillna(df_raw[col].mode()[0])

    print("Fitting preprocessing transformers...")
    # Calculate price per sqft
    df_raw['Price_per_SqFt'] = df_raw['Price_in_Lakhs'] / df_raw['Size_in_SqFt']
    df_raw['Age_of_Property'] = 2025 - df_raw['Year_Built']
    
    # Yeo-Johnson Power Transformer
    pt = PowerTransformer(method='yeo-johnson')
    df_raw['Price_per_SqFt_Transformed'] = pt.fit_transform(df_raw[['Price_per_SqFt']])
    
    # Target Encoding maps
    locality_map = df_raw.groupby('Locality')['Price_in_Lakhs'].mean().to_dict()
    city_map = df_raw.groupby('City')['Price_in_Lakhs'].mean().to_dict()
    # Overall mean for fallback
    global_mean_price = df_raw['Price_in_Lakhs'].mean()
    
    df_raw['Locality_Target_Encoded'] = df_raw['Locality'].map(locality_map)
    df_raw['City_Target_Encoded'] = df_raw['City'].map(city_map)
    
    # Ordinal mapping
    transport_map = {'Low': 0, 'Medium': 1, 'High': 2}
    furnished_map = {'Unfurnished': 0, 'Semi-furnished': 1, 'Furnished': 2}
    df_raw['Public_Transport_Accessibility'] = df_raw['Public_Transport_Accessibility'].map(transport_map)
    df_raw['Furnished_Status'] = df_raw['Furnished_Status'].map(furnished_map)
    
    # StandardScaler
    continuous_cols = [
        'BHK', 'Size_in_SqFt', 'Price_in_Lakhs', 'Price_per_SqFt_Transformed',
        'Year_Built', 'Floor_No', 'Total_Floors', 'Age_of_Property', 
        'Nearby_Schools', 'Nearby_Hospitals', 'Locality_Target_Encoded', 'City_Target_Encoded'
    ]
    scaler = StandardScaler()
    scaler.fit(df_raw[continuous_cols])
    
    # Save preprocessing objects
    print("Saving preprocessing pipelines...")
    preprocessing_assets = {
        "scaler": scaler,
        "power_transformer": pt,
        "locality_map": locality_map,
        "city_map": city_map,
        "global_mean_price": global_mean_price,
        "continuous_cols": continuous_cols
    }
    with open(os.path.join(script_dir, "preprocessing_assets.pkl"), "wb") as f:
        pickle.dump(preprocessing_assets, f)

    print("Loading preprocessed advanced features...")
    df_adv = pd.read_csv(os.path.join(script_dir, 'india_housing_prices_advanced.csv'))
    df_clean = pd.read_csv(os.path.join(script_dir, 'india_housing_prices_clean.csv'))

    # Merge Targets
    if 'Good_Investment' not in df_adv.columns:
        df_adv = df_adv.merge(df_clean[['ID', 'Good_Investment']], on='ID', how='inner')

    if 'Future_Price_5Y' not in df_adv.columns:
        appreciation_rate = 0.05
        appreciation_rate += 0.01 * df_clean['Property_Type_Villa'].astype(int)
        appreciation_rate += 0.01 * (df_clean['Public_Transport_Accessibility_Encoded'] == 2).astype(int)
        appreciation_rate += 0.005 * (df_clean['Nearby_Schools'] >= 5).astype(int)
        appreciation_rate += 0.005 * (df_clean['Nearby_Hospitals'] >= 5).astype(int)
        appreciation_rate += 0.01 * (df_clean['Age_of_Property'] <= 5).astype(int)
        appreciation_rate -= 0.01 * (df_clean['Age_of_Property'] >= 20).astype(int)

        df_clean['Future_Price_5Y'] = df_clean['Price_in_Lakhs'] * ((1 + appreciation_rate) ** 5)
        df_adv = df_adv.merge(df_clean[['ID', 'Future_Price_5Y']], on='ID', how='inner')

    # Separate features and targets
    X = df_adv.drop(columns=['ID', 'Good_Investment', 'Future_Price_5Y'])
    y_class = df_adv['Good_Investment']
    y_reg = df_adv['Future_Price_5Y']

    # Split data
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X, y_class, test_size=0.2, random_state=42)
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X, y_reg, test_size=0.2, random_state=42)

    # Save feature names for Streamlit lookup
    with open(os.path.join(script_dir, "feature_names.pkl"), "wb") as f:
        pickle.dump(list(X.columns), f)

    # Define classification models
    classifiers = {
        "Logistic_Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision_Tree_Classifier": DecisionTreeClassifier(max_depth=10, random_state=42),
        "Random_Forest_Classifier": RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1),
        "Gradient_Boosting_Classifier": GradientBoostingClassifier(n_estimators=50, max_depth=5, random_state=42),
        "XGBoost_Classifier": XGBClassifier(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)
    }

    # Define regression models
    regressors = {
        "Linear_Regression": LinearRegression(n_jobs=-1),
        "Decision_Tree_Regressor": DecisionTreeRegressor(max_depth=10, random_state=42),
        "Random_Forest_Regressor": RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1),
        "Gradient_Boosting_Regressor": GradientBoostingRegressor(n_estimators=50, max_depth=5, random_state=42),
        "XGBoost_Regressor": XGBRegressor(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)
    }

    # Train and Track Classifiers
    best_clf_name = None
    best_clf_score = -1
    best_clf_run_id = None
    
    print("\n--- Training Classifiers ---")
    for name, clf in classifiers.items():
        with mlflow.start_run(run_name=f"Clf_{name}"):
            print(f"Training and logging {name}...")
            t0 = time.time()
            clf.fit(X_train_c, y_train_c)
            t_train = time.time() - t0
            
            y_pred = clf.predict(X_test_c)
            y_prob = clf.predict_proba(X_test_c)[:, 1] if hasattr(clf, "predict_proba") else y_pred
            
            acc = accuracy_score(y_test_c, y_pred)
            prec = precision_score(y_test_c, y_pred)
            rec = recall_score(y_test_c, y_pred)
            f1 = f1_score(y_test_c, y_pred)
            auc = roc_auc_score(y_test_c, y_prob)
            
            # Log params
            mlflow.log_param("model_type", "classifier")
            mlflow.log_param("model_name", name)
            if hasattr(clf, "max_depth") and clf.max_depth:
                mlflow.log_param("max_depth", clf.max_depth)
            if hasattr(clf, "n_estimators") and clf.n_estimators:
                mlflow.log_param("n_estimators", clf.n_estimators)
            
            # Log metrics
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("precision", prec)
            mlflow.log_metric("recall", rec)
            mlflow.log_metric("f1_score", f1)
            mlflow.log_metric("roc_auc", auc)
            mlflow.log_metric("training_time_seconds", t_train)
            
            # Log model artifact
            if "XGBoost" in name:
                mlflow.xgboost.log_model(clf, "model")
            else:
                mlflow.sklearn.log_model(clf, "model")
            
            # Log preprocessing assets as artifacts
            mlflow.log_artifact(os.path.join(script_dir, "preprocessing_assets.pkl"))
            mlflow.log_artifact(os.path.join(script_dir, "feature_names.pkl"))
            
            print(f"  Accuracy: {acc:.4f} | ROC AUC: {auc:.4f} | Time: {t_train:.2f}s")
            
            if auc > best_clf_score:
                best_clf_score = auc
                best_clf_name = name
                best_clf_run_id = mlflow.active_run().info.run_id

    # Train and Track Regressors
    best_reg_name = None
    best_reg_score = -1
    best_reg_run_id = None
    
    print("\n--- Training Regressors ---")
    for name, reg in regressors.items():
        with mlflow.start_run(run_name=f"Reg_{name}"):
            print(f"Training and logging {name}...")
            t0 = time.time()
            reg.fit(X_train_r, y_train_r)
            t_train = time.time() - t0
            
            y_pred = reg.predict(X_test_r)
            
            rmse = np.sqrt(mean_squared_error(y_test_r, y_pred))
            mae = mean_absolute_error(y_test_r, y_pred)
            r2 = r2_score(y_test_r, y_pred)
            
            # Log params
            mlflow.log_param("model_type", "regressor")
            mlflow.log_param("model_name", name)
            if hasattr(reg, "max_depth") and reg.max_depth:
                mlflow.log_param("max_depth", reg.max_depth)
            if hasattr(reg, "n_estimators") and reg.n_estimators:
                mlflow.log_param("n_estimators", reg.n_estimators)
            
            # Log metrics
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("r2_score", r2)
            mlflow.log_metric("training_time_seconds", t_train)
            
            # Log model artifact
            if "XGBoost" in name:
                mlflow.xgboost.log_model(reg, "model")
            else:
                mlflow.sklearn.log_model(reg, "model")
                
            mlflow.log_artifact(os.path.join(script_dir, "preprocessing_assets.pkl"))
            mlflow.log_artifact(os.path.join(script_dir, "feature_names.pkl"))
            
            print(f"  RMSE: {rmse:.2f} | R2: {r2:.4f} | Time: {t_train:.2f}s")
            
            if r2 > best_reg_score:
                best_reg_score = r2
                best_reg_name = name
                best_reg_run_id = mlflow.active_run().info.run_id

    # Register the Best Models in the MLflow Model Registry
    print("\n--- Model Registration ---")
    if best_clf_run_id:
        clf_model_uri = f"runs:/{best_clf_run_id}/model"
        print(f"Registering Best Classifier '{best_clf_name}' (ROC AUC: {best_clf_score:.4f}) to registry...")
        mlflow.register_model(clf_model_uri, "Best_Real_Estate_Classifier")
        
        # Save a copy locally as a fallback in case database loading has issues in Streamlit
        best_clf = classifiers[best_clf_name]
        with open(os.path.join(script_dir, "best_classifier.pkl"), "wb") as f:
            pickle.dump(best_clf, f)
        print(f"Best classifier saved to: {os.path.join(script_dir, 'best_classifier.pkl')}")
        
    if best_reg_run_id:
        reg_model_uri = f"runs:/{best_reg_run_id}/model"
        print(f"Registering Best Regressor '{best_reg_name}' (R2 Score: {best_reg_score:.4f}) to registry...")
        mlflow.register_model(reg_model_uri, "Best_Real_Estate_Regressor")
        
        # Save a copy locally as a fallback
        best_reg = regressors[best_reg_name]
        with open(os.path.join(script_dir, "best_regressor.pkl"), "wb") as f:
            pickle.dump(best_reg, f)
        print(f"Best regressor saved to: {os.path.join(script_dir, 'best_regressor.pkl')}")

    print("\nTraining and experiment tracking completed successfully!")

if __name__ == "__main__":
    main()
