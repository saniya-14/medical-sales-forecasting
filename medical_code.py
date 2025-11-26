#Loading the necessary Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

#Loading the dataset
file_path = "medical_data.csv"  # ensure this file is in the same folder
df = pd.read_csv('medical_data.csv')
print("Dataset loaded successfully!")
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())

#Basic cleaning of dataset

# Standardize column names (remove spaces & special chars)
df.columns = df.columns.str.strip().str.replace(' ', '_').str.replace('/', '_')

# Remove duplicates
df.drop_duplicates(inplace=True)

# Handle missing values
for col in df.columns:
    if df[col].dtype == 'O':  # categorical/text
        df[col] = df[col].astype(str).str.strip().replace("nan", "Unknown")
    else:
        df[col] = df[col].fillna(df[col].median())

print("Cleaned missing values and removed duplicates.")

#Exploratory Data Analysis (EDA)

#Convert Daily_customer_count to numeric form
def convert_customer_range(value):
    value = str(value).strip().lower()
    if "more" in value:
        return 120
    elif "100" in value:
        return 75
    elif "50" in value:
        return 35
    elif "20" in value:
        return 10
    elif "0" in value:
        return 5
    else:
        return np.nan

df['Daily_customer_count_num'] = df['Daily_customer_count'].apply(convert_customer_range)
df['Daily_customer_count_num'] = df['Daily_customer_count_num'].fillna(df['Daily_customer_count_num'].median())

print("Converted 'Daily_customer_count' ranges into numeric values.")

#Calculating average and creating Sales field

average_customer_count = df['Daily_customer_count_num'].mean()
df['Sales'] = df['Daily_customer_count_num'] * average_customer_count

print(f"Average daily customer count: {average_customer_count:.2f}")
print("'Sales' column created successfully!")

# Save the cleaned version
df.to_csv("cleaned_medical_data_with_sales.csv", index=False)
print("Cleaned file saved as cleaned_medical_data_with_sales.csv")

#Loading the clean data for visualizations
# Load cleaned data
data = pd.read_csv("cleaned_medical_data_with_sales.csv")
data.columns = data.columns.str.strip()

# Visualizating data from multiple perspectives
sns.set_style("whitegrid")
sns.set_palette("pastel")

#Distribution of Customer Count
plt.figure(figsize=(8,5))
sns.histplot(df['Daily_customer_count_num'], kde=True, color="skyblue", bins=30)
plt.title("Distribution of Daily Customer Count")
plt.xlabel("Customers per Day")
plt.ylabel("Frequency")
plt.show()

#sales by Store (Top 5)
if 'Medical_store_name' in df.columns:
    store_sales = df.groupby('Medical_store_name')['Sales'].sum().sort_values(ascending=False).head(5).reset_index()
    plt.figure(figsize=(10,5))
    sns.barplot(x='Sales', y='Medical_store_name', data=store_sales, color='orange')
    plt.title("Top 5 Medical Stores by Total Sales")
    plt.xlabel("Total Sales")
    plt.ylabel("Store Name")
    plt.show()

#Average Sales per Medicine Category
if 'High_demand_categroies' in df.columns:
    avg_sales = df.groupby('High_demand_categroies')['Sales'].mean().reset_index()
    plt.figure(figsize=(9,5))
    sns.barplot(x='Sales', y='High_demand_categroies', data=avg_sales, color='violet')
    plt.title("Average Sales per High-Demand Category")
    plt.xlabel("Average Sales")
    plt.ylabel("Category")
    plt.show()

#Sales by Payment Method
if 'preferred_payment_methods' in df.columns:
    plt.figure(figsize=(8,5))
    sns.barplot(x='preferred_payment_methods', y='Sales', data=df, color='lightgreen', ci=None)
    plt.title("Sales by Preferred Payment Method")
    plt.xlabel("Payment Method")
    plt.ylabel("Total Sales")
    plt.xticks(rotation=30)
    plt.show()

#Sales vs Disease Demand Increase (Trend)
if 'Increased_disease_demand' in df.columns:
    plt.figure(figsize=(8,5))
    sns.boxplot(x='Increased_disease_demand', y='Sales', data=df, color='salmon')
    plt.title("Sales vs Increased Disease Demand")
    plt.xlabel("Increased Disease Demand")
    plt.ylabel("Sales")
    plt.show()

#Relationship Between Bulk Orders & Sales
if 'bulk_small_purchases' in df.columns:
    plt.figure(figsize=(7,5))
    sns.boxplot(x='bulk_small_purchases', y='Sales', data=df, color='gold')
    plt.title("Sales by Bulk/Small Purchases")
    plt.xlabel("Purchase Type")
    plt.ylabel("Sales")
    plt.show()

#Correlation Heatmap (All numeric columns)
plt.figure(figsize=(9,6))
sns.heatmap(data.corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap (All Numeric Columns)", fontsize=14, weight='bold')
plt.tight_layout()
plt.show()

#Pairplot to explore feature relationships
numeric_cols = [col for col in ['Daily_customer_count_num', 'Sales'] if col in df.columns]
if len(numeric_cols) >= 2:
    sns.pairplot(df[numeric_cols], diag_kind="kde")
    plt.suptitle("Pairwise Relationships Between Key Features", y=1.02)
    plt.show()

#Top 10 Stores with Highest Average Sales per Customer
if 'Medical_store_name' in df.columns:
    df['Sales_per_customer'] = df['Sales'] / df['Daily_customer_count_num']
    top_stores = df.groupby('Medical_store_name')['Sales_per_customer'].mean().nlargest(10).reset_index()
    plt.figure(figsize=(10,5))
    sns.barplot(x='Sales_per_customer', y='Medical_store_name', data=top_stores, color='deepskyblue')
    plt.title("Top 10 Stores by Average Sales per Customer")
    plt.xlabel("Average Sales per Customer")
    plt.ylabel("Store Name")
    plt.show()


print("\nAll visualizations generated successfully!")

#Creating ML Model

#Loading clean dataset for model
df = pd.read_csv("cleaned_medical_data_with_sales.csv")

#Basic checks
print("Data loaded successfully. Shape:", df.shape)

#Encode categorical columns
le = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    df[col] = le.fit_transform(df[col])

#Define features (X) and target (y)
target = "Sales"
features = [col for col in df.columns if col != target]
X = df[features]
y = df[target]

#split data into training & testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#Model 1: Random Forest Regressor
rf_model = RandomForestRegressor(
    n_estimators=200, random_state=42, max_depth=10
)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)


#Model 2: XGBoost Regressor

xgb_model = XGBRegressor(
    n_estimators=300, learning_rate=0.05, max_depth=6,
    subsample=0.8, colsample_bytree=0.8, random_state=42
)
xgb_model.fit(X_train, y_train)
xgb_preds = xgb_model.predict(X_test)

#Evaluate both models
def evaluate_model(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"\n{name} Performance:")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R² Score: {r2:.3f}")

evaluate_model("Random Forest", y_test, rf_preds)
evaluate_model("XGBoost", y_test, xgb_preds)

#Compare Actual vs Predicted Sales
plt.figure(figsize=(8,5))
sns.scatterplot(x=y_test, y=rf_preds, alpha=0.6, label='Random Forest', color='skyblue')
sns.scatterplot(x=y_test, y=xgb_preds, alpha=0.6, label='XGBoost', color='salmon')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.title("Actual vs Predicted Sales Comparison", fontsize=14, weight='bold')
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.legend()
plt.tight_layout()
plt.show()

#Feature Importance (from XGBoost)
xgb_importances = pd.Series(xgb_model.feature_importances_, index=features).sort_values(ascending=False)

plt.figure(figsize=(8,4))
sns.barplot(x=xgb_importances.head(10), y=xgb_importances.head(10).index, palette="pastel")
plt.title("Top 10 Important Features (XGBoost)", fontsize=14, weight='bold')
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()

print("\nForecasting models trained, evaluated, and visualized successfully!")


