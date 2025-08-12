import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
import joblib

# Load the dataset
df = pd.read_csv("sdn_dataset_2022.csv", sep=';', comment='@', engine='python')

# Clean column names
df.columns = df.columns.str.strip().str.lower()

# Drop missing values 
df = df.dropna()

# Clean and label 'category' column
df['category'] = df['category'].str.strip().str.upper()

# Map traffic_class
def label_traffic(label):
    if isinstance(label, str) and label.startswith('STR-'):
        return 'Streaming'
    else:
        return 'Non-Streaming'

df['traffic_class'] = df['category'].apply(label_traffic)

# Confirm class distribution
print("✅ Data cleaned.")
print(df['traffic_class'].value_counts())
print(df[['category', 'traffic_class']].head())

# 1. Drop non-feature columns
X = df.drop(columns=['category', 'traffic_class'])

# Store feature columns for QoS pipeline compatibility
feature_columns = X.columns.tolist()
print(f"\n✅ Feature columns extracted: {len(feature_columns)} features")
print(f"Features: {feature_columns[:5]}...")  # Show first 5 features

# 2. Encode the target
le = LabelEncoder()
y = le.fit_transform(df['traffic_class'])  # 0 = Non-Streaming, 1 = Streaming

print(f"\n✅ Label encoding completed.")
print(f"Classes: {le.classes_}")
print(f"Encoded values: {dict(zip(le.classes_, range(len(le.classes_))))}")

# 3. Balance with SMOTE
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X, y)

print(f"\n✅ SMOTE balancing completed.")
print(f"Original class distribution: {pd.Series(y).value_counts().sort_index().values}")
print(f"Balanced class distribution: {pd.Series(y_res).value_counts().sort_index().values}")

# 4. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_res, y_res, test_size=0.2, stratify=y_res, random_state=42
)

print(f"\n✅ Train/test split completed.")
print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# 5. Train model
print(f"\n🚀 Training Random Forest model...")
clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
clf.fit(X_train, y_train)

# 6. Evaluate
y_pred = clf.predict(X_test)
print("\n✅ Classification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# 7. Save models and feature schema for QoS pipeline
print(f"\n💾 Saving models for QoS pipeline integration...")

# Save model and label encoder
joblib.dump(clf, "traffic_classifier.pkl")
joblib.dump(le, "label_encoder.pkl")
joblib.dump(feature_columns, "feature_columns.pkl")

print("✅ Models saved successfully:")
print("  - traffic_classifier.pkl")
print("  - label_encoder.pkl") 
print("  - feature_columns.pkl")

# 8. Display model information for verification
print(f"\n📊 Model Summary:")
print(f"Model type: {type(clf).__name__}")
print(f"Number of estimators: {clf.n_estimators}")
print(f"Number of features: {len(feature_columns)}")
print(f"Number of classes: {len(le.classes_)}")
print(f"Classes: {list(le.classes_)}")

# 9. Feature importance (top 10)
if hasattr(clf, 'feature_importances_'):
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': clf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\n🎯 Top 10 Most Important Features:")
    print(feature_importance.head(10).to_string(index=False))

print(f"\n🎉 Training pipeline completed successfully!")
print(f"Models are ready for integration with QoS pipeline.")