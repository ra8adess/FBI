import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import numpy as np

# تحديد المسار الكامل للملف حسب جهازك
file_path = r'D:\PROJECT AI\Football_Dataset_2015_2025.csv'
df = pd.read_csv(file_path)

# تجهيز المتغير المستهدف بناء على الفريق المضيف
conditions = [df['Winner'] == 'Home Team', df['Winner'] == 'Away Team', df['Winner'] == 'Draw']
choices = ['Win', 'Loss', 'Draw']
df['result'] = np.select(conditions, choices, default='Draw')

# تجهيز الميزات
X = df[['Shots (Home)', 'Shots (Away)', 'Possession % (Home)', 'Possession % (Away)', 'Home Goals', 'Away Goals', 'Year']]

# ترميز النتائج
le = LabelEncoder()
y = le.fit_transform(df['result'])

# تقسيم البيانات
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# تدريب النموذج
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# حفظ النموذج والمحول في نفس المسار
joblib.dump(model, r'D:\PROJECT AI\football_match_predictor.pkl')
joblib.dump(le, r'D:\PROJECT AI\label_encoder.pkl')

print("✅ تم تدريب النموذج وحفظه باستخدام البيانات الجديدة بنجاح")
