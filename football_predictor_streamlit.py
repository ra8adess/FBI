import streamlit as st
import pandas as pd
import joblib
from PIL import Image

# إعداد الصفحة
st.set_page_config(
    page_title="⚽ توقع نتيجة مباراة كرة القدم", 
    layout="centered",
    page_icon="⚽"
)

# ===== إعداد الخلفية =====
def set_background():
    st.markdown(
        """
        <style>
        .stApp {
            background-image: linear-gradient(rgba(0, 0, 0, 0.7), rgba(0, 0, 0, 0.7)), 
                            url("https://i.imgur.com/WvT2sLF.jpg");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            font-weight: bold;
            border-radius: 5px;
            padding: 10px 24px;
        }
        .stButton>button:hover {
            background-color: #45a049;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

set_background()

# ===== تحميل النموذج والبيانات =====
@st.cache_resource
def load_model():
    try:
        model = joblib.load('football_match_predictor.pkl')
        return model
    except FileNotFoundError:
        st.error("❌ ملف النموذج غير موجود. يرجى التأكد من وجود ملف 'football_match_predictor.pkl'")
        return None

@st.cache_data
def load_data():
    try:
        matches = pd.read_csv('Football_Dataset_2015_2025.csv')
        teams = sorted(pd.unique(matches[['Home Team', 'Away Team']].values.ravel('K')))
        return teams
    except FileNotFoundError:
        st.error("❌ ملف البيانات غير موجود. يرجى التأكد من وجود ملف 'Football_Dataset_2015_2025.csv'")
        return []

model = load_model()
teams = load_data()

# ===== واجهة المستخدم =====
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>🏟️ نظام توقع نتائج المباريات</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: white;'>استخدم هذه الأداة للتنبؤ بنتائج مباريات كرة القدم بناءً على الإحصائيات التاريخية</p>", unsafe_allow_html=True)
st.markdown("---")

# ===== اختيار الفرق =====
st.markdown("<h3 style='color: #4CAF50;'>🔽 اختيار الفرق</h3>", unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    home_team = st.selectbox("🏠 الفريق المستضيف", teams, key='home_team')
with col2:
    away_team = st.selectbox("🚗 الفريق الضيف", [t for t in teams if t != home_team], key='away_team')

if home_team == away_team:
    st.warning("⚠️ لا يمكن اختيار نفس الفريق في الجهتين. الرجاء اختيار فريق مختلف.")

# ===== مدخلات إحصائيات المباراة =====
st.markdown("<h3 style='color: #4CAF50;'>📊 إحصائيات المباراة</h3>", unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    st.markdown("<h4 style='text-align: center; color: white;'>الفريق المستضيف</h4>", unsafe_allow_html=True)
    shots_home = st.slider("🎯 عدد التسديدات", 0, 30, 12, key='shots_home')
    possession_home = st.slider("🕹️ نسبة الاستحواذ (%)", 0, 100, 55, key='possession_home')
    goals_home_prev = st.slider("⚽ الأهداف في المباريات السابقة", 0, 10, 2, key='goals_home_prev')

with col2:
    st.markdown("<h4 style='text-align: center; color: white;'>الفريق الضيف</h4>", unsafe_allow_html=True)
    shots_away = st.slider("🎯 عدد التسديدات", 0, 30, 10, key='shots_away')
    possession_away = st.slider("🕹️ نسبة الاستحواذ (%)", 0, 100, 45, key='possession_away')
    goals_away_prev = st.slider("⚽ الأهداف في المباريات السابقة", 0, 10, 1, key='goals_away_prev')

year = st.slider("📅 سنة المباراة", 2015, 2025, 2024, key='year')

st.markdown("---")

# ===== التوقع وعرض النتائج =====
if st.button("🔮 تنبؤ بالنتيجة", key='predict'):
    if home_team == away_team:
        st.error("❌ الرجاء اختيار فريقين مختلفين!")
    elif not model:
        st.error("❌ النظام غير قادر على عمل تنبؤ بسبب مشكلة في تحميل النموذج")
    else:
        with st.spinner('جاري تحليل البيانات وتوقع النتيجة...'):
            # إنشاء dataframe للإدخال
            input_df = pd.DataFrame({
                'Shots (Home)': [shots_home],
                'Shots (Away)': [shots_away],
                'Possession % (Home)': [possession_home],
                'Possession % (Away)': [100 - possession_home],  # التأكد من أن المجموع 100%
                'Home Goals': [goals_home_prev],
                'Away Goals': [goals_away_prev],
                'Year': [year]
            })
            
            # التنبؤ
            try:
                pred = model.predict(input_df)[0]
                
                # تحويل التوقع إلى نتيجة مقروءة
                try:
                    le = joblib.load('label_encoder.pkl')
                    pred_label = le.inverse_transform([pred])[0]
                except:
                    pred_labels = {0: 'تعادل', 1: 'فوز المستضيف', 2: 'فوز الضيف'}
                    pred_label = pred_labels.get(pred, "غير معروف")
                
                # عرض النتيجة
                st.markdown("---")
                if pred_label == 'فوز المستضيف':
                    st.success(f"## 🏆 النتيجة المتوقعة: فوز {home_team} 🎉")
                elif pred_label == 'فوز الضيف':
                    st.success(f"## 🏆 النتيجة المتوقعة: فوز {away_team} 🎉")
                else:
                    st.success(f"## 🏆 النتيجة المتوقعة: تعادل ⚖️")
                
                # إظهار الإحصائيات المدخلة
                st.markdown("### 📊 الإحصائيات المدخلة:")
                stats_df = pd.DataFrame({
                    'الإحصائية': ['التسديدات', 'نسبة الاستحواذ', 'الأهداف السابقة'],
                    home_team: [shots_home, f"{possession_home}%", goals_home_prev],
                    away_team: [shots_away, f"{possession_away}%", goals_away_prev]
                }).set_index('الإحصائية')
                
                st.table(stats_df.style.set_properties(**{'background-color': '#2a2a2a', 'color': 'white'}))
                
            except Exception as e:
                st.error(f"❌ حدث خطأ أثناء التنبؤ: {str(e)}")

# ===== تذييل الصفحة =====
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: white;'>
    <p>تم تطوير هذا النظام باستخدام خوارزميات تعلم الآلة</p>
    <p>© 2023 نظام توقع نتائج المباريات - جميع الحقوق محفوظة</p>
    </div>
    """,
    unsafe_allow_html=True
)