# SMS Spam Detector with Streamlit UI - 修正版本
"""
使用方式：
執行時於命令列輸入
    streamlit run streamlit_sms_spam_detector.py
並將 sms_spam_no_header.csv 放在同目錄下
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import warnings
warnings.filterwarnings('ignore')

# 設定 Streamlit 頁面
st.set_page_config(page_title="SMS Spam Detector", layout="wide")

# 設定中文字型（不依賴系統字型）
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

st.title("📧 SMS Spam Detector")
st.markdown("Logistic Regression 垃圾郵件偵測系統")

# ===============================
# 1. 資料載入（使用 cache 加速）
# ===============================

@st.cache_data(show_spinner=True)
def load_data():
    """載入並預處理資料集"""
    try:
        df = pd.read_csv('sms_spam_no_header.csv', header=None, names=['label', 'message'])
        df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})
        df['message_length'] = df['message'].str.len()
        df['word_count'] = df['message'].str.split().str.len()
        return df
    except FileNotFoundError:
        st.error("❌ 找不到 sms_spam_no_header.csv 檔案！請確保此檔案在同一目錄下。")
        st.stop()

df = load_data()

# ===============================
# 2. 側邊欄 - 資料探索區
# ===============================

with st.sidebar:
    st.header('📊 Data Overview')
    
    # 基本統計
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Messages", df.shape[0])
    col2.metric("Ham", sum(df['label'] == 'ham'))
    col3.metric("Spam", sum(df['label'] == 'spam'))
    
    # 樣本預覽
    st.subheader("Sample Messages")
    st.dataframe(df[['label', 'message']].head(5), use_container_width=True)
    
    st.divider()
    
    # 類別分布圖
    st.subheader("Label Distribution")
    fig1, ax1 = plt.subplots(figsize=(6, 3))
    label_counts = df['label'].value_counts()
    colors = ['#87CEEB', '#FF6B6B']
    ax1.pie(label_counts.values, labels=label_counts.index, autopct='%1.1f%%', 
            colors=colors, textprops={'fontsize': 10})
    ax1.set_title('Ham vs Spam', fontsize=12, fontweight='bold')
    st.pyplot(fig1, use_container_width=True)
    
    st.divider()
    
    # 訊息長度分布
    st.subheader("Message Length Distribution")
    fig2, ax2 = plt.subplots(figsize=(6, 3))
    ax2.hist(df[df['label'] == 'ham']['message_length'], bins=40, alpha=0.6, 
            label='Ham', color='#87CEEB', density=True)
    ax2.hist(df[df['label'] == 'spam']['message_length'], bins=40, alpha=0.6, 
            label='Spam', color='#FF6B6B', density=True)
    ax2.set_xlabel('Message Length', fontsize=9)
    ax2.set_ylabel('Density', fontsize=9)
    ax2.legend()
    ax2.set_title('Length Comparison', fontsize=11, fontweight='bold')
    st.pyplot(fig2, use_container_width=True)
    
    st.divider()
    
    # 文字雲
    st.subheader("Word Clouds")
    col_wc1, col_wc2 = st.columns(2)
    
    try:
        with col_wc1:
            st.write("**Ham**")
            ham_text = ' '.join(df[df['label'] == 'ham']['message'].astype(str))
            wc_ham = WordCloud(width=250, height=200, background_color='white', 
                             prefer_horizontal=0.7).generate(ham_text)
            fig_wc1, ax_wc1 = plt.subplots(figsize=(4, 3))
            ax_wc1.imshow(wc_ham, interpolation='bilinear')
            ax_wc1.axis('off')
            st.pyplot(fig_wc1, use_container_width=True)
        
        with col_wc2:
            st.write("**Spam**")
            spam_text = ' '.join(df[df['label'] == 'spam']['message'].astype(str))
            wc_spam = WordCloud(width=250, height=200, background_color='white', 
                              prefer_horizontal=0.7).generate(spam_text)
            fig_wc2, ax_wc2 = plt.subplots(figsize=(4, 3))
            ax_wc2.imshow(wc_spam, interpolation='bilinear')
            ax_wc2.axis('off')
            st.pyplot(fig_wc2, use_container_width=True)
    except Exception as e:
        st.warning(f"❌ Word Cloud 載入失敗: {str(e)}")

# ===============================
# 3. 模型訓練（使用 cache 加速）
# ===============================

@st.cache_resource(show_spinner=True)
def train_model(df):
    """訓練 Logistic Regression 模型"""
    X = df['message']
    y = df['label_num']
    
    # 資料分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)
    
    # 特徵提取
    vectorizer = TfidfVectorizer(max_features=3000, stop_words='english', max_df=0.95, min_df=2)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    # 模型訓練
    model = LogisticRegression(max_iter=1000, random_state=42, solver='lbfgs')
    model.fit(X_train_tfidf, y_train)
    
    return model, vectorizer, X_test, y_test, X_test_tfidf

model, vectorizer, X_test, y_test, X_test_tfidf = train_model(df)

# ===============================
# 4. 主頁 - 模型效能展示
# ===============================

st.header("🎯 Model Performance")

# 預測
y_pred = model.predict(X_test_tfidf)
y_proba = model.predict_proba(X_test_tfidf)[:, 1]
acc = accuracy_score(y_test, y_pred)

# 計算指標
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

# 效能指標卡片
col_metric1, col_metric2, col_metric3, col_metric4 = st.columns(4)
with col_metric1:
    st.metric("Accuracy", f"{acc*100:.2f}%")
with col_metric2:
    st.metric("Sensitivity (Recall)", f"{sensitivity*100:.2f}%")
with col_metric3:
    st.metric("Specificity", f"{specificity*100:.2f}%")
with col_metric4:
    st.metric("ROC AUC", f"{roc_auc:.3f}")

st.divider()

# 視覺化圖表
col_chart1, col_chart2 = st.columns(2)

with col_chart1:
    st.subheader("Confusion Matrix")
    fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'], 
                ax=ax_cm, cbar_kws={'label': 'Count'})
    ax_cm.set_xlabel('Predicted Label', fontsize=10)
    ax_cm.set_ylabel('True Label', fontsize=10)
    st.pyplot(fig_cm, use_container_width=True)

with col_chart2:
    st.subheader("ROC Curve")
    fig_roc, ax_roc = plt.subplots(figsize=(5, 4))
    ax_roc.plot(fpr, tpr, color='#FF6B6B', lw=2, label=f'ROC (AUC={roc_auc:.3f})')
    ax_roc.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label='Random')
    ax_roc.set_xlabel('False Positive Rate', fontsize=10)
    ax_roc.set_ylabel('True Positive Rate', fontsize=10)
    ax_roc.set_title('ROC Curve', fontsize=11, fontweight='bold')
    ax_roc.legend(loc='lower right', fontsize=9)
    st.pyplot(fig_roc, use_container_width=True)

st.divider()

# 詳細分類報告
st.subheader("Classification Report")
report_dict = classification_report(y_test, y_pred, target_names=['Ham', 'Spam'], 
                                    output_dict=True)
report_df = pd.DataFrame(report_dict).T
st.dataframe(report_df.round(3), use_container_width=True)

# ===============================
# 5. 互動式預測
# ===============================

st.header("💬 Test Your Message")

user_input = st.text_area(
    "Enter an SMS message to classify:",
    placeholder="Congratulations! You've won a free prize. Call now!",
    height=100
)

if user_input:
    msg_tfidf = vectorizer.transform([user_input])
    pred = model.predict(msg_tfidf)[0]
    proba = model.predict_proba(msg_tfidf)[0]
    
    pred_label = "🔴 SPAM" if pred == 1 else "🟢 HAM"
    
    col_pred1, col_pred2 = st.columns([1, 2])
    with col_pred1:
        st.subheader(pred_label)
    with col_pred2:
        st.metric("Spam Probability", f"{proba[1]*100:.1f}%")
        st.metric("Ham Probability", f"{proba[0]*100:.1f}%")

# ===============================
# 6. 特徵重要性
# ===============================

st.header("🔍 Top Important Features")

features = vectorizer.get_feature_names_out()
importances = np.abs(model.coef_[0])
top_indices = np.argsort(importances)[-15:][::-1]

fig_imp, ax_imp = plt.subplots(figsize=(8, 5))
sns.barplot(x=importances[top_indices], y=features[top_indices], palette='viridis', ax=ax_imp)
ax_imp.set_xlabel('Importance (Absolute Coefficient)', fontsize=10)
ax_imp.set_title('Top 15 Important Keywords', fontsize=12, fontweight='bold')
st.pyplot(fig_imp, use_container_width=True)

st.success("✅ Application loaded successfully!")
