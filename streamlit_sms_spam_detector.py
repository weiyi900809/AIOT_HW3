# SMS Spam Detector with Streamlit UI - 輕量化版本（無 wordcloud）
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
    accuracy_score, classification_report, confusion_matrix, roc_curve, auc
)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# 設定 Streamlit 頁面
st.set_page_config(page_title="SMS Spam Detector", layout="wide", initial_sidebar_state="expanded")

# 設定圖表風格
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

st.title("📧 SMS Spam Detector")
st.markdown("**Logistic Regression 垃圾郵件偵測系統**")

# ===============================
# 1. 資料載入（使用 cache 加速）
# ===============================

@st.cache_data(show_spinner=True)
def load_data():
    """載入並預處理資料集"""
    try:
        df = pd.read_csv('sms_spam_no_header.csv', header=None, names=['label', 'message'], encoding='latin-1')
        df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})
        df['message_length'] = df['message'].str.len()
        df['word_count'] = df['message'].str.split().str.len()
        return df
    except FileNotFoundError:
        st.error("❌ 找不到 sms_spam_no_header.csv 檔案！請確保此檔案在同一目錄下。")
        st.stop()
    except Exception as e:
        st.error(f"❌ 資料載入失敗: {str(e)}")
        st.stop()

df = load_data()
st.success("✅ 資料載入成功")

# ===============================
# 2. 側邊欄 - 資料探索區
# ===============================

with st.sidebar:
    st.header("📊 Data Overview")
    
    # 基本統計
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total", df.shape[0])
    with col2:
        st.metric("Ham", sum(df['label'] == 'ham'))
    with col3:
        st.metric("Spam", sum(df['label'] == 'spam'))
    
    st.divider()
    
    # 類別分布
    st.subheader("📈 Label Distribution")
    fig1, ax1 = plt.subplots(figsize=(5, 3))
    label_counts = df['label'].value_counts()
    colors_pie = ['#3498db', '#e74c3c']
    wedges, texts, autotexts = ax1.pie(label_counts.values, labels=label_counts.index, 
                                         autopct='%1.1f%%', colors=colors_pie, startangle=90)
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    ax1.set_title('Ham vs Spam', fontsize=11, fontweight='bold')
    st.pyplot(fig1, use_container_width=True)
    
    st.divider()
    
    # 訊息長度統計
    st.subheader("📏 Message Length Stats")
    stats = df.groupby('label')['message_length'].describe()[['mean', '50%', 'max']]
    st.dataframe(stats.round(0), use_container_width=True)
    
    # 訊息長度分布圖
    fig2, ax2 = plt.subplots(figsize=(5, 3))
    ax2.hist(df[df['label'] == 'ham']['message_length'], bins=40, alpha=0.6, 
            label='Ham', color='#3498db', density=True)
    ax2.hist(df[df['label'] == 'spam']['message_length'], bins=40, alpha=0.6, 
            label='Spam', color='#e74c3c', density=True)
    ax2.set_xlabel('Message Length', fontsize=9)
    ax2.set_ylabel('Density', fontsize=9)
    ax2.legend(fontsize=9)
    ax2.set_title('Length Distribution', fontsize=10, fontweight='bold')
    st.pyplot(fig2, use_container_width=True)
    
    st.divider()
    
    # 樣本預覽
    st.subheader("📝 Sample Messages")
    sample_idx = st.slider("Select sample index", 0, min(20, len(df)-1), 0)
    sample = df.iloc[sample_idx]
    col_label, col_msg = st.columns([1, 3])
    with col_label:
        st.metric("Label", sample['label'].upper())
    with col_msg:
        st.write(f"**Message:** {sample['message'][:100]}...")

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
    vectorizer = TfidfVectorizer(max_features=3000, stop_words='english', 
                                max_df=0.95, min_df=2, ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    # 模型訓練
    model = LogisticRegression(max_iter=1000, random_state=42, solver='lbfgs', C=1.0)
    model.fit(X_train_tfidf, y_train)
    
    return model, vectorizer, X_test, y_test, X_test_tfidf

# 訓練模型
with st.spinner("🤖 訓練模型中..."):
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
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

# 效能指標卡片
col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
with col_m1:
    st.metric("Accuracy", f"{acc*100:.1f}%", delta="📊")
with col_m2:
    st.metric("Sensitivity", f"{sensitivity*100:.1f}%", delta="📈")
with col_m3:
    st.metric("Specificity", f"{specificity*100:.1f}%", delta="📈")
with col_m4:
    st.metric("Precision", f"{precision*100:.1f}%", delta="📈")
with col_m5:
    st.metric("ROC AUC", f"{roc_auc:.3f}", delta="🎯")

st.divider()

# 視覺化圖表
col_chart1, col_chart2, col_chart3 = st.columns(3)

with col_chart1:
    st.subheader("Confusion Matrix")
    fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'], 
                ax=ax_cm, cbar_kws={'label': 'Count'}, annot_kws={'size': 14})
    ax_cm.set_xlabel('Predicted Label', fontsize=10, fontweight='bold')
    ax_cm.set_ylabel('True Label', fontsize=10, fontweight='bold')
    st.pyplot(fig_cm, use_container_width=True)

with col_chart2:
    st.subheader("ROC Curve")
    fig_roc, ax_roc = plt.subplots(figsize=(5, 4))
    ax_roc.plot(fpr, tpr, color='#e74c3c', lw=3, label=f'AUC = {roc_auc:.3f}')
    ax_roc.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label='Random')
    ax_roc.fill_between(fpr, tpr, alpha=0.3, color='#e74c3c')
    ax_roc.set_xlabel('False Positive Rate', fontsize=10, fontweight='bold')
    ax_roc.set_ylabel('True Positive Rate', fontsize=10, fontweight='bold')
    ax_roc.legend(loc='lower right', fontsize=10)
    ax_roc.grid(alpha=0.3)
    st.pyplot(fig_roc, use_container_width=True)

with col_chart3:
    st.subheader("Test Set Distribution")
    fig_test, ax_test = plt.subplots(figsize=(5, 4))
    pred_dist = pd.Series(y_pred).value_counts()
    colors_test = ['#3498db', '#e74c3c']
    ax_test.bar(['Predicted Ham', 'Predicted Spam'], 
               [pred_dist.get(0, 0), pred_dist.get(1, 0)],
               color=colors_test, alpha=0.7, edgecolor='black', linewidth=2)
    ax_test.set_ylabel('Count', fontsize=10, fontweight='bold')
    ax_test.set_title('Test Set Predictions', fontsize=11, fontweight='bold')
    for i, v in enumerate([pred_dist.get(0, 0), pred_dist.get(1, 0)]):
        ax_test.text(i, v + 5, str(v), ha='center', fontweight='bold')
    st.pyplot(fig_test, use_container_width=True)

st.divider()

# 詳細分類報告
st.subheader("📋 Classification Report")
report_dict = classification_report(y_test, y_pred, target_names=['Ham', 'Spam'], 
                                    output_dict=True)
report_df = pd.DataFrame(report_dict).T
st.dataframe(report_df.round(3), use_container_width=True)

# ===============================
# 5. 互動式預測
# ===============================

st.header("💬 Test Your Message")
st.write("輸入一則 SMS 訊息，系統會自動判斷是 Ham（正常訊息）還是 Spam（垃圾訊息）")

user_input = st.text_area(
    "Enter an SMS message:",
    placeholder="e.g., Congratulations! You've won a prize. Call now!",
    height=80,
    key="user_message"
)

if user_input and len(user_input.strip()) > 0:
    msg_tfidf = vectorizer.transform([user_input])
    pred = model.predict(msg_tfidf)[0]
    proba = model.predict_proba(msg_tfidf)[0]
    
    # 顯示預測結果
    col_res1, col_res2, col_res3 = st.columns([2, 1, 1])
    
    with col_res1:
        if pred == 1:
            st.error("🔴 **SPAM** - This message is likely spam!")
        else:
            st.success("🟢 **HAM** - This message appears to be legitimate!")
    
    with col_res2:
        st.metric("Spam Score", f"{proba[1]*100:.1f}%")
    
    with col_res3:
        st.metric("Ham Score", f"{proba[0]*100:.1f}%")
    
    # 信心度指示
    st.write("**Confidence Level:**")
    max_prob = max(proba)
    col_conf = st.columns([int(max_prob * 100), 100 - int(max_prob * 100)])
    with col_conf[0]:
        st.success(f"{'█' * int(max_prob * 20)}")
    st.write(f"信心度: {max_prob*100:.1f}%")

# ===============================
# 6. 特徵重要性
# ===============================

st.header("🔍 Top Important Keywords")
st.write("這些關鍵字對模型判斷是否為 Spam 最有影響力")

features = vectorizer.get_feature_names_out()
importances = np.abs(model.coef_[0])
top_indices = np.argsort(importances)[-12:][::-1]

fig_imp, ax_imp = plt.subplots(figsize=(8, 5))
colors_imp = ['#e74c3c' if model.coef_[0][i] > 0 else '#3498db' for i in top_indices]
bars = ax_imp.barh(range(len(top_indices)), importances[top_indices], color=colors_imp, alpha=0.7, edgecolor='black')
ax_imp.set_yticks(range(len(top_indices)))
ax_imp.set_yticklabels(features[top_indices], fontsize=10)
ax_imp.set_xlabel('Importance Score', fontsize=10, fontweight='bold')
ax_imp.set_title('Top 12 Important Keywords', fontsize=12, fontweight='bold')
ax_imp.invert_yaxis()
st.pyplot(fig_imp, use_container_width=True)

# 頁尾
st.divider()
col_footer1, col_footer2 = st.columns([3, 1])
with col_footer1:
    st.write("**Model**: Logistic Regression | **Algorithm**: TF-IDF Vectorization | **Dataset**: SMS Spam Collection")
with col_footer2:
    st.write("✅ App Status: Online")
