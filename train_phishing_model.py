#!/usr/bin/env python3
"""
Logistic Regression Phishing Detector - 訓練模組
包含完整的前處理步驟、資料探索、特徵分析和模型評估
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report, roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
import warnings
import joblib
warnings.simplefilter('ignore')

# 設定圖表風格
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

class PhishingDetector:
    """釣魚網站偵測類"""
    
    def __init__(self, data_path='./phishing_dataset.csv'):
        self.data_path = data_path
        self.model = None
        self.scaler = None
        self.metrics = {}
        self.feature_names = None
        
    def load_data(self):
        """載入資料集"""
        print("=" * 70)
        print("📊 STEP 1: 資料載入")
        print("=" * 70)
        
        self.dataset = np.genfromtxt(self.data_path, delimiter=',', dtype=np.int32)
        self.samples = self.dataset[:, :-1]
        self.targets = self.dataset[:, -1]
        
        print(f"✓ 資料集大小: {self.dataset.shape}")
        print(f"✓ 樣本數: {self.samples.shape[0]}")
        print(f"✓ 特徵數: {self.samples.shape[1]}")
        print(f"✓ 類別分佈:")
        unique, counts = np.unique(self.targets, return_counts=True)
        for label, count in zip(unique, counts):
            percentage = (count / len(self.targets)) * 100
            print(f"  - 類別 {label}: {count} 筆 ({percentage:.2f}%)")
    
    def explore_data(self):
        """資料探索與統計分析"""
        print("\n" + "=" * 70)
        print("🔍 STEP 2: 資料探索與統計分析")
        print("=" * 70)
        
        print(f"\n✓ 特徵統計:")
        print(f"  - 最小值: {self.samples.min()}")
        print(f"  - 最大值: {self.samples.max()}")
        print(f"  - 平均值: {self.samples.mean():.4f}")
        print(f"  - 標準差: {self.samples.std():.4f}")
        print(f"  - 缺失值: {np.isnan(self.samples).sum()}")
        
        # 繪製特徵分佈
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        fig.suptitle('前 6 個特徵的分佈', fontsize=14, fontweight='bold')
        
        for idx in range(min(6, self.samples.shape[1])):
            ax = axes[idx // 3, idx % 3]
            ax.hist(self.samples[:, idx], bins=30, edgecolor='black', alpha=0.7, color='steelblue')
            ax.set_title(f'特徵 {idx}')
            ax.set_xlabel('值')
            ax.set_ylabel('頻率')
        
        plt.tight_layout()
        plt.savefig('01_feature_distribution.png', dpi=300, bbox_inches='tight')
        print(f"\n✓ 已保存: 01_feature_distribution.png")
        plt.close()
    
    def preprocess_data(self):
        """資料前處理"""
        print("\n" + "=" * 70)
        print("🔧 STEP 3: 資料前處理")
        print("=" * 70)
        
        # 檢查異常值
        print(f"\n✓ 異常值檢測 (使用 IQR 方法):")
        Q1 = np.percentile(self.samples, 25, axis=0)
        Q3 = np.percentile(self.samples, 75, axis=0)
        IQR = Q3 - Q1
        outliers_mask = ((self.samples < (Q1 - 1.5 * IQR)) | (self.samples > (Q3 + 1.5 * IQR)))
        outlier_count = outliers_mask.sum()
        print(f"  - 異常值數量: {outlier_count} ({(outlier_count/(self.samples.size))*100:.2f}%)")
        
        # 標準化特徵
        print(f"\n✓ 特徵標準化 (StandardScaler)...")
        self.scaler = StandardScaler()
        self.samples_scaled = self.scaler.fit_transform(self.samples)
        print(f"  - 標準化後平均值: {self.samples_scaled.mean():.6f}")
        print(f"  - 標準化後標準差: {self.samples_scaled.std():.6f}")
    
    def split_data(self, test_size=0.2, random_state=42):
        """分割訓練/測試集"""
        print("\n" + "=" * 70)
        print("✂️  STEP 4: 資料分割")
        print("=" * 70)
        
        self.train_samples, self.test_samples, self.train_targets, self.test_targets = train_test_split(
            self.samples_scaled, self.targets, test_size=test_size, random_state=random_state, stratify=self.targets
        )
        
        print(f"\n✓ 訓練集: {self.train_samples.shape[0]} 筆 ({(1-test_size)*100:.0f}%)")
        print(f"✓ 測試集: {self.test_samples.shape[0]} 筆 ({test_size*100:.0f}%)")
        print(f"\n✓ 訓練集類別分佈:")
        train_unique, train_counts = np.unique(self.train_targets, return_counts=True)
        for label, count in zip(train_unique, train_counts):
            print(f"  - 類別 {label}: {count} 筆")
    
    def train_model(self):
        """模型訓練"""
        print("\n" + "=" * 70)
        print("🤖 STEP 5: 模型訓練")
        print("=" * 70)
        
        print(f"\n✓ 建立 Logistic Regression 模型...")
        self.model = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
        self.model.fit(self.train_samples, self.train_targets)
        
        print(f"✓ 模型訓練完成")
        print(f"  - 模型參數: {self.model.get_params()}")
    
    def evaluate_model(self):
        """模型評估"""
        print("\n" + "=" * 70)
        print("📈 STEP 6: 模型評估")
        print("=" * 70)
        
        # 訓練集預測
        train_pred = self.model.predict(self.train_samples)
        train_pred_proba = self.model.predict_proba(self.train_samples)[:, 1]
        
        # 測試集預測
        test_pred = self.model.predict(self.test_samples)
        test_pred_proba = self.model.predict_proba(self.test_samples)[:, 1]
        
        # 計算指標
        print(f"\n✓ 訓練集性能:")
        train_accuracy = accuracy_score(self.train_targets, train_pred)
        train_precision = precision_score(self.train_targets, train_pred)
        train_recall = recall_score(self.train_targets, train_pred)
        train_f1 = f1_score(self.train_targets, train_pred)
        train_auc = roc_auc_score(self.train_targets, train_pred_proba)
        
        print(f"  - 準確率 (Accuracy): {train_accuracy:.4f}")
        print(f"  - 精準率 (Precision): {train_precision:.4f}")
        print(f"  - 召回率 (Recall): {train_recall:.4f}")
        print(f"  - F1 分數: {train_f1:.4f}")
        print(f"  - AUC 分數: {train_auc:.4f}")
        
        print(f"\n✓ 測試集性能:")
        test_accuracy = accuracy_score(self.test_targets, test_pred)
        test_precision = precision_score(self.test_targets, test_pred)
        test_recall = recall_score(self.test_targets, test_pred)
        test_f1 = f1_score(self.test_targets, test_pred)
        test_auc = roc_auc_score(self.test_targets, test_pred_proba)
        
        print(f"  - 準確率 (Accuracy): {test_accuracy:.4f}")
        print(f"  - 精準率 (Precision): {test_precision:.4f}")
        print(f"  - 召回率 (Recall): {test_recall:.4f}")
        print(f"  - F1 分數: {test_f1:.4f}")
        print(f"  - AUC 分數: {test_auc:.4f}")
        
        # 交叉驗證
        print(f"\n✓ 5 折交叉驗證:")
        cv_scores = cross_val_score(self.model, self.train_samples, self.train_targets, cv=5, scoring='accuracy')
        print(f"  - 平均 CV 準確率: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        # 儲存指標
        self.metrics = {
            'train_accuracy': train_accuracy, 'train_precision': train_precision,
            'train_recall': train_recall, 'train_f1': train_f1, 'train_auc': train_auc,
            'test_accuracy': test_accuracy, 'test_precision': test_precision,
            'test_recall': test_recall, 'test_f1': test_f1, 'test_auc': test_auc,
            'cv_mean': cv_scores.mean(), 'cv_std': cv_scores.std()
        }
        
        # 混淆矩陣
        cm = confusion_matrix(self.test_targets, test_pred)
        print(f"\n✓ 混淆矩陣:")
        print(f"  {cm}")
        
        # 視覺化混淆矩陣與 ROC 曲線
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # 混淆矩陣
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0])
        axes[0].set_title('混淆矩陣 (測試集)', fontweight='bold')
        axes[0].set_xlabel('預測標籤')
        axes[0].set_ylabel('真實標籤')
        
        # ROC 曲線
        fpr, tpr, _ = roc_curve(self.test_targets, test_pred_proba)
        axes[1].plot(fpr, tpr, label=f'AUC = {test_auc:.4f}', linewidth=2, color='steelblue')
        axes[1].plot([0, 1], [0, 1], 'k--', linewidth=1)
        axes[1].set_xlabel('False Positive Rate')
        axes[1].set_ylabel('True Positive Rate')
        axes[1].set_title('ROC 曲線 (測試集)', fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('02_confusion_matrix_roc.png', dpi=300, bbox_inches='tight')
        print(f"\n✓ 已保存: 02_confusion_matrix_roc.png")
        plt.close()
    
    def save_model(self):
        """保存模型和結果"""
        print("\n" + "=" * 70)
        print("💾 STEP 7: 模型保存")
        print("=" * 70)
        
        joblib.dump(self.model, 'phishing_logistic_model.pkl')
        joblib.dump(self.scaler, 'phishing_scaler.pkl')
        joblib.dump(self.metrics, 'phishing_metrics.pkl')
        
        print(f"\n✓ 已保存:")
        print(f"  - phishing_logistic_model.pkl")
        print(f"  - phishing_scaler.pkl")
        print(f"  - phishing_metrics.pkl")
    
    def run_pipeline(self):
        """執行完整 Pipeline"""
        print("\n" + "🎯 釣魚網站偵測 - Logistic Regression Pipeline\n")
        
        self.load_data()
        self.explore_data()
        self.preprocess_data()
        self.split_data()
        self.train_model()
        self.evaluate_model()
        self.save_model()
        
        print("\n" + "=" * 70)
        print("✅ 訓練完成！")
        print("=" * 70 + "\n")

if __name__ == "__main__":
    detector = PhishingDetector()
    detector.run_pipeline()
