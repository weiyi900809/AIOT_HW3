import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve, classification_report
)
import pickle
import os
from pathlib import Path

class PhishingDetector:
    """
    釣魚郵件檢測模型類別
    包含資料載入、前處理、訓練、預測等功能
    """
    
    def __init__(self, model_path='models/phishing_model.pkl'):
        """初始化模型"""
        self.model = None
        self.scaler = None
        self.model_path = model_path
        self.metrics = {}
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        # 建立模型資料夾
        Path(os.path.dirname(model_path)).mkdir(parents=True, exist_ok=True)
    
    def load_data(self, filepath):
        """
        載入資料集
        
        Args:
            filepath: CSV 檔案路徑
        
        Returns:
            tuple: (特徵, 標籤)
        """
        print("📂 載入資料集...")
        data = np.genfromtxt(filepath, delimiter=',', dtype=np.int32)
        samples = data[:, :-1]
        targets = data[:, -1]
        
        print(f"✓ 資料集大小: {samples.shape[0]} 個樣本, {samples.shape[1]} 個特徵")
        print(f"✓ 標籤分布: {np.unique(targets, return_counts=True)}")
        
        return samples, targets
    
    def check_data_quality(self, X, y):
        """
        檢查資料品質
        
        Args:
            X: 特徵
            y: 標籤
        """
        print("\\n🔍 資料品質檢查...")
        
        # 檢查缺失值
        missing_features = np.isnan(X).sum()
        print(f"✓ 缺失值: {missing_features}")
        
        # 檢查異常值（超出 [-1, 1] 範圍）
        invalid_values = np.sum((X < -1) | (X > 1))
        print(f"✓ 異常值: {invalid_values}")
        
        # 檢查標籤分布
        unique_labels, counts = np.unique(y, return_counts=True)
        print(f"✓ 標籤分布:")
        for label, count in zip(unique_labels, counts):
            print(f"  - 標籤 {label}: {count} ({count/len(y)*100:.2f}%)")
        
        # 檢查類別不平衡
        min_class = counts.min()
        max_class = counts.max()
        imbalance_ratio = max_class / min_class
        print(f"✓ 類別不平衡比例: {imbalance_ratio:.2f}:1")
        
        return {
            'missing': missing_features,
            'invalid': invalid_values,
            'imbalance_ratio': imbalance_ratio
        }
    
    def preprocess_data(self, X, y, test_size=0.2, random_state=42):
        """
        前處理資料
        
        Args:
            X: 特徵
            y: 標籤
            test_size: 測試集比例
            random_state: 隨機種子
        """
        print("\\n⚙️  資料前處理...")
        
        # 1. 資料分割（使用分層抽樣保持標籤分布）
        print("  1️⃣  分割訓練/測試集...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, 
            stratify=y  # 保持標籤比例
        )
        print(f"     ✓ 訓練集: {X_train.shape[0]} 個樣本")
        print(f"     ✓ 測試集: {X_test.shape[0]} 個樣本")
        
        # 2. 特徵縮放
        print("  2️⃣  特徵標準化...")
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        print(f"     ✓ 訓練集 - 均值: {X_train_scaled.mean():.4f}, 標準差: {X_train_scaled.std():.4f}")
        print(f"     ✓ 測試集 - 均值: {X_test_scaled.mean():.4f}, 標準差: {X_test_scaled.std():.4f}")
        
        # 3. 檢測異常值
        print("  3️⃣  異常值檢測...")
        outliers = np.sum(np.abs(X_train_scaled) > 3)  # 3-sigma rule
        print(f"     ✓ 發現 {outliers} 個潛在異常值")
        
        self.X_train = X_train_scaled
        self.X_test = X_test_scaled
        self.y_train = y_train
        self.y_test = y_test
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train(self, X_train, y_train, cv=5):
        \"\"\"
        訓練模型
        
        Args:
            X_train: 訓練特徵
            y_train: 訓練標籤
            cv: 交叉驗證摺數
        \"\"\"
        print("\\n🚀 模型訓練...")
        
        # 建立模型
        self.model = LogisticRegression(
            max_iter=1000,
            random_state=42,
            class_weight='balanced'  # 處理類別不平衡
        )
        
        # 訓練
        print("  訓練中...")
        self.model.fit(X_train, y_train)
        
        # 交叉驗證
        print(f"  進行 {cv} 折交叉驗證...")
        cv_scores = cross_val_score(
            self.model, X_train, y_train,
            cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=42),
            scoring='f1'
        )
        
        print(f"✓ 訓練完成")
        print(f"  交叉驗證 F1 分數: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        self.metrics['cv_scores'] = cv_scores
    
    def evaluate(self, X_test, y_test):
        \"\"\"
        評估模型性能
        
        Args:
            X_test: 測試特徵
            y_test: 測試標籤
        
        Returns:
            dict: 評估指標
        \"\"\"
        print("\\n📊 模型評估...")
        
        # 預測
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # 計算指標
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        # 混淆矩陣
        cm = confusion_matrix(y_test, y_pred)
        
        # 詳細報告
        report = classification_report(y_test, y_pred)
        
        # 儲存指標
        self.metrics['accuracy'] = accuracy
        self.metrics['precision'] = precision
        self.metrics['recall'] = recall
        self.metrics['f1'] = f1
        self.metrics['roc_auc'] = roc_auc
        self.metrics['confusion_matrix'] = cm
        self.metrics['y_test'] = y_test
        self.metrics['y_pred'] = y_pred
        self.metrics['y_pred_proba'] = y_pred_proba
        
        print(f"✓ 準確度 (Accuracy):   {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"✓ 精度 (Precision):   {precision:.4f}")
        print(f"✓ 召回率 (Recall):    {recall:.4f}")
        print(f"✓ F1 分數:            {f1:.4f}")
        print(f"✓ ROC-AUC 分數:       {roc_auc:.4f}")
        print(f"\\n詳細分類報告:\\n{report}")
        
        return self.metrics
    
    def predict(self, X):
        \"\"\"
        進行預測
        
        Args:
            X: 輸入特徵
        
        Returns:
            tuple: (預測標籤, 預測機率)
        \"\"\"
        if self.model is None:
            raise ValueError("模型尚未訓練，請先訓練模型")
        
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)
        
        return predictions, probabilities
    
    def save_model(self):
        \"\"\"保存模型\"\"\"
        if self.model is None:
            raise ValueError("沒有模型可以保存")
        
        with open(self.model_path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler': self.scaler,
                'metrics': self.metrics
            }, f)
        
        print(f"\\n💾 模型已保存至: {self.model_path}")
    
    def load_model(self):
        \"\"\"載入已保存的模型\"\"\"
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"模型檔案不存在: {self.model_path}")
        
        with open(self.model_path, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.scaler = data['scaler']
            self.metrics = data['metrics']
        
        print(f"✓ 模型已載入")
