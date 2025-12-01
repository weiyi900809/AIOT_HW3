import argparse
import sys
from phishing_model import PhishingDetector
from visualization import Visualizer
import numpy as np

def main():
    parser = argparse.ArgumentParser(
        description='釣魚郵件檢測系統 - CLI 介面',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
範例:
  # 訓練模型
  python cli.py --train

  # 進行預測
  python cli.py --predict --input test.csv

  # 顯示完整報告
  python cli.py --report

  # 僅預測（使用已訓練模型）
  python cli.py --predict-only --input test.csv
        '''
    )
    
    parser.add_argument('--train', action='store_true', 
                       help='訓練新模型')
    parser.add_argument('--predict', action='store_true',
                       help='訓練並預測')
    parser.add_argument('--predict-only', action='store_true',
                       help='使用已訓練模型進行預測')
    parser.add_argument('--input', type=str,
                       help='輸入資料檔案路徑')
    parser.add_argument('--report', action='store_true',
                       help='顯示完整評估報告')
    parser.add_argument('--data', type=str, default='phishing_dataset.csv',
                       help='訓練資料集路徑 (預設: phishing_dataset.csv)')
    
    args = parser.parse_args()
    
    # 初始化
    detector = PhishingDetector()
    visualizer = Visualizer()
    
    if args.train:
        print("=" * 60)
        print("🔧 釣魚郵件檢測系統 - 訓練模式")
        print("=" * 60)
        
        # 載入資料
        X, y = detector.load_data(args.data)
        
        # 資料品質檢查
        detector.check_data_quality(X, y)
        
        # 前處理
        X_train, X_test, y_train, y_test = detector.preprocess_data(X, y)
        
        # 訓練
        detector.train(X_train, y_train)
        
        # 評估
        metrics = detector.evaluate(X_test, y_test)
        
        # 視覺化
        print("\\n📊 生成視覺化圖表...")
        visualizer.plot_data_distribution(X, y)
        visualizer.plot_confusion_matrix(y_test, metrics['y_pred'])
        visualizer.plot_roc_curve(y_test, metrics['y_pred_proba'])
        visualizer.plot_model_metrics(metrics)
        visualizer.plot_feature_importance(detector.model.coef_)
        
        # 保存模型
        detector.save_model()
        
        print("\\n✅ 訓練完成！")
    
    elif args.predict:
        if not args.input:
            print("❌ 錯誤: 預測模式需要指定 --input 檔案")
            sys.exit(1)
        
        print("=" * 60)
        print("🔮 釣魚郵件檢測系統 - 訓練並預測模式")
        print("=" * 60)
        
        # 載入資料
        X, y = detector.load_data(args.data)
        detector.check_data_quality(X, y)
        
        # 前處理
        X_train, X_test, y_train, y_test = detector.preprocess_data(X, y)
        
        # 訓練
        detector.train(X_train, y_train)
        metrics = detector.evaluate(X_test, y_test)
        
        # 載入預測資料
        print(f"\\n📂 載入預測資料: {args.input}")
        try:
            test_data = np.genfromtxt(args.input, delimiter=',', dtype=np.int32)
            if test_data.ndim == 1:
                test_data = test_data.reshape(1, -1)
            
            # 預測
            predictions, probabilities = detector.predict(test_data)
            
            print(f"\\n📊 預測結果:")
            print("-" * 60)
            for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
                label = "釣魚郵件 ⚠️" if pred == 1 else "合法郵件 ✓"
                confidence = max(prob) * 100
                print(f"樣本 {i+1}: {label} (信心度: {confidence:.2f}%)")
            
            # 保存模型
            detector.save_model()
            
        except Exception as e:
            print(f"❌ 預測失敗: {e}")
            sys.exit(1)
    
    elif args.predict_only:
        if not args.input:
            print("❌ 錯誤: 需要指定 --input 檔案")
            sys.exit(1)
        
        print("=" * 60)
        print("🔮 釣魚郵件檢測系統 - 預測模式")
        print("=" * 60)
        
        try:
            # 載入已訓練模型
            detector.load_model()
            
            # 載入預測資料
            print(f"📂 載入預測資料: {args.input}")
            test_data = np.genfromtxt(args.input, delimiter=',', dtype=np.int32)
            if test_data.ndim == 1:
                test_data = test_data.reshape(1, -1)
            
            # 預測
            predictions, probabilities = detector.predict(test_data)
            
            print(f"\\n📊 預測結果:")
            print("-" * 60)
            for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
                label = "釣魚郵件 ⚠️" if pred == 1 else "合法郵件 ✓"
                confidence = max(prob) * 100
                print(f"樣本 {i+1}: {label} (信心度: {confidence:.2f}%)")
                
        except Exception as e:
            print(f"❌ 預測失敗: {e}")
            sys.exit(1)
    
    elif args.report:
        print("=" * 60)
        print("📋 釣魚郵件檢測系統 - 完整報告")
        print("=" * 60)
        
        try:
            # 載入資料
            X, y = detector.load_data(args.data)
            
            # 資料品質檢查
            quality = detector.check_data_quality(X, y)
            
            # 前處理
            X_train, X_test, y_train, y_test = detector.preprocess_data(X, y)
            
            # 訓練
            detector.train(X_train, y_train)
            
            # 評估
            metrics = detector.evaluate(X_test, y_test)
            
            # 顯示統計信息
            print("\\n" + "=" * 60)
            print("📈 詳細統計信息")
            print("=" * 60)
            print(f"\\n資料集統計:")
            print(f"  - 訓練集大小: {X_train.shape[0]}")
            print(f"  - 測試集大小: {X_test.shape[0]}")
            print(f"  - 特徵數量: {X_train.shape[1]}")
            print(f"  - 類別不平衡比例: {quality['imbalance_ratio']:.2f}:1")
            
            print(f"\\n模型性能:")
            print(f"  - 準確度: {metrics['accuracy']:.4f}")
            print(f"  - 精度: {metrics['precision']:.4f}")
            print(f"  - 召回率: {metrics['recall']:.4f}")
            print(f"  - F1 分數: {metrics['f1']:.4f}")
            print(f"  - ROC-AUC: {metrics['roc_auc']:.4f}")
            
            if 'cv_scores' in metrics:
                cv_scores = metrics['cv_scores']
                print(f"\\n交叉驗證結果:")
                print(f"  - 均值: {cv_scores.mean():.4f}")
                print(f"  - 標準差: {cv_scores.std():.4f}")
                print(f"  - 最小值: {cv_scores.min():.4f}")
                print(f"  - 最大值: {cv_scores.max():.4f}")
            
            # 保存模型
            detector.save_model()
            
        except Exception as e:
            print(f"❌ 報告生成失敗: {e}")
            sys.exit(1)
    
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
