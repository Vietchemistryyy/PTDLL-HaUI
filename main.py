"""
COMPLETE PIPELINE - Chạy toàn bộ project từ đầu đến cuối
File: run_complete_pipeline.py

Thứ tự:
1. Load raw data
2. Preprocessing
3. Feature Engineering
4. Train với K-fold CV
5. Evaluate & Save
"""

import sys
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import các modules
from src.utils import SparkManager
from src.data_preprocessing import CardioDataLoader, CardioDataPreprocessor
from src.feature_engineering import CardioFeatureEngineer
from src.model import CardioLogisticModel
import config


def main():
    """Pipeline đầy đủ từ A-Z"""
    
    print("\n" + "=" * 80)
    print("🚀 CARDIOVASCULAR DISEASE PREDICTION - COMPLETE PIPELINE")
    print("=" * 80)
    
    # ========================================================================
    # BƯỚC 0: Khởi tạo Spark
    # ========================================================================
    print("\n" + "=" * 80)
    print("BƯỚC 0: KHỞI TẠO SPARK")
    print("=" * 80)
    
    spark = SparkManager.get_spark(
        app_name=config.SPARK_CONFIG['app_name'],
        master=config.SPARK_CONFIG['master'],
        driver_memory=config.SPARK_CONFIG['driver_memory'],
        executor_memory=config.SPARK_CONFIG['executor_memory']
    )
    
    # ========================================================================
    # BƯỚC 1: Load dữ liệu RAW
    # ========================================================================
    print("\n" + "=" * 80)
    print("BƯỚC 1: LOAD DỮ LIỆU RAW")
    print("=" * 80)
    
    loader = CardioDataLoader(spark)
    df_raw = loader.load_data()
    
    logger.info(f"✓ Đã load {df_raw.count():,} records")
    logger.info(f"✓ Số cột: {len(df_raw.columns)}")
    
    # Xem sample
    print("\n📊 Sample dữ liệu (5 dòng đầu):")
    df_raw.show(5, truncate=False)
    
    # ========================================================================
    # BƯỚC 2: PREPROCESSING
    # ========================================================================
    print("\n" + "=" * 80)
    print("BƯỚC 2: TIỀN XỬ LÝ DỮ LIỆU")
    print("=" * 80)
    
    preprocessor = CardioDataPreprocessor(spark)
    df_processed = preprocessor.preprocess_pipeline(df_raw)
    
    logger.info(f"✓ Dữ liệu sau preprocessing: {df_processed.count():,} records")
    
    # Xem các cột mới
    new_cols = [col for col in df_processed.columns if col not in df_raw.columns]
    logger.info(f"✓ Các cột mới được tạo: {', '.join(new_cols)}")
    
    # ========================================================================
    # BƯỚC 3: FEATURE ENGINEERING
    # ========================================================================
    print("\n" + "=" * 80)
    print("BƯỚC 3: FEATURE ENGINEERING")
    print("=" * 80)
    
    feature_engineer = CardioFeatureEngineer(spark)
    
    # Chuẩn bị features cho ML
    df_ml = feature_engineer.prepare_features_for_ml(df_processed)
    
    # Tạo interaction features
    df_ml = feature_engineer.create_interaction_features(df_ml)
    
    # Tạo binned features
    df_ml = feature_engineer.create_binned_features(df_ml)
    
    logger.info(f"✓ Đã tạo engineered features")
    
    # ========================================================================
    # BƯỚC 4: PREPARE FEATURES VECTOR
    # ========================================================================
    print("\n" + "=" * 80)
    print("BƯỚC 4: PREPARE FEATURES VECTOR")
    print("=" * 80)
    
    # Chọn các features để train
    feature_cols = [
        # Base features
        'age_years', 'gender', 'height', 'weight',
        'ap_hi', 'ap_lo', 'cholesterol', 'gluc',
        'smoke', 'alco', 'active', 'bmi', 'pulse_pressure',
        # Engineered features
        'bmi_age_interaction', 'bp_index', 
        'lifestyle_risk_score', 'health_score',
        'height_weight_ratio', 'age_bin', 'bmi_bin', 'bp_bin'
    ]
    
    logger.info(f"📊 Tổng số features: {len(feature_cols)}")
    logger.info(f"   Features: {', '.join(feature_cols)}")
    
    # Assemble features thành vector
    from pyspark.ml.feature import VectorAssembler
    
    assembler = VectorAssembler(
        inputCols=feature_cols,
        outputCol="features",
        handleInvalid="skip"
    )
    
    df_with_features = assembler.transform(df_ml)
    
    # Select chỉ features và target
    df_final = df_with_features.select("features", "cardio")
    
    logger.info(f"✓ Đã tạo feature vector")
    logger.info(f"✓ Dữ liệu cuối: {df_final.count():,} records")
    
    # Cache để tăng tốc
    df_final.cache()
    
    # ========================================================================
    # BƯỚC 5: TRAIN VỚI K-FOLD CROSS VALIDATION
    # ========================================================================
    print("\n" + "=" * 80)
    print("BƯỚC 5: TRAINING VỚI K-FOLD CROSS VALIDATION")
    print("=" * 80)
    
    # Tạo model instance
    cardio_model = CardioLogisticModel(spark)
    
    # Định nghĩa param grid để test
    param_grid = {
        'maxIter': [50, 100, 150],
        'regParam': [0.001, 0.01, 0.1],
        'elasticNetParam': [0.0, 0.5, 1.0]
    }
    
    # Chạy pipeline đầy đủ: Split → K-fold CV → Evaluate
    results = cardio_model.train_with_cv_pipeline(
        df=df_final,
        train_ratio=0.8,
        param_grid=param_grid,
        num_folds=5,
        seed=42
    )
    
    # ========================================================================
    # BƯỚC 6: PHÂN TÍCH KẾT QUẢ
    # ========================================================================
    print("\n" + "=" * 80)
    print("BƯỚC 6: PHÂN TÍCH KẾT QUẢ CHI TIẾT")
    print("=" * 80)
    
    # Best parameters
    print("\n🎯 BEST HYPERPARAMETERS:")
    for param, value in results['best_params'].items():
        print(f"  {param}: {value}")
    
    # CV metrics
    print("\n📊 CROSS VALIDATION METRICS:")
    print(f"  Best CV AUC: {results['cv_metrics']['best_auc']:.4f}")
    print(f"  Num folds:   {results['cv_metrics']['num_folds']}")
    
    # Test metrics
    print("\n📈 TEST SET METRICS (Unseen Data):")
    test_metrics = results['test_metrics']
    print(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall:    {test_metrics['recall']:.4f}")
    print(f"  F1-Score:  {test_metrics['f1_score']:.4f}")
    print(f"  AUC-ROC:   {test_metrics['auc_roc']:.4f}")
    
    # Confusion Matrix
    cm = test_metrics['confusion_matrix']
    print("\n📋 CONFUSION MATRIX:")
    print(f"              Predicted")
    print(f"              No    Yes")
    print(f"  Actual No   {cm['TN']:>5} {cm['FP']:>5}")
    print(f"  Actual Yes  {cm['FN']:>5} {cm['TP']:>5}")
    
    # Calculate additional metrics
    sensitivity = cm['TP'] / (cm['TP'] + cm['FN']) if (cm['TP'] + cm['FN']) > 0 else 0
    specificity = cm['TN'] / (cm['TN'] + cm['FP']) if (cm['TN'] + cm['FP']) > 0 else 0
    
    print(f"\n  Sensitivity (Recall):  {sensitivity:.4f}")
    print(f"  Specificity:           {specificity:.4f}")
    
    # ========================================================================
    # BƯỚC 7: FEATURE IMPORTANCE
    # ========================================================================
    print("\n" + "=" * 80)
    print("BƯỚC 7: FEATURE IMPORTANCE")
    print("=" * 80)
    
    importance_df = cardio_model.get_feature_importance(feature_cols)
    
    print("\n🔝 TOP 10 IMPORTANT FEATURES:")
    print(importance_df.head(10).to_string(index=False))
    
    # ========================================================================
    # BƯỚC 8: SAVE MODEL
    # ========================================================================
    print("\n" + "=" * 80)
    print("BƯỚC 8: LƯU MODEL")
    print("=" * 80)
    
    cardio_model.save_model()
    logger.info(f"✓ Model đã được lưu tại: {config.MODEL_PATH}")
    
    # ========================================================================
    # BƯỚC 9: TEST PREDICTION
    # ========================================================================
    print("\n" + "=" * 80)
    print("BƯỚC 9: TEST PREDICTION VỚI MỘT SAMPLE")
    print("=" * 80)
    
    # Lấy một sample từ test set
    sample = results['test_df'].limit(1).select("features", "cardio").collect()[0]
    sample_features = sample["features"].toArray().tolist()
    actual_label = sample["cardio"]
    
    # Predict
    prediction_result = cardio_model.predict_single(sample_features)
    
    print(f"\n📝 Sample Prediction:")
    print(f"  Actual label:     {actual_label} ({'Có bệnh' if actual_label == 1 else 'Không bệnh'})")
    print(f"  Predicted:        {prediction_result['prediction']} ({prediction_result['prediction_label']})")
    print(f"  Probability (No): {prediction_result['probability_negative']:.4f}")
    print(f"  Probability (Yes): {prediction_result['probability_positive']:.4f}")
    
    # ========================================================================
    # KẾT THÚC
    # ========================================================================
    print("\n" + "=" * 80)
    print("✅ HOÀN THÀNH TOÀN BỘ PIPELINE!")
    print("=" * 80)
    
    print("\n📁 CÁC FILE ĐÃ TẠO:")
    print(f"  - Processed data: {config.PROCESSED_DATA_FILE}")
    print(f"  - Model:          {config.MODEL_PATH}")
    
    print("\n🎉 Pipeline hoàn thành thành công!")
    print("=" * 80 + "\n")
    
    # Cleanup
    df_final.unpersist()
    
    return results


if __name__ == "__main__":
    try:
        results = main()
        
        # Optional: Save results to file
        import json
        import numpy as np
        
        # Convert results to JSON-serializable format
        results_summary = {
            'best_params': results['best_params'],
            'cv_metrics': {
                'best_auc': float(results['cv_metrics']['best_auc']),
                'num_folds': results['cv_metrics']['num_folds']
            },
            'test_metrics': {
                k: float(v) if isinstance(v, (int, float, np.number)) else v 
                for k, v in results['test_metrics'].items()
                if k != 'confusion_matrix'
            },
            'confusion_matrix': results['test_metrics']['confusion_matrix']
        }
        
        # Save to file
        with open('results/pipeline_results.json', 'w') as f:
            json.dump(results_summary, f, indent=2)
        
        print("✓ Kết quả đã được lưu vào: results/pipeline_results.json")
        
    except Exception as e:
        logger.error(f"❌ Lỗi khi chạy pipeline: {str(e)}", exc_info=True)
        sys.exit(1)
    
    finally:
        # Stop Spark
        SparkManager.stop()