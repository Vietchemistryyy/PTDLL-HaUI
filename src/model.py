"""
Module Training và Evaluation cho Model Logistic Regression
Sử dụng PySpark MLlib với K-fold Cross Validation Pipeline
"""
from pyspark.sql import SparkSession, DataFrame
from pyspark.ml.classification import LogisticRegression, LogisticRegressionModel
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.sql import functions as F
import pandas as pd
import logging
from pathlib import Path
import config

logger = logging.getLogger(__name__)


class CardioLogisticModel:
    """Class training Logistic Regression model cho dự đoán bệnh tim mạch"""
    
    def __init__(self, spark: SparkSession):
        self.spark = spark
        self.model = None
        self.pipeline_model = None
        self.feature_names = None
        self.train_df = None
        self.test_df = None
        self.best_params = None
        
    def create_logistic_model(self, 
                             max_iter: int = 100,
                             reg_param: float = 0.01,
                             elastic_net_param: float = 0.0) -> LogisticRegression:
        """
        Tạo Logistic Regression model
        
        Args:
            max_iter: Số iteration tối đa
            reg_param: Regularization parameter
            elastic_net_param: ElasticNet mixing parameter (0=L2, 1=L1)
            
        Returns:
            LogisticRegression model
        """
        logger.info("Tạo Logistic Regression model...")
        logger.info(f"  - Max iterations: {max_iter}")
        logger.info(f"  - Regularization: {reg_param}")
        logger.info(f"  - ElasticNet: {elastic_net_param}")
        
        lr = LogisticRegression(
            featuresCol="features",
            labelCol="cardio",
            maxIter=max_iter,
            regParam=reg_param,
            elasticNetParam=elastic_net_param,
            family="binomial"
        )
        
        return lr
    
    def split_data(self, df: DataFrame, train_ratio: float = 0.8, seed: int = 42):
        """
        BƯỚC 1: Chia dữ liệu Train/Test
        
        Args:
            df: DataFrame với features và label
            train_ratio: Tỷ lệ train (default 0.8)
            seed: Random seed
        """
        logger.info("=" * 70)
        logger.info("BƯỚC 1: CHIA DỮ LIỆU TRAIN/TEST")
        logger.info("=" * 70)
        
        self.train_df, self.test_df = df.randomSplit([train_ratio, 1 - train_ratio], seed=seed)
        
        train_count = self.train_df.count()
        test_count = self.test_df.count()
        total_count = train_count + test_count
        
        # Kiểm tra phân bố class trong train/test
        train_pos = self.train_df.filter("cardio = 1").count()
        train_neg = train_count - train_pos
        test_pos = self.test_df.filter("cardio = 1").count()
        test_neg = test_count - test_pos
        
        logger.info(f"\n📊 THỐNG KÊ PHÂN CHIA:")
        logger.info(f"  Total:     {total_count:,} samples")
        logger.info(f"  Train:     {train_count:,} samples ({train_count/total_count*100:.1f}%)")
        logger.info(f"  Test:      {test_count:,} samples ({test_count/total_count*100:.1f}%)")
        logger.info(f"\n  Train - Positive: {train_pos:,} ({train_pos/train_count*100:.1f}%)")
        logger.info(f"  Train - Negative: {train_neg:,} ({train_neg/train_count*100:.1f}%)")
        logger.info(f"  Test  - Positive: {test_pos:,} ({test_pos/test_count*100:.1f}%)")
        logger.info(f"  Test  - Negative: {test_neg:,} ({test_neg/test_count*100:.1f}%)")
        
        logger.info("\n✓ Test set được giữ nguyên - KHÔNG động đến cho đến evaluation cuối")
        logger.info("=" * 70)
        
        return self.train_df, self.test_df
    
    def train(self, 
             train_df: DataFrame,
             max_iter: int = 100,
             reg_param: float = 0.01,
             elastic_net_param: float = 0.0) -> LogisticRegressionModel:
        """
        Training model (training thủ công - không dùng CV)
        
        Args:
            train_df: Training DataFrame (phải có cột 'features' và 'cardio')
            max_iter: Số iteration tối đa
            reg_param: Regularization parameter
            elastic_net_param: ElasticNet mixing parameter
            
        Returns:
            Trained model
        """
        logger.info("=" * 60)
        logger.info("TRAINING MODEL (Manual - không dùng CV)")
        logger.info("=" * 60)
        
        # Tạo model
        lr = self.create_logistic_model(max_iter, reg_param, elastic_net_param)
        
        # Training
        logger.info(f"Training với {train_df.count():,} samples...")
        self.model = lr.fit(train_df)
        
        logger.info("✓ Training hoàn thành!")
        logger.info(f"✓ Số coefficients: {len(self.model.coefficients)}")
        
        return self.model
    
    def cross_validate(self, 
                      train_df: DataFrame,
                      param_grid: dict = None,
                      num_folds: int = 5) -> tuple:
        """
        BƯỚC 2: K-fold Cross Validation trên TRAIN set để tìm best hyperparameters
        
        Args:
            train_df: Training DataFrame (CHỈ train set, không bao gồm test)
            param_grid: Dictionary với các parameters cần test
            num_folds: Số folds cho cross validation
            
        Returns:
            Tuple (best_model, best_params, avg_metrics)
        """
        logger.info("=" * 70)
        logger.info(f"BƯỚC 2: K-FOLD CROSS VALIDATION ({num_folds} folds)")
        logger.info("=" * 70)
        logger.info("⚠️  CHÚ Ý: CV chỉ chạy trên TRAIN set, test set KHÔNG được động đến!")
        
        # Tạo base model
        lr = LogisticRegression(
            featuresCol="features",
            labelCol="cardio",
            family="binomial"
        )
        
        # Param grid
        if param_grid is None:
            param_grid = {
                'maxIter': [50, 100, 150],
                'regParam': [0.001, 0.01, 0.1],
                'elasticNetParam': [0.0, 0.5, 1.0]
            }
        
        logger.info(f"\n🔍 Testing parameters:")
        for key, values in param_grid.items():
            logger.info(f"  {key}: {values}")
        
        # Build param grid
        paramGridBuilder = ParamGridBuilder()
        for param_name, param_values in param_grid.items():
            if param_name == 'maxIter':
                paramGridBuilder = paramGridBuilder.addGrid(lr.maxIter, param_values)
            elif param_name == 'regParam':
                paramGridBuilder = paramGridBuilder.addGrid(lr.regParam, param_values)
            elif param_name == 'elasticNetParam':
                paramGridBuilder = paramGridBuilder.addGrid(lr.elasticNetParam, param_values)
        
        paramGrid = paramGridBuilder.build()
        
        logger.info(f"  → Tổng số combinations: {len(paramGrid)}")
        
        # Evaluator
        evaluator = BinaryClassificationEvaluator(
            labelCol="cardio",
            rawPredictionCol="rawPrediction",
            metricName="areaUnderROC"
        )
        
        # Cross Validator
        cv = CrossValidator(
            estimator=lr,
            estimatorParamMaps=paramGrid,
            evaluator=evaluator,
            numFolds=num_folds,
            parallelism=2,
            seed=42
        )
        
        logger.info(f"\n🚀 Running {num_folds}-fold cross validation...")
        logger.info(f"   Training {len(paramGrid)} models × {num_folds} folds = {len(paramGrid) * num_folds} total fits")
        
        cv_model = cv.fit(train_df)
        
        # Best model
        best_model = cv_model.bestModel
        
        # Best params
        best_params = {
            'maxIter': best_model.getMaxIter(),
            'regParam': best_model.getRegParam(),
            'elasticNetParam': best_model.getElasticNetParam()
        }
        
        # CV results
        avg_metrics = cv_model.avgMetrics
        best_auc = max(avg_metrics)
        
        logger.info("\n✓ Cross validation hoàn thành!")
        logger.info(f"\n📊 BEST PARAMETERS:")
        for key, value in best_params.items():
            logger.info(f"  {key}: {value}")
        logger.info(f"\n📈 BEST CV AUC: {best_auc:.4f}")
        logger.info(f"   (Trung bình trên {num_folds} folds)")
        
        logger.info("=" * 70)
        
        self.model = best_model
        self.best_params = best_params
        
        return best_model, best_params, avg_metrics
    
    def train_with_cv_pipeline(self, 
                               df: DataFrame,
                               train_ratio: float = 0.8,
                               param_grid: dict = None,
                               num_folds: int = 5,
                               seed: int = 42) -> dict:
        """
        PIPELINE ĐẦY ĐỦ: Split → K-fold CV → Evaluate
        
        Workflow đúng chuẩn:
        1. Split data thành Train/Test (80/20)
        2. K-fold CV trên TRAIN set → tìm best hyperparameters
        3. Model tốt nhất đã được train trên full train set trong CV
        4. Evaluate trên TEST set (unseen data)
        
        Args:
            df: DataFrame đầy đủ với features và label
            train_ratio: Tỷ lệ train/test (default 0.8)
            param_grid: Dict hyperparameters để test
            num_folds: Số folds cho CV (default 5)
            seed: Random seed
            
        Returns:
            Dict chứa: train_df, test_df, best_model, best_params, cv_metrics, test_metrics
        """
        logger.info("\n" + "=" * 70)
        logger.info("🚀 PIPELINE TRAINING ĐẦY ĐỦ VỚI K-FOLD CROSS VALIDATION")
        logger.info("=" * 70)
        
        # Bước 1: Split train/test
        train_df, test_df = self.split_data(df, train_ratio=train_ratio, seed=seed)
        
        # Bước 2: K-fold CV trên TRAIN set
        best_model, best_params, cv_results = self.cross_validate(
            train_df=train_df,
            param_grid=param_grid,
            num_folds=num_folds
        )
        
        # Bước 3: Evaluate trên TEST set (unseen)
        logger.info("\n" + "=" * 70)
        logger.info("BƯỚC 3: ĐÁNH GIÁ CUỐI CÙNG TRÊN TEST SET")
        logger.info("=" * 70)
        logger.info("📊 Đây là lần ĐẦU TIÊN model nhìn thấy test data!")
        
        test_metrics = self.evaluate(test_df)
        
        # Tổng kết
        logger.info("\n" + "=" * 70)
        logger.info("✅ HOÀN THÀNH PIPELINE")
        logger.info("=" * 70)
        logger.info(f"\n📋 SUMMARY:")
        logger.info(f"  Train samples: {train_df.count():,}")
        logger.info(f"  Test samples:  {test_df.count():,}")
        logger.info(f"  Best CV AUC:   {max(cv_results):.4f}")
        logger.info(f"  Test AUC:      {test_metrics['auc_roc']:.4f}")
        logger.info(f"  Test Accuracy: {test_metrics['accuracy']:.4f}")
        logger.info(f"  Test F1:       {test_metrics['f1_score']:.4f}")
        logger.info("=" * 70 + "\n")
        
        return {
            'train_df': train_df,
            'test_df': test_df,
            'best_model': best_model,
            'best_params': best_params,
            'cv_metrics': {
                'best_auc': max(cv_results),
                'all_auc_scores': cv_results,
                'num_folds': num_folds
            },
            'test_metrics': test_metrics
        }
    
    def predict(self, df: DataFrame) -> DataFrame:
        """
        Dự đoán trên DataFrame
        
        Args:
            df: DataFrame cần dự đoán (phải có cột 'features')
            
        Returns:
            DataFrame với cột prediction và probability
        """
        if self.model is None:
            raise ValueError("Model chưa được training! Hãy gọi train() hoặc train_with_cv_pipeline() trước.")
        
        logger.info("Đang dự đoán...")
        predictions = self.model.transform(df)
        
        return predictions
    
    def evaluate(self, test_df: DataFrame) -> dict:
        """
        Đánh giá model trên test set
        
        Args:
            test_df: Test DataFrame
            
        Returns:
            Dictionary chứa các metrics
        """
        logger.info("\n📊 ĐÁNH GIÁ MODEL")
        logger.info("-" * 70)
        
        # Dự đoán
        predictions = self.predict(test_df)
        
        # Binary Classification Evaluator
        binary_evaluator = BinaryClassificationEvaluator(
            labelCol="cardio",
            rawPredictionCol="rawPrediction",
            metricName="areaUnderROC"
        )
        
        auc = binary_evaluator.evaluate(predictions)
        
        # Multiclass Evaluator cho accuracy, precision, recall, f1
        mc_evaluator = MulticlassClassificationEvaluator(
            labelCol="cardio",
            predictionCol="prediction"
        )
        
        accuracy = mc_evaluator.evaluate(predictions, {mc_evaluator.metricName: "accuracy"})
        precision = mc_evaluator.evaluate(predictions, {mc_evaluator.metricName: "weightedPrecision"})
        recall = mc_evaluator.evaluate(predictions, {mc_evaluator.metricName: "weightedRecall"})
        f1 = mc_evaluator.evaluate(predictions, {mc_evaluator.metricName: "f1"})
        
        # Confusion Matrix
        confusion_matrix = self.get_confusion_matrix(predictions)
        
        metrics = {
            'accuracy': round(accuracy, 4),
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1_score': round(f1, 4),
            'auc_roc': round(auc, 4),
            'confusion_matrix': confusion_matrix
        }
        
        # Log metrics
        logger.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
        logger.info(f"  Precision: {metrics['precision']:.4f}")
        logger.info(f"  Recall:    {metrics['recall']:.4f}")
        logger.info(f"  F1-Score:  {metrics['f1_score']:.4f}")
        logger.info(f"  AUC-ROC:   {metrics['auc_roc']:.4f}")
        logger.info(f"\n  Confusion Matrix:")
        logger.info(f"    TN={confusion_matrix['TN']:,}  FP={confusion_matrix['FP']:,}")
        logger.info(f"    FN={confusion_matrix['FN']:,}  TP={confusion_matrix['TP']:,}")
        logger.info("-" * 70)
        
        return metrics
    
    def get_confusion_matrix(self, predictions: DataFrame) -> dict:
        """
        Tính confusion matrix
        
        Args:
            predictions: DataFrame với predictions
            
        Returns:
            Dictionary với TN, FP, FN, TP
        """
        tp = predictions.filter((F.col("prediction") == 1) & (F.col("cardio") == 1)).count()
        tn = predictions.filter((F.col("prediction") == 0) & (F.col("cardio") == 0)).count()
        fp = predictions.filter((F.col("prediction") == 1) & (F.col("cardio") == 0)).count()
        fn = predictions.filter((F.col("prediction") == 0) & (F.col("cardio") == 1)).count()
        
        return {
            'TP': tp,
            'TN': tn,
            'FP': fp,
            'FN': fn
        }
    
    def get_feature_importance(self, feature_names: list = None) -> pd.DataFrame:
        """
        Lấy feature importance từ coefficients
        
        Args:
            feature_names: List tên các features
            
        Returns:
            Pandas DataFrame với feature importance
        """
        if self.model is None:
            raise ValueError("Model chưa được training!")
        
        logger.info("Tính feature importance...")
        
        coefficients = self.model.coefficients.toArray()
        
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(coefficients))]
        
        # Tạo DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'coefficient': coefficients,
            'abs_coefficient': abs(coefficients)
        })
        
        # Sort by absolute coefficient
        importance_df = importance_df.sort_values('abs_coefficient', ascending=False)
        
        logger.info(f"✓ Top 5 features quan trọng nhất:")
        for idx, row in importance_df.head(5).iterrows():
            logger.info(f"  {row['feature']}: {row['coefficient']:.4f}")
        
        return importance_df
    
    def save_model(self, path: str = None):
        """
        Lưu model
        
        Args:
            path: Đường dẫn lưu model
        """
        if self.model is None:
            raise ValueError("Model chưa được training!")
        
        if path is None:
            path = str(config.MODEL_PATH)
        
        logger.info(f"Lưu model tại: {path}")
        
        # Tạo thư mục nếu chưa có
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save model
        self.model.write().overwrite().save(path)
        
        logger.info("✓ Đã lưu model")
    
    def load_model(self, path: str = None):
        """
        Load model đã lưu
        
        Args:
            path: Đường dẫn model
        """
        if path is None:
            path = str(config.MODEL_PATH)
        
        logger.info(f"Load model từ: {path}")
        
        self.model = LogisticRegressionModel.load(path)
        
        logger.info("✓ Đã load model")
        
        return self.model
    
    def get_prediction_probabilities(self, predictions: DataFrame) -> DataFrame:
        """
        Lấy probabilities của predictions
        
        Args:
            predictions: DataFrame với predictions
            
        Returns:
            DataFrame với probability cho mỗi class
        """
        from pyspark.ml.functions import vector_to_array
        
        predictions = predictions.withColumn(
            "prob_array",
            vector_to_array("probability")
        )
        
        predictions = predictions.withColumn(
            "prob_negative",
            F.col("prob_array")[0]
        ).withColumn(
            "prob_positive",
            F.col("prob_array")[1]
        )
        
        return predictions.select(
            "cardio",
            "prediction",
            "prob_negative",
            "prob_positive"
        )
    
    def predict_single(self, features: list) -> dict:
        """
        Dự đoán cho một sample đơn
        
        Args:
            features: List các giá trị features
            
        Returns:
            Dictionary với prediction và probability
        """
        if self.model is None:
            raise ValueError("Model chưa được training!")
        
        from pyspark.ml.linalg import Vectors
        
        # Tạo DataFrame từ features
        feature_vector = Vectors.dense(features)
        df = self.spark.createDataFrame(
            [(feature_vector,)],
            ["features"]
        )
        
        # Predict
        prediction = self.model.transform(df)
        
        # Extract results
        result = prediction.select("prediction", "probability").collect()[0]
        
        prob_array = result["probability"].toArray()
        
        return {
            'prediction': int(result["prediction"]),
            'probability_negative': float(prob_array[0]),
            'probability_positive': float(prob_array[1]),
            'prediction_label': 'Có nguy cơ bệnh tim' if result["prediction"] == 1 else 'Không có nguy cơ'
        }
    
    def get_model_summary(self) -> dict:
        """
        Lấy summary của model
        
        Returns:
            Dictionary chứa model summary
        """
        if self.model is None:
            raise ValueError("Model chưa được training!")
        
        summary = self.model.summary
        
        return {
            'total_iterations': summary.totalIterations,
            'objective_history': summary.objectiveHistory,
            'area_under_roc': summary.areaUnderROC,
            'features_count': len(self.model.coefficients)
        }


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_and_train_model(train_df: DataFrame, 
                          test_df: DataFrame,
                          max_iter: int = 100,
                          reg_param: float = 0.01,
                          save_model: bool = True) -> tuple:
    """
    Hàm tiện ích để tạo, train và evaluate model (Manual - không dùng CV)
    
    Args:
        train_df: Training DataFrame
        test_df: Test DataFrame
        max_iter: Max iterations
        reg_param: Regularization parameter
        save_model: Lưu model hay không
        
    Returns:
        Tuple (model, metrics)
    """
    from src.utils import SparkManager
    
    spark = SparkManager.get_spark()
    
    # Tạo model
    cardio_model = CardioLogisticModel(spark)
    
    # Train
    cardio_model.train(train_df, max_iter=max_iter, reg_param=reg_param)
    
    # Evaluate
    metrics = cardio_model.evaluate(test_df)
    
    # Save
    if save_model:
        cardio_model.save_model()
    
    return cardio_model, metrics


def train_with_cv(df: DataFrame,
                  param_grid: dict = None,
                  num_folds: int = 5,
                  train_ratio: float = 0.8,
                  save_model: bool = True) -> dict:
    """
    Hàm tiện ích để train với K-fold CV pipeline đầy đủ
    
    Args:
        df: DataFrame đầy đủ với features và label
        param_grid: Dict hyperparameters để test
        num_folds: Số folds cho CV
        train_ratio: Tỷ lệ train/test
        save_model: Lưu model hay không
        
    Returns:
        Dict với đầy đủ kết quả
    """
    from src.utils import SparkManager
    
    spark = SparkManager.get_spark()
    
    # Tạo model
    cardio_model = CardioLogisticModel(spark)
    
    # Chạy pipeline đầy đủ
    results = cardio_model.train_with_cv_pipeline(
        df=df,
        train_ratio=train_ratio,
        param_grid=param_grid,
        num_folds=num_folds
    )
    
    # Save model
    if save_model:
        cardio_model.save_model()
    
    results['model_instance'] = cardio_model
    
    return results