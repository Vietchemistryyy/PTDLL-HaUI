"""
Module Feature Engineering cho dữ liệu bệnh tim mạch
Chuẩn bị features cho model Machine Learning với PySpark
"""
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler, StandardScaler, StringIndexer, OneHotEncoder
from pyspark.ml import Pipeline
from pyspark.ml.feature import MinMaxScaler
import logging
import config

logger = logging.getLogger(__name__)


class CardioFeatureEngineer:
    """Class xử lý feature engineering cho dữ liệu bệnh tim mạch"""
    
    def __init__(self, spark: SparkSession):
        self.spark = spark
        
    def prepare_features_for_ml(self, df: DataFrame) -> DataFrame:
        """
        Chuẩn bị features cho Machine Learning
        
        Args:
            df: Spark DataFrame đã được preprocessing
            
        Returns:
            DataFrame với features đã được chuẩn bị
        """
        logger.info("Chuẩn bị features cho Machine Learning...")
        
        # Chọn các features cần thiết
        feature_cols = [
            'age_years',
            'gender',
            'height',
            'weight',
            'ap_hi',
            'ap_lo',
            'cholesterol',
            'gluc',
            'smoke',
            'alco',
            'active',
            'bmi',
            'pulse_pressure'
        ]
        
        # Kiểm tra xem các cột có tồn tại không
        available_cols = [col for col in feature_cols if col in df.columns]
        
        if len(available_cols) != len(feature_cols):
            missing = set(feature_cols) - set(available_cols)
            logger.warning(f"Thiếu các cột: {missing}")
        
        # Select features và target
        df_ml = df.select(available_cols + ['cardio'])
        
        logger.info(f"✓ Đã chọn {len(available_cols)} features")
        return df_ml
    
    def create_interaction_features(self, df: DataFrame) -> DataFrame:
        """
        Tạo interaction features (tương tác giữa các features)
        
        Args:
            df: Spark DataFrame
            
        Returns:
            DataFrame với interaction features mới
        """
        logger.info("Tạo interaction features...")
        
        # 1. BMI * Age (tương tác giữa BMI và tuổi)
        df = df.withColumn('bmi_age_interaction', F.col('bmi') * F.col('age_years'))
        
        # 2. Blood Pressure Index (chỉ số huyết áp kết hợp)
        df = df.withColumn('bp_index', (F.col('ap_hi') + F.col('ap_lo')) / 2)
        
        # 3. Risk Score (điểm nguy cơ từ các yếu tố lifestyle)
        df = df.withColumn(
            'lifestyle_risk_score',
            F.col('smoke') + F.col('alco') - F.col('active')
        )
        
        # 4. Health Score (điểm sức khỏe từ cholesterol và glucose)
        df = df.withColumn(
            'health_score',
            F.col('cholesterol') + F.col('gluc')
        )
        
        # 5. Height/Weight ratio
        df = df.withColumn('height_weight_ratio', F.col('height') / F.col('weight'))
        
        logger.info("✓ Đã tạo 5 interaction features")
        return df
    
    def create_polynomial_features(self, df: DataFrame, features: list = None) -> DataFrame:
        """
        Tạo polynomial features (bậc 2)
        
        Args:
            df: Spark DataFrame
            features: List các features cần tạo polynomial
            
        Returns:
            DataFrame với polynomial features
        """
        if features is None:
            features = ['bmi', 'age_years', 'ap_hi', 'ap_lo']
        
        logger.info(f"Tạo polynomial features cho: {features}")
        
        for feature in features:
            if feature in df.columns:
                # Tạo feature bậc 2
                df = df.withColumn(f'{feature}_squared', F.pow(F.col(feature), 2))
        
        logger.info(f"✓ Đã tạo polynomial features cho {len(features)} biến")
        return df
    
    def create_binned_features(self, df: DataFrame) -> DataFrame:
        """
        Tạo binned features (phân loại thành nhóm)
        
        Args:
            df: Spark DataFrame
            
        Returns:
            DataFrame với binned features
        """
        logger.info("Tạo binned features...")
        
        # 1. Age bins (đã có age_group rồi, có thể tạo thêm fine-grained)
        df = df.withColumn(
            'age_bin',
            F.when(F.col('age_years') < 35, 0)
            .when((F.col('age_years') >= 35) & (F.col('age_years') < 45), 1)
            .when((F.col('age_years') >= 45) & (F.col('age_years') < 55), 2)
            .when((F.col('age_years') >= 55) & (F.col('age_years') < 65), 3)
            .otherwise(4)
        )
        
        # 2. BMI bins (numerical version of bmi_category)
        df = df.withColumn(
            'bmi_bin',
            F.when(F.col('bmi') < 18.5, 0)
            .when((F.col('bmi') >= 18.5) & (F.col('bmi') < 25), 1)
            .when((F.col('bmi') >= 25) & (F.col('bmi') < 30), 2)
            .otherwise(3)
        )
        
        # 3. Blood Pressure bins
        df = df.withColumn(
            'bp_bin',
            F.when((F.col('ap_hi') < 120) & (F.col('ap_lo') < 80), 0)
            .when((F.col('ap_hi') < 130) & (F.col('ap_lo') < 80), 1)
            .when((F.col('ap_hi') < 140) | (F.col('ap_lo') < 90), 2)
            .otherwise(3)
        )
        
        logger.info("✓ Đã tạo 3 binned features")
        return df
    
    def normalize_features(self, df: DataFrame, feature_cols: list) -> tuple:
        """
        Normalize features sử dụng StandardScaler
        
        Args:
            df: Spark DataFrame
            feature_cols: List các cột cần normalize
            
        Returns:
            Tuple (DataFrame với features normalized, scaler model)
        """
        logger.info(f"Normalize {len(feature_cols)} features...")
        
        # Assemble features into vector
        assembler = VectorAssembler(
            inputCols=feature_cols,
            outputCol="features_raw"
        )
        
        df_assembled = assembler.transform(df)
        
        # Standard Scaler
        scaler = StandardScaler(
            inputCol="features_raw",
            outputCol="features_scaled",
            withStd=True,
            withMean=True
        )
        
        scaler_model = scaler.fit(df_assembled)
        df_scaled = scaler_model.transform(df_assembled)
        
        logger.info("✓ Đã normalize features")
        return df_scaled, scaler_model
    
    def minmax_scale_features(self, df: DataFrame, feature_cols: list) -> tuple:
        """
        MinMax scale features (scale về [0,1])
        
        Args:
            df: Spark DataFrame
            feature_cols: List các cột cần scale
            
        Returns:
            Tuple (DataFrame với features scaled, scaler model)
        """
        logger.info(f"MinMax scale {len(feature_cols)} features...")
        
        # Assemble features into vector
        assembler = VectorAssembler(
            inputCols=feature_cols,
            outputCol="features_raw"
        )
        
        df_assembled = assembler.transform(df)
        
        # MinMax Scaler
        scaler = MinMaxScaler(
            inputCol="features_raw",
            outputCol="features_minmax"
        )
        
        scaler_model = scaler.fit(df_assembled)
        df_scaled = scaler_model.transform(df_assembled)
        
        logger.info("✓ Đã MinMax scale features")
        return df_scaled, scaler_model
    
    def prepare_final_features(self, df: DataFrame, 
                               include_interactions: bool = True,
                               include_polynomial: bool = False,
                               include_binned: bool = True) -> DataFrame:
        """
        Chuẩn bị features cuối cùng cho model
        
        Args:
            df: Spark DataFrame
            include_interactions: Có tạo interaction features không
            include_polynomial: Có tạo polynomial features không
            include_binned: Có tạo binned features không
            
        Returns:
            DataFrame với features đã được chuẩn bị đầy đủ
        """
        logger.info("=" * 60)
        logger.info("CHUẨN BỊ FEATURES CUỐI CÙNG")
        logger.info("=" * 60)
        
        # Bắt đầu với features cơ bản
        df_final = self.prepare_features_for_ml(df)
        
        # Thêm interaction features
        if include_interactions:
            df_final = self.create_interaction_features(df_final)
            logger.info("✓ Đã thêm interaction features")
        
        # Thêm polynomial features
        if include_polynomial:
            df_final = self.create_polynomial_features(df_final)
            logger.info("✓ Đã thêm polynomial features")
        
        # Thêm binned features
        if include_binned:
            df_final = self.create_binned_features(df_final)
            logger.info("✓ Đã thêm binned features")
        
        # Lấy danh sách tất cả features (trừ target)
        all_features = [col for col in df_final.columns if col != 'cardio']
        
        # Assemble tất cả features thành vector
        assembler = VectorAssembler(
            inputCols=all_features,
            outputCol="features_vector",
            handleInvalid="skip"
        )
        
        df_final = assembler.transform(df_final)
        
        logger.info(f"✓ Tổng số features: {len(all_features)}")
        logger.info(f"✓ Các features: {all_features}")
        
        logger.info("=" * 60)
        logger.info("✓ HOÀN THÀNH CHUẨN BỊ FEATURES")
        logger.info("=" * 60)
        
        return df_final
    
    def get_feature_importance_data(self, df: DataFrame, feature_names: list) -> DataFrame:
        """
        Chuẩn bị dữ liệu để tính feature importance
        
        Args:
            df: Spark DataFrame
            feature_names: List tên các features
            
        Returns:
            DataFrame với features và target
        """
        logger.info("Chuẩn bị dữ liệu cho feature importance...")
        
        # Assemble features
        assembler = VectorAssembler(
            inputCols=feature_names,
            outputCol="features",
            handleInvalid="skip"
        )
        
        df_assembled = assembler.transform(df)
        
        return df_assembled.select("features", "cardio")
    
    def split_train_test(self, df: DataFrame, train_ratio: float = 0.8, 
                         seed: int = 42) -> tuple:
        """
        Chia dữ liệu thành train và test
        
        Args:
            df: Spark DataFrame
            train_ratio: Tỷ lệ dữ liệu train
            seed: Random seed
            
        Returns:
            Tuple (train_df, test_df)
        """
        logger.info(f"Chia dữ liệu train/test với tỷ lệ {train_ratio}/{1-train_ratio}")
        
        train_df, test_df = df.randomSplit([train_ratio, 1 - train_ratio], seed=seed)
        
        train_count = train_df.count()
        test_count = test_df.count()
        
        logger.info(f"✓ Train set: {train_count:,} samples")
        logger.info(f"✓ Test set: {test_count:,} samples")
        
        return train_df, test_df
    
    def create_ml_pipeline(self, feature_cols: list) -> Pipeline:
        """
        Tạo ML Pipeline cho feature engineering
        
        Args:
            feature_cols: List các cột features
            
        Returns:
            Spark ML Pipeline
        """
        logger.info("Tạo ML Pipeline...")
        
        # Stage 1: Assemble features
        assembler = VectorAssembler(
            inputCols=feature_cols,
            outputCol="features_raw",
            handleInvalid="skip"
        )
        
        # Stage 2: Scale features
        scaler = StandardScaler(
            inputCol="features_raw",
            outputCol="features",
            withStd=True,
            withMean=True
        )
        
        # Create pipeline
        pipeline = Pipeline(stages=[assembler, scaler])
        
        logger.info("✓ Đã tạo ML Pipeline với 2 stages")
        return pipeline
    
    def get_feature_statistics(self, df: DataFrame, feature_cols: list) -> dict:
        """
        Lấy thống kê về features
        
        Args:
            df: Spark DataFrame
            feature_cols: List các cột features
            
        Returns:
            Dictionary chứa thống kê
        """
        logger.info("Tính thống kê features...")
        
        stats = {}
        
        for col in feature_cols:
            col_stats = df.select(
                F.mean(col).alias('mean'),
                F.stddev(col).alias('std'),
                F.min(col).alias('min'),
                F.max(col).alias('max')
            ).collect()[0]
            
            stats[col] = {
                'mean': round(col_stats['mean'], 4) if col_stats['mean'] else None,
                'std': round(col_stats['std'], 4) if col_stats['std'] else None,
                'min': col_stats['min'],
                'max': col_stats['max']
            }
        
        return stats


def get_default_feature_list(include_engineered: bool = True) -> list:
    """
    Lấy danh sách features mặc định
    
    Args:
        include_engineered: Có bao gồm engineered features không
        
    Returns:
        List các feature names
    """
    base_features = [
        'age_years',
        'gender',
        'height',
        'weight',
        'ap_hi',
        'ap_lo',
        'cholesterol',
        'gluc',
        'smoke',
        'alco',
        'active',
        'bmi',
        'pulse_pressure'
    ]
    
    if include_engineered:
        engineered_features = [
            'bmi_age_interaction',
            'bp_index',
            'lifestyle_risk_score',
            'health_score',
            'height_weight_ratio',
            'age_bin',
            'bmi_bin',
            'bp_bin'
        ]
        return base_features + engineered_features
    
    return base_features