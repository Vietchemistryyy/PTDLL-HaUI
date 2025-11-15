"""
Module Exploratory Data Analysis (EDA) cho dữ liệu bệnh tim mạch
Sử dụng PySpark cho xử lý và phân tích dữ liệu lớn
"""
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import *
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class CardioEDA:
    """Class phân tích khám phá dữ liệu bệnh tim mạch"""
    
    def __init__(self, spark: SparkSession):
        self.spark = spark
        
    def get_basic_statistics(self, df: DataFrame) -> dict:
        """
        Lấy thống kê cơ bản về dataset
        
        Args:
            df: Spark DataFrame
            
        Returns:
            Dictionary chứa các thống kê cơ bản
        """
        logger.info("Tính toán thống kê cơ bản...")
        
        total_records = df.count()
        total_columns = len(df.columns)
        
        # Đếm số lượng bệnh nhân có/không có bệnh tim
        cardio_counts = df.groupBy("cardio").count().collect()
        cardio_positive = [row['count'] for row in cardio_counts if row['cardio'] == 1][0]
        cardio_negative = total_records - cardio_positive
        
        stats = {
            'total_records': total_records,
            'total_columns': total_columns,
            'cardio_positive': cardio_positive,
            'cardio_negative': cardio_negative,
            'cardio_rate': round(cardio_positive / total_records * 100, 2),
            'columns': df.columns
        }
        
        return stats
    
    def analyze_numerical_features(self, df: DataFrame, features: list = None) -> pd.DataFrame:
        """
        Phân tích các biến số (numerical features)
        
        Args:
            df: Spark DataFrame
            features: List các cột cần phân tích (nếu None thì tự động detect)
            
        Returns:
            Pandas DataFrame chứa thống kê
        """
        logger.info("Phân tích các biến số...")
        
        if features is None:
            # Tự động detect numeric columns
            features = [f.name for f in df.schema.fields 
                       if isinstance(f.dataType, (IntegerType, LongType, DoubleType, FloatType))]
        
        # Tính các thống kê cho từng feature
        stats_list = []
        
        for col in features:
            stats = df.select(
                F.mean(col).alias('mean'),
                F.stddev(col).alias('std'),
                F.min(col).alias('min'),
                F.expr(f'percentile({col}, 0.25)').alias('q25'),
                F.expr(f'percentile({col}, 0.50)').alias('median'),
                F.expr(f'percentile({col}, 0.75)').alias('q75'),
                F.max(col).alias('max')
            ).collect()[0]
            
            stats_dict = {
                'feature': col,
                'mean': round(stats['mean'], 2) if stats['mean'] else None,
                'std': round(stats['std'], 2) if stats['std'] else None,
                'min': stats['min'],
                'q25': stats['q25'],
                'median': stats['median'],
                'q75': stats['q75'],
                'max': stats['max']
            }
            stats_list.append(stats_dict)
        
        return pd.DataFrame(stats_list)
    
    def analyze_categorical_features(self, df: DataFrame, features: list) -> dict:
        """
        Phân tích các biến phân loại (categorical features)
        
        Args:
            df: Spark DataFrame
            features: List các cột categorical
            
        Returns:
            Dictionary chứa phân bố cho từng feature
        """
        logger.info("Phân tích các biến phân loại...")
        
        result = {}
        
        for col in features:
            # Đếm số lượng và tỷ lệ cho mỗi category
            counts = df.groupBy(col).agg(
                F.count("*").alias("count")
            ).orderBy(col)
            
            total = df.count()
            
            # Convert to pandas for easier handling
            counts_pd = counts.toPandas()
            counts_pd['percentage'] = (counts_pd['count'] / total * 100).round(2)
            
            result[col] = counts_pd
        
        return result
    
    def analyze_correlation_with_target(self, df: DataFrame, target_col: str = 'cardio') -> pd.DataFrame:
        """
        Phân tích correlation của các features với target variable
        
        Args:
            df: Spark DataFrame
            target_col: Tên cột target
            
        Returns:
            Pandas DataFrame chứa correlation scores
        """
        logger.info(f"Phân tích correlation với {target_col}...")
        
        # Lấy các cột numeric
        numeric_cols = [f.name for f in df.schema.fields 
                       if isinstance(f.dataType, (IntegerType, LongType, DoubleType, FloatType))
                       and f.name != target_col]
        
        correlations = []
        
        for col in numeric_cols:
            corr = df.stat.corr(col, target_col)
            correlations.append({
                'feature': col,
                'correlation': round(corr, 4)
            })
        
        # Sort by absolute correlation
        corr_df = pd.DataFrame(correlations)
        corr_df['abs_correlation'] = corr_df['correlation'].abs()
        corr_df = corr_df.sort_values('abs_correlation', ascending=False)
        corr_df = corr_df.drop('abs_correlation', axis=1)
        
        return corr_df
    
    def analyze_feature_by_target(self, df: DataFrame, feature: str, target: str = 'cardio') -> pd.DataFrame:
        """
        Phân tích phân bố của feature theo target variable
        
        Args:
            df: Spark DataFrame
            feature: Tên cột feature cần phân tích
            target: Tên cột target
            
        Returns:
            Pandas DataFrame chứa thống kê theo từng nhóm target
        """
        logger.info(f"Phân tích {feature} theo {target}...")
        
        # Group by target và tính thống kê
        stats = df.groupBy(target).agg(
            F.mean(feature).alias('mean'),
            F.stddev(feature).alias('std'),
            F.min(feature).alias('min'),
            F.expr(f'percentile({feature}, 0.50)').alias('median'),
            F.max(feature).alias('max'),
            F.count("*").alias('count')
        ).orderBy(target)
        
        result = stats.toPandas()
        result['mean'] = result['mean'].round(2)
        result['std'] = result['std'].round(2)
        
        return result
    
    def analyze_age_distribution(self, df: DataFrame) -> pd.DataFrame:
        """
        Phân tích phân bố tuổi chi tiết
        
        Args:
            df: Spark DataFrame (phải có cột age_years)
            
        Returns:
            Pandas DataFrame chứa phân bố tuổi
        """
        logger.info("Phân tích phân bố tuổi...")
        
        age_stats = df.groupBy('age_group', 'cardio').agg(
            F.count("*").alias('count')
        ).orderBy('age_group', 'cardio')
        
        result = age_stats.toPandas()
        
        # Tính tỷ lệ trong mỗi nhóm tuổi
        total_by_age = result.groupby('age_group')['count'].sum()
        result['percentage'] = result.apply(
            lambda row: round(row['count'] / total_by_age[row['age_group']] * 100, 2),
            axis=1
        )
        
        return result
    
    def analyze_bmi_distribution(self, df: DataFrame) -> pd.DataFrame:
        """
        Phân tích phân bố BMI chi tiết
        
        Args:
            df: Spark DataFrame (phải có cột bmi_category)
            
        Returns:
            Pandas DataFrame chứa phân bố BMI
        """
        logger.info("Phân tích phân bố BMI...")
        
        bmi_stats = df.groupBy('bmi_category', 'cardio').agg(
            F.count("*").alias('count')
        ).orderBy('bmi_category', 'cardio')
        
        result = bmi_stats.toPandas()
        
        # Tính tỷ lệ trong mỗi nhóm BMI
        total_by_bmi = result.groupby('bmi_category')['count'].sum()
        result['percentage'] = result.apply(
            lambda row: round(row['count'] / total_by_bmi[row['bmi_category']] * 100, 2),
            axis=1
        )
        
        return result
    
    def analyze_risk_factors(self, df: DataFrame) -> pd.DataFrame:
        """
        Phân tích các yếu tố nguy cơ (smoke, alco, cholesterol, gluc)
        
        Args:
            df: Spark DataFrame
            
        Returns:
            Pandas DataFrame chứa tỷ lệ mắc bệnh theo từng yếu tố
        """
        logger.info("Phân tích các yếu tố nguy cơ...")
        
        risk_factors = ['smoke', 'alco', 'cholesterol', 'gluc', 'active']
        results = []
        
        for factor in risk_factors:
            # Tính tỷ lệ mắc bệnh cho mỗi level của risk factor
            factor_stats = df.groupBy(factor).agg(
                F.count("*").alias('total'),
                F.sum(F.when(F.col('cardio') == 1, 1).otherwise(0)).alias('cardio_positive')
            )
            
            factor_pd = factor_stats.toPandas()
            factor_pd['cardio_rate'] = (factor_pd['cardio_positive'] / factor_pd['total'] * 100).round(2)
            factor_pd['risk_factor'] = factor
            factor_pd = factor_pd.rename(columns={factor: 'level'})
            
            results.append(factor_pd)
        
        return pd.concat(results, ignore_index=True)
    
    def analyze_blood_pressure(self, df: DataFrame) -> dict:
        """
        Phân tích chi tiết về huyết áp
        
        Args:
            df: Spark DataFrame
            
        Returns:
            Dictionary chứa các phân tích về huyết áp
        """
        logger.info("Phân tích huyết áp...")
        
        # Phân loại huyết áp theo chuẩn
        df_bp = df.withColumn(
            'bp_category',
            F.when((F.col('ap_hi') < 120) & (F.col('ap_lo') < 80), 'Bình thường')
            .when((F.col('ap_hi').between(120, 129)) & (F.col('ap_lo') < 80), 'Cao nhẹ')
            .when((F.col('ap_hi').between(130, 139)) | (F.col('ap_lo').between(80, 89)), 'Tăng HA độ 1')
            .when((F.col('ap_hi') >= 140) | (F.col('ap_lo') >= 90), 'Tăng HA độ 2')
            .otherwise('Khác')
        )
        
        # Thống kê theo category
        bp_stats = df_bp.groupBy('bp_category', 'cardio').agg(
            F.count("*").alias('count')
        ).orderBy('bp_category', 'cardio')
        
        bp_result = bp_stats.toPandas()
        
        # Tính tỷ lệ
        total_by_bp = bp_result.groupby('bp_category')['count'].sum()
        bp_result['percentage'] = bp_result.apply(
            lambda row: round(row['count'] / total_by_bp[row['bp_category']] * 100, 2),
            axis=1
        )
        
        return {
            'distribution': bp_result,
            'by_target': self.analyze_feature_by_target(df, 'ap_hi', 'cardio'),
            'pulse_pressure': self.analyze_feature_by_target(df, 'pulse_pressure', 'cardio')
        }
    
    def generate_comprehensive_report(self, df: DataFrame) -> dict:
        """
        Tạo báo cáo phân tích toàn diện
        
        Args:
            df: Spark DataFrame
            
        Returns:
            Dictionary chứa tất cả các phân tích
        """
        logger.info("=" * 60)
        logger.info("TẠO BÁO CÁO PHÂN TÍCH TOÀN DIỆN")
        logger.info("=" * 60)
        
        report = {
            'basic_stats': self.get_basic_statistics(df),
            'numerical_features': self.analyze_numerical_features(df),
            'categorical_features': self.analyze_categorical_features(
                df, ['gender', 'cholesterol', 'gluc', 'smoke', 'alco', 'active']
            ),
            'correlation': self.analyze_correlation_with_target(df),
            'age_analysis': self.analyze_age_distribution(df),
            'bmi_analysis': self.analyze_bmi_distribution(df),
            'risk_factors': self.analyze_risk_factors(df),
            'blood_pressure': self.analyze_blood_pressure(df)
        }
        
        logger.info("✓ Hoàn thành báo cáo phân tích")
        return report
    
    def get_data_for_visualization(self, df: DataFrame, sample_size: int = None) -> pd.DataFrame:
        """
        Lấy dữ liệu để visualization (convert sang Pandas)
        
        Args:
            df: Spark DataFrame
            sample_size: Số lượng samples cần lấy (None = lấy tất cả)
            
        Returns:
            Pandas DataFrame
        """
        logger.info("Chuyển đổi dữ liệu sang Pandas để visualization...")
        
        if sample_size and sample_size < df.count():
            # Sample dữ liệu nếu quá lớn
            fraction = sample_size / df.count()
            df_sample = df.sample(fraction=fraction, seed=42)
            logger.info(f"Đã sample {sample_size} records")
        else:
            df_sample = df
        
        return df_sample.toPandas()