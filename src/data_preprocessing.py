"""
Module xử lý và tiền xử lý dữ liệu bệnh tim mạch
"""
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import *
import config
import logging

logger = logging.getLogger(__name__)


class CardioDataLoader:
    """Load và xử lý dữ liệu cardiovascular disease"""

    def __init__(self, spark: SparkSession):
        self.spark = spark

    def load_data(self, file_path: str = None) -> DataFrame:
        """
        Load dữ liệu từ CSV

        Args:
            file_path: Đường dẫn file CSV

        Returns:
            Spark DataFrame
        """
        if file_path is None:
            file_path = str(config.RAW_DATA_FILE)

        logger.info(f"Đang load dữ liệu từ: {file_path}")

        # Load với inferSchema để tự động nhận diện kiểu dữ liệu
        df = self.spark.read.csv(
            file_path,
            header=True,
            sep=",",  # Dataset dùng dấu phẩy
            inferSchema=True
        )

        logger.info(f"✓ Đã load {df.count()} dòng dữ liệu")
        return df

    def get_data_info(self, df: DataFrame) -> dict:
        """
        Lấy thông tin tổng quan về dữ liệu

        Args:
            df: Spark DataFrame

        Returns:
            Dictionary chứa thông tin
        """
        info = {
            "total_records": df.count(),
            "total_columns": len(df.columns),
            "columns": df.columns,
            "schema": df.schema,
            "missing_values": {}
        }

        # Kiểm tra missing values
        for col in df.columns:
            missing_count = df.filter(F.col(col).isNull()).count()
            if missing_count > 0:
                info["missing_values"][col] = missing_count

        return info


class CardioDataPreprocessor:
    """Tiền xử lý dữ liệu cardiovascular disease"""

    def __init__(self, spark: SparkSession):
        self.spark = spark

    def convert_age_to_years(self, df: DataFrame) -> DataFrame:
        """
        Chuyển đổi tuổi từ ngày sang năm

        Args:
            df: Spark DataFrame

        Returns:
            DataFrame với cột age_years mới
        """
        logger.info("Chuyển đổi tuổi từ ngày sang năm")
        df = df.withColumn("age_years", F.round(F.col("age") / 365.25, 1))
        return df

    def calculate_bmi(self, df: DataFrame) -> DataFrame:
        """
        Tính chỉ số BMI (Body Mass Index)

        Formula: BMI = weight(kg) / (height(m))^2

        Args:
            df: Spark DataFrame

        Returns:
            DataFrame với cột bmi mới
        """
        logger.info("Tính chỉ số BMI")
        df = df.withColumn(
            "bmi",
            F.round(
                F.col("weight") / F.pow(F.col("height") / 100, 2),
                2
            )
        )
        return df

    def calculate_pulse_pressure(self, df: DataFrame) -> DataFrame:
        """
        Tính pulse pressure (hiệu số huyết áp)

        Formula: Pulse Pressure = ap_hi - ap_lo

        Args:
            df: Spark DataFrame

        Returns:
            DataFrame với cột pulse_pressure mới
        """
        logger.info("Tính pulse pressure")
        df = df.withColumn(
            "pulse_pressure",
            F.col("ap_hi") - F.col("ap_lo")
        )
        return df

    def remove_outliers(self, df: DataFrame, thresholds: dict = None) -> DataFrame:
        """
        Loại bỏ outliers dựa trên ngưỡng cho trước

        Args:
            df: Spark DataFrame
            thresholds: Dictionary chứa min/max cho mỗi cột

        Returns:
            DataFrame đã loại bỏ outliers
        """
        if thresholds is None:
            thresholds = config.OUTLIER_THRESHOLDS

        logger.info("Loại bỏ outliers")
        original_count = df.count()

        # Áp dụng filter cho từng cột
        for col_name, (min_val, max_val) in thresholds.items():
            if col_name in df.columns:
                df = df.filter(
                    (F.col(col_name) >= min_val) &
                    (F.col(col_name) <= max_val)
                )

        new_count = df.count()
        removed = original_count - new_count

        if original_count > 0:
            percentage = removed / original_count * 100
            logger.info(f"✓ Đã loại bỏ {removed} outliers ({percentage:.2f}%)")
        else:
            logger.info(f"✓ Đã loại bỏ {removed} outliers")

        return df

    def handle_missing_values(self, df: DataFrame) -> DataFrame:
        """
        Xử lý missing values

        Args:
            df: Spark DataFrame

        Returns:
            DataFrame đã xử lý missing values
        """
        logger.info("Xử lý missing values")

        # Đếm missing values
        missing_counts = {}
        for col in df.columns:
            missing = df.filter(F.col(col).isNull()).count()
            if missing > 0:
                missing_counts[col] = missing

        if missing_counts:
            logger.info(f"Tìm thấy missing values: {missing_counts}")
            # Drop rows với missing values
            df = df.dropna()
            logger.info(f"✓ Đã xóa các dòng có missing values")
        else:
            logger.info("✓ Không có missing values")

        return df

    def create_age_groups(self, df: DataFrame) -> DataFrame:
        """
        Tạo nhóm tuổi

        Args:
            df: Spark DataFrame (phải có cột age_years)

        Returns:
            DataFrame với cột age_group mới
        """
        logger.info("Tạo nhóm tuổi")
        df = df.withColumn(
            "age_group",
            F.when(F.col("age_years") < 40, "< 40")
            .when((F.col("age_years") >= 40) & (F.col("age_years") < 50), "40-49")
            .when((F.col("age_years") >= 50) & (F.col("age_years") < 60), "50-59")
            .otherwise("≥ 60")
        )
        return df

    def create_bmi_categories(self, df: DataFrame) -> DataFrame:
        """
        Phân loại BMI theo WHO

        Args:
            df: Spark DataFrame (phải có cột bmi)

        Returns:
            DataFrame với cột bmi_category mới
        """
        logger.info("Phân loại BMI")
        df = df.withColumn(
            "bmi_category",
            F.when(F.col("bmi") < 18.5, "Thiếu cân")
            .when((F.col("bmi") >= 18.5) & (F.col("bmi") < 25), "Bình thường")
            .when((F.col("bmi") >= 25) & (F.col("bmi") < 30), "Thừa cân")
            .otherwise("Béo phì")
        )
        return df

    def preprocess_pipeline(self, df: DataFrame) -> DataFrame:
        """
        Pipeline xử lý hoàn chỉnh

        Args:
            df: Spark DataFrame raw

        Returns:
            DataFrame đã được xử lý
        """
        logger.info("=" * 50)
        logger.info("BẮT ĐẦU PIPELINE TIỀN XỬ LÝ DỮ LIỆU")
        logger.info("=" * 50)

        # 1. Xử lý missing values
        df = self.handle_missing_values(df)

        # 2. Loại bỏ outliers
        df = self.remove_outliers(df)

        # 3. Tạo features mới
        df = self.convert_age_to_years(df)
        df = self.calculate_bmi(df)
        df = self.calculate_pulse_pressure(df)

        # 4. Tạo categories
        df = self.create_age_groups(df)
        df = self.create_bmi_categories(df)

        logger.info("=" * 50)
        logger.info("✓ HOÀN THÀNH PIPELINE TIỀN XỬ LÝ")
        logger.info(f"✓ Dữ liệu cuối cùng: {df.count()} dòng")
        logger.info("=" * 50)

        return df

    def save_processed_data(self, df: DataFrame, output_path: str = None):
        """
        Lưu dữ liệu đã xử lý

        Args:
            df: Spark DataFrame
            output_path: Đường dẫn lưu file
        """
        if output_path is None:
            output_path = str(config.PROCESSED_DATA_FILE)

        logger.info(f"Lưu dữ liệu vào: {output_path}")
        df.write.mode("overwrite").parquet(output_path)
        logger.info("✓ Đã lưu dữ liệu")


def get_statistical_summary(df: DataFrame) -> DataFrame:
    """
    Lấy thống kê mô tả cho dữ liệu

    Args:
        df: Spark DataFrame

    Returns:
        DataFrame chứa thống kê
    """
    numeric_cols = [col for col in df.columns
                    if df.schema[col].dataType in [IntegerType(), DoubleType()]]

    return df.select(numeric_cols).describe()