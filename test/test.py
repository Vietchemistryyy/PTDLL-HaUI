"""
Script test tiền xử lý dữ liệu
Chạy file này để test toàn bộ pipeline
"""
import sys

sys.path.append('src')

from src.utils import init_spark
from src.data_preprocessing import CardioDataLoader, CardioDataPreprocessor, get_statistical_summary
import config


def main():
    print("=" * 60)
    print("TEST PIPELINE TIỀN XỬ LÝ DỮ LIỆU BỆNH TIM MẠCH")
    print("=" * 60)

    # 1. Khởi tạo Spark
    print("\n1. Khởi tạo Spark...")
    spark = init_spark(config.SPARK_CONFIG)

    # 2. Load dữ liệu
    print("\n2. Load dữ liệu...")
    loader = CardioDataLoader(spark)
    df_raw = loader.load_data()

    # 3. Xem thông tin dữ liệu
    print("\n3. Thông tin dữ liệu:")
    info = loader.get_data_info(df_raw)
    print(f"   - Tổng số dòng: {info['total_records']:,}")
    print(f"   - Tổng số cột: {info['total_columns']}")
    print(f"   - Các cột: {', '.join(info['columns'])}")

    if info['missing_values']:
        print(f"   - Missing values: {info['missing_values']}")
    else:
        print("   - Không có missing values")

    # 4. Xem mẫu dữ liệu
    print("\n4. Mẫu dữ liệu (5 dòng đầu):")
    df_raw.show(5, truncate=False)

    # 5. Thống kê mô tả trước xử lý
    print("\n5. Thống kê mô tả (trước xử lý):")
    get_statistical_summary(df_raw).show()

    # 6. Tiền xử lý dữ liệu
    print("\n6. Bắt đầu tiền xử lý...")
    preprocessor = CardioDataPreprocessor(spark)
    df_processed = preprocessor.preprocess_pipeline(df_raw)

    # 7. Xem kết quả sau xử lý
    print("\n7. Dữ liệu sau xử lý (5 dòng đầu):")
    df_processed.show(5, truncate=False)

    # 8. Thống kê sau xử lý
    print("\n8. Thống kê mô tả (sau xử lý):")
    get_statistical_summary(df_processed).show()

    # 9. Kiểm tra các cột mới
    print("\n9. Các cột đã thêm:")
    new_cols = [col for col in df_processed.columns if col not in df_raw.columns]
    print(f"   {', '.join(new_cols)}")

    # 10. Phân bố nhóm tuổi
    print("\n10. Phân bố nhóm tuổi:")
    df_processed.groupBy("age_group").count().orderBy("age_group").show()

    # 11. Phân bố BMI
    print("\n11. Phân bố BMI:")
    df_processed.groupBy("bmi_category").count().orderBy("bmi_category").show()

    # 12. Phân bố target (cardio)
    print("\n12. Phân bố bệnh tim mạch:")
    df_processed.groupBy("cardio").count().show()
    cardio_rate = df_processed.filter("cardio = 1").count() / df_processed.count() * 100
    print(f"   Tỷ lệ mắc bệnh: {cardio_rate:.2f}%")

    # 13. Lưu dữ liệu
    print("\n13. Lưu dữ liệu đã xử lý...")
    preprocessor.save_processed_data(df_processed)

    print("\n" + "=" * 60)
    print("✓ HOÀN THÀNH TEST PIPELINE")
    print("=" * 60)

    # Dừng Spark
    spark.stop()


if __name__ == "__main__":
    main()