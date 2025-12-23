import os
import sys
from pathlib import Path

# Setup environment BEFORE importing PySpark
if 'SPARK_HOME' in os.environ:
    del os.environ['SPARK_HOME']

# FIX: Xử lý đường dẫn có dấu cách và tiếng Việt
python_exe = sys.executable
if ' ' in python_exe:
    # Wrap trong quotes nếu có dấu cách
    python_exe = f'"{python_exe}"'

os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

# Disable Spark UI để tránh lỗi
os.environ['SPARK_LOCAL_IP'] = '127.0.0.1'

# Initialize findspark
try:
    import findspark
    findspark.init()
except:
    pass

from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.ml.feature import VectorAssembler

class HeartDiseasePredictor:
    """Class để load PySpark model và thực hiện dự đoán"""
    
    def __init__(self):
        """Khởi tạo Spark session và load model"""
        self.spark = None
        self.model = None
        self.feature_names = None
        self.init_spark()
        self.load_model()
    
    def init_spark(self):
        """Khởi tạo Spark session"""
        try:
            # Stop existing Spark session if any
            try:
                SparkSession.builder.getOrCreate().stop()
            except:
                pass
            
            self.spark = SparkSession.builder \
                .appName("HeartDiseasePredictor") \
                .master("local[1]") \
                .config("spark.driver.memory", "512m") \
                .config("spark.executor.memory", "512m") \
                .config("spark.sql.shuffle.partitions", "2") \
                .config("spark.driver.host", "127.0.0.1") \
                .config("spark.driver.bindAddress", "127.0.0.1") \
                .config("spark.ui.enabled", "false") \
                .config("spark.ui.showConsoleProgress", "false") \
                .config("spark.sql.execution.arrow.pyspark.enabled", "false") \
                .getOrCreate()
            
            # Set log level to ERROR to reduce noise
            self.spark.sparkContext.setLogLevel("ERROR")
            
            print("Da khoi tao Spark session thanh cong")
            print(f"Spark version: {self.spark.version}")
            
        except Exception as e:
            print(f"Loi khi khoi tao Spark: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def load_model(self):
        """Load PySpark Pipeline (bao gồm StandardScaler và LogisticRegression)"""
        try:
            base_path = Path(__file__).parent.parent
            pipeline_path = str(base_path / 'model' / 'heart_disease_pipeline')
            
            # Load PySpark Pipeline
            self.model = PipelineModel.load(pipeline_path)
            
            # Load feature names
            import json
            features_path = base_path / 'model' / 'feature_names.json'
            with open(features_path, 'r') as f:
                feature_data = json.load(f)
                self.feature_names = feature_data['features']
            
            print(f"Da tai PySpark Pipeline thanh cong tu {pipeline_path}")
            print(f"Pipeline stages: {len(self.model.stages)}")
            print(f"Cac features: {self.feature_names}")
            
        except Exception as e:
            print(f"Loi khi tai Pipeline: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def predict(self, input_data):
        """
        Dự đoán từ input data sử dụng PySpark
        
        Args:
            input_data (dict): Dictionary chứa các features cơ bản
            
        Returns:
            dict: Kết quả dự đoán với probability
        """
        try:
            # Lấy và validate các features cơ bản
            age_years = float(input_data.get('age_years', 50))
            if age_years < 0 or age_years > 120:
                raise ValueError("Tuổi không hợp lệ")
                
            gender = float(input_data.get('gender', 1))
            if gender not in [1, 2]:
                raise ValueError("Giới tính không hợp lệ")
                
            height = float(input_data.get('height', 165))
            if height < 50 or height > 300:
                raise ValueError("Chiều cao không hợp lệ")
                
            weight = float(input_data.get('weight', 70))
            if weight < 20 or weight > 300:
                raise ValueError("Cân nặng không hợp lệ")
                
            ap_hi = float(input_data.get('ap_hi', 120))
            if ap_hi < 50 or ap_hi > 300:
                raise ValueError("Huyết áp tâm thu không hợp lệ")
                
            ap_lo = float(input_data.get('ap_lo', 80))
            if ap_lo < 30 or ap_lo > 200:
                raise ValueError("Huyết áp tâm trương không hợp lệ")
                
            if ap_lo >= ap_hi:
                raise ValueError("Huyết áp tâm trương phải nhỏ hơn huyết áp tâm thu")
                
            cholesterol = float(input_data.get('cholesterol', 1))
            if cholesterol not in [1, 2, 3]:
                raise ValueError("Cholesterol không hợp lệ")
                
            gluc = float(input_data.get('gluc', 1))
            if gluc not in [1, 2, 3]:
                raise ValueError("Glucose không hợp lệ")
                
            smoke = float(input_data.get('smoke', 0))
            if smoke not in [0, 1]:
                raise ValueError("Hút thuốc không hợp lệ")
                
            alco = float(input_data.get('alco', 0))
            if alco not in [0, 1]:
                raise ValueError("Uống rượu không hợp lệ")
                
            active = float(input_data.get('active', 1))
            if active not in [0, 1]:
                raise ValueError("Hoạt động thể chất không hợp lệ")
            
            # Tính toán engineered features
            bmi = weight / ((height / 100) ** 2)
            pulse_pressure = ap_hi - ap_lo
            
            # Tạo Spark DataFrame với features (số lượng tùy theo model)
            # Nếu model có 13 features: chỉ dùng features cơ bản
            # Nếu model có 17 features: thêm 4 features nâng cao
            if len(self.feature_names) == 13:
                # Model cũ - 13 features
                data = [(
                    age_years, gender, height, weight, ap_hi, ap_lo,
                    cholesterol, gluc, smoke, alco, active,
                    bmi, pulse_pressure
                )]
            else:
                # Model mới - 17 features (giống notebook 01)
                # BP Category
                if ap_hi < 120 and ap_lo < 80:
                    bp_category_idx = 0  # Normal
                elif ap_hi < 130 and ap_lo < 80:
                    bp_category_idx = 1  # Elevated
                elif ap_hi < 140 or ap_lo < 90:
                    bp_category_idx = 2  # Stage 1
                elif ap_hi < 180 or ap_lo < 120:
                    bp_category_idx = 3  # Stage 2
                else:
                    bp_category_idx = 4  # Crisis
                
                # Age Group
                if age_years < 40:
                    age_group_idx = 0
                elif age_years < 50:
                    age_group_idx = 1
                elif age_years < 60:
                    age_group_idx = 2
                else:
                    age_group_idx = 3
                
                # BMI Category
                if bmi < 18.5:
                    bmi_category_idx = 0  # Underweight
                elif bmi < 25:
                    bmi_category_idx = 1  # Normal
                elif bmi < 30:
                    bmi_category_idx = 2  # Overweight
                else:
                    bmi_category_idx = 3  # Obese
                
                # Risk Score
                risk_score_feature = (
                    (cholesterol - 1) * 2 +
                    (gluc - 1) * 2 +
                    smoke * 3 +
                    alco * 1 +
                    (1 - active) * 2
                )
                
                data = [(
                    age_years, gender, height, weight, ap_hi, ap_lo,
                    cholesterol, gluc, smoke, alco, active,
                    bmi, pulse_pressure, bp_category_idx, age_group_idx,
                    bmi_category_idx, risk_score_feature
                )]
            
            columns = self.feature_names
            input_df = self.spark.createDataFrame(data, columns)
            
            # Tạo cột 'features' vector từ các cột feature
            assembler = VectorAssembler(
                inputCols=self.feature_names,
                outputCol="features"
            )
            input_df = assembler.transform(input_df)
            
            # Dự đoán với PySpark Pipeline
            # Pipeline tự động thực hiện: StandardScaler -> LogisticRegression
            predictions = self.model.transform(input_df)
            
            # Lấy kết quả
            result_row = predictions.select("probability", "prediction").collect()[0]
            probability = result_row['probability'].toArray()
            prediction = int(result_row['prediction'])
            
            # Tính risk score để hiển thị
            risk_score = (
                (cholesterol - 1) * 2 +
                (gluc - 1) * 2 +
                smoke * 3 +
                alco * 1 +
                (1 - active) * 2
            )
            
            # Tạo giải thích cho độ tin cậy
            confidence_explanation = self._explain_confidence(
                probability[1], bmi, ap_hi, ap_lo, cholesterol, gluc, 
                smoke, alco, active, age_years, risk_score
            )
            
            # Chuẩn bị kết quả
            result = {
                'prediction': int(prediction),
                'prediction_label': 'Có bệnh tim' if prediction == 1 else 'Không có bệnh tim',
                'probability': {
                    'no_disease': float(probability[0]),
                    'has_disease': float(probability[1])
                },
                'confidence': float(max(probability)) * 100,
                'risk_level': self._get_risk_level(probability[1]),
                'bmi': round(bmi, 2),
                'pulse_pressure': round(pulse_pressure, 2),
                'risk_score': round(risk_score, 2),
                'explanation': confidence_explanation
            }
            
            return result
            
        except ValueError as e:
            raise ValueError(f"Dữ liệu không hợp lệ: {str(e)}")
        except Exception as e:
            raise Exception(f"Lỗi khi dự đoán: {str(e)}")
    
    def _get_risk_level(self, prob):
        """Xác định mức độ rủi ro dựa trên probability"""
        if prob < 0.3:
            return 'Thấp'
        elif prob < 0.6:
            return 'Trung bình'
        elif prob < 0.8:
            return 'Cao'
        else:
            return 'Rất cao'
    
    def _explain_confidence(self, prob_disease, bmi, ap_hi, ap_lo, cholesterol, 
                           gluc, smoke, alco, active, age, risk_score):
        """Giải thích tại sao độ tin cậy cao/thấp"""
        reasons = []
        
        # Phân tích các yếu tố nguy cơ
        if ap_hi >= 180 or ap_lo >= 120:
            reasons.append("Huyết áp rất cao (Crisis)")
        elif ap_hi >= 140 or ap_lo >= 90:
            reasons.append("Huyết áp cao (Stage 2)")
        elif ap_hi >= 130 or ap_lo >= 80:
            reasons.append("Huyết áp hơi cao (Stage 1)")
            
        if cholesterol >= 3:
            reasons.append("Cholesterol rất cao")
        elif cholesterol == 2:
            reasons.append("Cholesterol cao hơn bình thường")
            
        if gluc >= 3:
            reasons.append("Đường huyết rất cao")
        elif gluc == 2:
            reasons.append("Đường huyết cao hơn bình thường")
            
        if bmi >= 30:
            reasons.append(f"BMI cao ({bmi:.1f} - Béo phì)")
        elif bmi >= 25:
            reasons.append(f"BMI hơi cao ({bmi:.1f} - Thừa cân)")
            
        if smoke == 1:
            reasons.append("Có hút thuốc")
            
        if alco == 1:
            reasons.append("Có uống rượu")
            
        if active == 0:
            reasons.append("Không hoạt động thể chất")
            
        if age >= 60:
            reasons.append(f"Tuổi cao ({int(age)} tuổi)")
        elif age >= 50:
            reasons.append(f"Tuổi trung niên ({int(age)} tuổi)")
        
        # Tạo câu giải thích
        if prob_disease >= 0.7:
            if len(reasons) >= 3:
                explanation = f"Độ tin cậy cao do có nhiều yếu tố nguy cơ: {', '.join(reasons[:3])}"
                if len(reasons) > 3:
                    explanation += f" và {len(reasons) - 3} yếu tố khác"
            elif len(reasons) > 0:
                explanation = f"Độ tin cậy cao do: {', '.join(reasons)}"
            else:
                explanation = f"Độ tin cậy cao (Risk Score: {risk_score:.1f})"
        elif prob_disease >= 0.5:
            if len(reasons) > 0:
                explanation = f"Có một số yếu tố nguy cơ: {', '.join(reasons[:2])}"
            else:
                explanation = "Có nguy cơ trung bình"
        else:
            if len(reasons) == 0:
                explanation = "Các chỉ số đều trong giới hạn bình thường"
            else:
                explanation = f"Nguy cơ thấp mặc dù có: {', '.join(reasons[:2])}"
        
        return explanation
