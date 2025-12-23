from flask import Flask, render_template, request, jsonify
import sys
import os

# Add parent directory to path to import model_loader
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from webapp.model_loader import HeartDiseasePredictor

app = Flask(__name__)

# Initialize predictor
predictor = HeartDiseasePredictor()

@app.route('/')
def home():
    """Trang chủ"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint để dự đoán"""
    try:
        # Lấy dữ liệu từ form
        data = request.get_json()
        
        # Validate input
        required_fields = ['age_years', 'gender', 'height', 'weight', 'ap_hi', 
                          'ap_lo', 'cholesterol', 'gluc', 'smoke', 'alco', 'active']
        
        # Kiểm tra thiếu trường
        for field in required_fields:
            if field not in data or data[field] == '':
                return jsonify({'error': f'Vui lòng nhập đầy đủ thông tin: {field}'}), 400
        
        # Validate từng trường
        try:
            age_years = float(data['age_years'])
            if age_years < 18 or age_years > 100:
                return jsonify({'error': 'Tuổi phải từ 18 đến 100'}), 400
                
            gender = int(data['gender'])
            if gender not in [1, 2]:
                return jsonify({'error': 'Giới tính phải là 1 (Nữ) hoặc 2 (Nam)'}), 400
                
            height = float(data['height'])
            if height < 100 or height > 250:
                return jsonify({'error': 'Chiều cao phải từ 100 đến 250 cm'}), 400
                
            weight = float(data['weight'])
            if weight < 30 or weight > 200:
                return jsonify({'error': 'Cân nặng phải từ 30 đến 200 kg'}), 400
                
            ap_hi = float(data['ap_hi'])
            if ap_hi < 80 or ap_hi > 250:
                return jsonify({'error': 'Huyết áp tâm thu phải từ 80 đến 250 mmHg'}), 400
                
            ap_lo = float(data['ap_lo'])
            if ap_lo < 40 or ap_lo > 150:
                return jsonify({'error': 'Huyết áp tâm trương phải từ 40 đến 150 mmHg'}), 400
                
            if ap_lo >= ap_hi:
                return jsonify({'error': 'Huyết áp tâm trương phải nhỏ hơn huyết áp tâm thu'}), 400
                
            cholesterol = int(data['cholesterol'])
            if cholesterol not in [1, 2, 3]:
                return jsonify({'error': 'Cholesterol phải là 1, 2 hoặc 3'}), 400
                
            gluc = int(data['gluc'])
            if gluc not in [1, 2, 3]:
                return jsonify({'error': 'Glucose phải là 1, 2 hoặc 3'}), 400
                
            smoke = int(data['smoke'])
            if smoke not in [0, 1]:
                return jsonify({'error': 'Hút thuốc phải là 0 (Không) hoặc 1 (Có)'}), 400
                
            alco = int(data['alco'])
            if alco not in [0, 1]:
                return jsonify({'error': 'Uống rượu phải là 0 (Không) hoặc 1 (Có)'}), 400
                
            active = int(data['active'])
            if active not in [0, 1]:
                return jsonify({'error': 'Hoạt động thể chất phải là 0 (Không) hoặc 1 (Có)'}), 400
                
        except ValueError:
            return jsonify({'error': 'Dữ liệu không hợp lệ. Vui lòng nhập đúng định dạng số'}), 400
        
        # Dự đoán
        result = predictor.predict(data)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': f'Lỗi hệ thống: {str(e)}'}), 500

@app.route('/health')
def health():
    """Kiểm tra trạng thái hệ thống"""
    return jsonify({
        'status': 'Hoạt động bình thường', 
        'model_loaded': predictor.coefficients is not None,
        'message': 'Model đã sẵn sàng' if predictor.coefficients is not None else 'Model chưa được tải'
    })

if __name__ == '__main__':
    print("=" * 60)
    print("Hệ thống Dự đoán Bệnh Tim")
    print("=" * 60)
    print("Đang khởi động server...")
    print("Truy cập: http://localhost:5000")
    print("Nhấn Ctrl+C để dừng server")
    print("=" * 60)
    app.run(debug=True, host='0.0.0.0', port=5000)
