# Heart Disease Prediction Web App

Ứng dụng web dự đoán nguy cơ bệnh tim sử dụng Machine Learning.

## Cài đặt

```bash
pip install flask numpy
```

## Chạy ứng dụng

```bash
python webapp/app.py
```

Sau đó mở trình duyệt tại: http://localhost:5000

## Hướng dẫn sử dụng

### Nhập thông tin bệnh nhân:

1. **Tuổi**: 18-100 tuổi
2. **Giới tính**: Nữ (1) hoặc Nam (2)
3. **Chiều cao**: 100-250 cm
4. **Cân nặng**: 30-200 kg
5. **Huyết áp tâm thu**: 80-250 mmHg
6. **Huyết áp tâm trương**: 40-150 mmHg (phải nhỏ hơn huyết áp tâm thu)
7. **Cholesterol**: Bình thường (1), Cao (2), Rất cao (3)
8. **Glucose**: Bình thường (1), Cao (2), Rất cao (3)
9. **Hút thuốc**: Không (0) hoặc Có (1)
10. **Uống rượu**: Không (0) hoặc Có (1)
11. **Hoạt động thể chất**: Không (0) hoặc Có (1)

### Kết quả hiển thị:

- **Dự đoán**: Có bệnh tim / Không có bệnh tim
- **Xác suất**: Phần trăm khả năng có/không có bệnh
- **Độ tin cậy**: Mức độ chắc chắn của dự đoán
- **Mức độ rủi ro**: Thấp / Trung bình / Cao / Rất cao
- **BMI**: Chỉ số khối cơ thể
- **Pulse Pressure**: Chênh lệch huyết áp
- **Risk Score**: Điểm rủi ro tổng hợp
- **Giải thích**: Lý do tại sao có kết quả này

## Validation

Ứng dụng có validation đầy đủ:

- Kiểm tra giá trị rỗng
- Kiểm tra kiểu dữ liệu (phải là số)
- Kiểm tra khoảng giá trị hợp lệ
- Kiểm tra logic (huyết áp tâm trương < tâm thu)
- Kiểm tra BMI hợp lý (10-60)
- Hiển thị lỗi rõ ràng

## Cấu trúc

```
webapp/
├── app.py              # Flask backend
├── model_loader.py     # Load model và prediction
├── templates/
│   └── index.html      # Giao diện web
└── README.md           # Hướng dẫn
```

## Model

Model sử dụng Logistic Regression được train trên Cardio dataset với 17 features (11 features cơ bản + 6 engineered features).

Accuracy: ~73%
ROC-AUC: ~0.80
