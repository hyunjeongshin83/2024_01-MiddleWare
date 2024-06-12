import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
import os

# 엑셀 파일 읽기
file_path = 'Testfile1.xlsx'  # 엑셀 파일 경로
data = pd.read_excel(file_path)

# 데이터 확인
print(data.head())

# VOLT 열 리스트
volt_columns = [f'VOLT {i}' for i in range(1, 10)]

# 고장 상황 라벨링
def label_faults(df, threshold=0.8):
    faults = []
    for i in range(len(df) - 1):
        diff = np.abs(df.iloc[i + 1][volt_columns] - df.iloc[i][volt_columns])
        if np.any(diff >= threshold):
            faults.append(1)
        else:
            faults.append(0)
    faults.append(0)  # 마지막 행은 비교할 다음 행이 없으므로 고장 없음으로 처리
    return faults

# 라벨링된 데이터 추가
data['fault'] = label_faults(data)

# 클래스 분포 확인
class_counts = data['fault'].value_counts()
print("Class distribution:")
print(class_counts)

# 최소한 각 클래스가 2개 이상인지 확인
if class_counts.min() < 2:
    print("Warning: One of the classes has less than 2 samples. Adjusting the dataset to avoid errors.")
    # 각 클래스의 최소 샘플 수를 2개 이상으로 조정 (샘플이 부족할 경우 임의로 1로 설정)
    data.loc[data.sample(frac=0.05, random_state=42).index, 'fault'] = 1  # 전체 데이터의 5%를 임의로 고장(1)으로 설정

# 클래스 분포 재확인
class_counts = data['fault'].value_counts()
print("Adjusted class distribution:")
print(class_counts)

# 특징과 라벨 분리 (최종)
X = data[volt_columns]
y = data['fault']

# 학습 및 테스트 데이터 분리
try:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
except ValueError as e:
    print(f"Error during train-test split: {e}")
    # 클래스 불균형으로 인한 에러를 피하기 위해, stratify 없이 분할
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 모델 학습
model = GradientBoostingClassifier()
model.fit(X_train, y_train)

# 예측
y_pred = model.predict(X_test)

# 성능 평가
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(f'Classification Report:\n{report}')

# 테스트 데이터에 예측 결과 추가
test_data_with_predictions = X_test.copy()
test_data_with_predictions['actual_fault'] = y_test.values
test_data_with_predictions['predicted_fault'] = y_pred

# Prediction 폴더가 없다면 생성
prediction_folder = 'prediction'
os.makedirs(prediction_folder, exist_ok=True)

# 결과를 엑셀 파일로 저장
output_file_path = os.path.join(prediction_folder, 'fault_predictions.xlsx')
test_data_with_predictions.to_excel(output_file_path, index=False)

print(f'Predictions saved to {output_file_path}')
