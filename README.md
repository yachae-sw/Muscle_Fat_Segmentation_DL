# Muscle & Fat Segmentation Deep Learning Model

**근육 및 지방 분할 딥러닝 모델 개발 프로젝트**

## 프로젝트 소개

- **프로젝트 목표 :** 추출된 세 번째 요추 데이터에서 근육과 배경으로 이미지 분할 딥러닝 모델 개발
- **수행 기간 :** 2023.06 ~ 2022.09 (약 4개월)
- **팀 구성 :** 공동 연구(3명 진행)
- **담당역할 :**  CT 데이터 전처리 및  딥러닝 모델 훈련 담당

## 데이터세트
- L3 슬라이스 검출을 위하여 전립선암 환자 104명, 방광암 환자 46명의 데이터가 사용되었다.
- 총 150명의 데이터 모두 횡단면 CT 영상으로 구성되었다.
- 본 데이터는 강릉아산병원에서 수집되었다
- 비뇨의학과 임상의가 ITK-SNAP v3.8 소프트웨어를 사용하여 L3 슬라이스를 수동으로 분할하였으며, 학습모델
의 출력 정보로 활용되었다.

| Dataset |
|---|
| ![segmentation_dataset](https://github.com/yachae-sw/Muscle_Fat_Segmentation_DL/assets/93850398/33916055-3288-403c-b1ef-9b41ee1e4ee5) |

## 프로젝트 진행 과정

1. 방광암 및 전립선암 환자 150명 CT data 및 Label data 전처리
    - 세 번째 요추 CT 영상을 딥러닝 모델에 적용 가능한 형태로 변환하기 위해 Hounsfield Unit (HU) 값을 [-200 800] 범위로 조절하고 **PNG 형식**으로 변환합니다.
    - 전문가가 근육 부분을 1로 레이블링하기 위해 ITK-SNAP 프로그램을 사용하여 Mask 데이터를 생성합니다.
2. 이미지 분할 딥러닝 모델을 사용하여 근육 및 지방 부분 예측
    - U-Net 모델을 사용하여 의료 이미지 분할을 수행합니다. 모델은 피하지방을 1, 근육을 2, 내장지방을 3으로 예측하고 나머지 부분을 0으로 예측하도록 훈련됩니다.

3. 베이지안 최적화로 최적의 파라미터 값 추출
    - 최적의 데이터 비율과 클래스 가중치를 찾기 위해 **베이지안 최적화 방법**을 사용합니다.
    - 프로젝트에서 제안한 방법과 최적화를 적용하지 않은 모델의 결과를 비교하여 모델을 평가합니다.

4. 근육, 피하지방 그리고 내장지방 예측결과 시각화

| 연구 과정 |
|---|
| ![segmentation_절차](https://github.com/yachae-sw/Muscle_Fat_Segmentation_DL/assets/93850398/e72f7fb9-6eb4-4f39-b48d-cdbbec451ba2) |

## 딥러닝 모델

| U-Net |
|---|
| ![unet](https://github.com/yachae-sw/Muscle_Fat_Segmentation_DL/assets/93850398/14278705-5e79-449d-99bd-ff13fe67d0a9) |

## 프로젝트 결과

- - background는 0으로, 근육은 1, 피하지방은 2, 내장지방은 3으로 지정하여 다중 분할하도록 딥러닝 모델을 훈련하였습니다. 입력된 이미지는 척추뼈 수준의 2D 슬라이스 중 L3 부분만 사용하였습니다.
- U-Net 모델을 훈련시켜 근육을 분할한 결과 기존 모델은 Dice Score가 평균 0.92인 반면, 최적화된 모델의 Dice Score는 평균 0.96으로 더 정확한 이미지 분할을 달성하였습니다.
- 최종적으로, 연구 결과는 기존 모델 대비 최적화된 모델이 더 **효과적인 성능**을 보여주었으며 작업시간도 1~2초로 매우 단축했습니다.

![segmentation_result](https://github.com/yachae-sw/Muscle_Fat_Segmentation_DL/assets/93850398/5216779d-32f8-4e73-b741-2b17a7bf48c1)

## 프로젝트 후기

- 프로젝트를 통해 이미지 분할 딥러닝 모델에 대해 공부할 수 있었으며 **분류 모델과의 유사성**도 이해할 수 있었습니다.
- 다양한 클래스로 나누는 모델을 높은 성능이 나올 수 있게 훈련시켜 보면서 다양한 문제가 발생하더라도 **다양한 해결방법**이 있음을 알게 되었습니다.
- 암의 대표적인 예후 인자인 근감소증과 같은 신체 구성의 변화를 빠르게 예측하여 암 진단 및 치료에 사용될 것입니다.
- 영양 상태를 평가하고 비만 및 대사질환, 고혈압과 같은 심혈관 질환 등과 관련된 문제를 해결하기 위해 일반적으로 사용될 수 있습니다.