# Financial_Transaction_Detection_2024

## overview
FSI AIxData Challenge 2024에서 이상금융거래(FDS) 탐지를 위한 AI 모델을 구축하는 것을 목표로 진행되었습니다. 본 프로젝트에서는 생성형 AI 모델인 CTGAN과 머신러닝 모델인 XGBoost를 활용하여 금융 거래에서 발생할 수 있는 이상 거래를 탐지하는 모델을 개발했습니다. 대회 초기에 데이터 클래스 불균형 문제와 같은 어려움에 직면했지만 데이터 증강, 샘플링, 축소 기법을 적용하여 이러한 문제를 해결했습니다. 그 결과 모델의 성능을 Macro F1-Score 기준으로 75%까지 향상시킬 수 있었으며, 실제 금융 보안 분야에서의 AI 활용 가능성을 더욱 높일 수 있었습니다. 이번 경험을 통해 실제 산업에서 발생할 수 있는 문제를 해결하는 능력을 키우며, 데이터 분석과 AI 모델링의 실전 역량을 한층 강화할 수 있었습니다.

### <기여한 부분만>

**[데이터 전처리]**

 1. 데이터 형태를 보기 위해 진행

  ![전처리_1](https://i.esdrop.com/d/f/roqIf5Zmhy/bUDWaGXAnL.png)

1-1. 범주형 데이터엔 어떤 것들이 있는지 확인

 ![전처리_2](https://i.esdrop.com/d/f/roqIf5Zmhy/XKg1ItSSfW.png)

1-2. 수치형 데이터엔 어떤 것들이 있는 지 확인

 ![전처리_3](https://i.esdrop.com/d/f/roqIf5Zmhy/ut0tkWMqSm.png)
 ![전처리_4](https://i.esdrop.com/d/f/roqIf5Zmhy/IR9cXEyn1D.png)

 - Account_initial_balance : 음수 값 존재
- Account_balance : 음수 값 존재
- Transaction_Amount : 음수 값 존재
- Time_difference : 음수 값 존재
  
  : 실존하는 개인정보가 아닌 임의로 랜덤 생성해서 음수값이 존재
  
  : Transaction_Amount 변수를 제외하고 음수인 데이터는 소수

2. 범주형 변수

![전처리_5](https://i.esdrop.com/d/f/roqIf5Zmhy/OHHGPy212o.png)

- 전반적으로 변수들이 불균형
- Error_Code : 'b', 'd', 'e', 'f' 데이터가 훈련데이터셋에 존재 X
- Access_Medium : 'h' 데이터가 훈련데이터셋에 존재 X
- Another_Person_Account : 0 데이터가 훈련데이터셋에 존재 X

3. 수치형 변수

- 대부분 변수들이 극단값을 지님 (IQR 기준)
  
![결과물_1](https://i.esdrop.com/d/f/roqIf5Zmhy/iO3Lvo8fdP.png)
![결과물_2](https://i.esdrop.com/d/f/roqIf5Zmhy/5nbuoRb3Af.png)
![결과물_3](https://i.esdrop.com/d/f/roqIf5Zmhy/C9jdbABBmH.png)
![결과물_4](https://i.esdrop.com/d/f/roqIf5Zmhy/UMNHJ3HTZc.png)

4. Fraud_Type

![전처리_6](https://i.esdrop.com/d/f/roqIf5Zmhy/rkrZlgkrRs.png)


| 사기유형    | 갯수                                  |
| ---------- | ---------------------------------------------- |
| a ~ l | 각 100개                |
|  m  | 118,800개                |

- 데이터 불균형은 모델의 성능에 영향을 미침
  
  - 다수 클래스에 편향되어 소수 클래스를 제대로 분류 X
    
  - 전체 정확도는 높을 수 있지만, 소수 클래스에 대한 재현율(recall) 낮아짐
    
  - 새로운 데이터에 대한 예측 성능 저하


------------------------------------------------
 
**이에 대한 해결책으로 'm' 사기 유형을 언더샘플링 하여 소수 클래스들에 맞추고자 진행함**

**[Clustering]**
