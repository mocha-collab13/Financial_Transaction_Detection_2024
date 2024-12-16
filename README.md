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

| Clustering Method    | Remaining Samples(m)                   |
| ---------- | ---------------------------------------------- |
| Original Data | 118,800                |
|  KNN Clustering  | 52,709                |
| Mahalanobis + Ledoit-Wolf Clustering | 46,186                |
|  DBSCAN Clustering  | 2,491                |
|  FCM(Fuzzy C-Means)  | 1,200               |

1. KNN Clustering

-> 데이터가 가진 패턴을 이웃 관계르 그룹화하여 불필요한 데이터를 줄이는 방식으로 유사한 데이터 간의 밀집도를 기반으로 데이터 분포를 학습하며, 모델 학습 속도를 높이면서 데이터의 중요한 패턴을 유지하는 장점이 있다.

-> 이에 'm' 클래스의 갯수를 52,709개의 데이터가 유지 되었으며 기존에 비해 30% 이상 개선되었다.

![Clustering_1](https://i.esdrop.com/d/f/roqIf5Zmhy/L0XwRNVL2T.png)

2. Mahalanobis + Ledoit-Wolf

-> Mahalanobis 거리와 Ledoit-Wolf 방법의 결합으로 Mahalanobis 거리는 데이터 간의 분산과 공분산을 반영하여 두 데이터 포인트 간의 유사도를 계산하며, 특히 데이터 간의 분산과 공분산을 반영하여 두 데이터 포인트 간의 유사도를 계산하고, 특히 다차원 데이터에서 효과적이며 이는 데이터의 평균에서 각 데이터가 얼마나 떨어져 있는지, 상관관계를 함께 고려하여 거리를 측정한다.

-> 공분산 행렬 추정에 있어 정확성을 높이기 위해 Ledoit-Wolf 방법을 함께 적용했다. Ledoit-Wolf 방법은 고차원 데이터의 공분산 행렬을 추정할 때 노이즈를 줄여 안정적인 결과를 제공하며, 특히 작은 데이터 샘플로 공분산 행렬을 추정하는 상황에서 유용하다.

-> 데이터 간 유사성 평가를 보다 정밀하게 수행하여 소수 클래스의 특성을 더욱 잘 보존하는 효과가 있으며, 약 46,186개의 데이터로 축소한 결과 KNN 클러스터링 대비 15% 향상되었다.

-> 앞선 KNN 클러스터링과 Mahalanobis + Ledoit-Wolf 방법은 수학적으로 다르지만, 데이터셋의 구조가 복잡하지 않아 클러스터링 결과가 비슷하게 나왔다.

![Clustering_2](https://i.esdrop.com/d/f/roqIf5Zmhy/dhx60HPepj.png)

3. DBSCAN

-> DBSCAN 은 밀도 기반 클러스터링 기법으로 데이터 밀집도를 기반으로 클러스터를 형성하고 밀도가 낮은 영역에 있는 데이터를 노이즈로 간주하여 제거하는 방식이다.

-> 이는 다수 클래스가 고밀도 영역에 존재할 경우 다수와 소수 클래스를 분리하는 데 효과적이며 특히 불필요한 데이터 제거에 유리하다. 본 실험에서 DBSCAN을 실험적으로 적용하여 노이즈가 많은 다수 클래스 데이터를 효과적으로 제거하고자 하였다.

-> DBSCAN은 밀도가 낮은 데이터 포인트를 노이즈로 간주하여 약 2,491개로 데이터가 축소되었으며, 이로 인해 모델의 학습 속도는 50% 이상 향상되었다. 그러나 노이즈 제거 과정에서 일부 소수 클래스 데이터가 손실되었음을 관찰하여, 이를 보완하기 위해 FCM(Fuzzy C-Means) 클러스터링을 진행하였다.

![Clustering_3](https://i.esdrop.com/d/f/roqIf5Zmhy/Zc6Iwu1gcA.png)

4. FCM(Fuzzy C-Means)

-> FCM(Fuzzy C-Means) 은 기존의 K-Means 클러스터링과는 달리 각 데이터 포인트가 다수의 클러스터에 동시에 소속될 수 있는 소속도(membership degree)를 가진다. 이를 통해 데이터 간의 유사도를 세밀하게 반영하여 소수 클래스 예측 성능을 강화할 수 있으며, FCM은 데이터의 불확실성과 다중 클러스터 소속도를 기반으로 유연한 클러스터링을 수행하는 특징이 있다.

-> FCM을 적용하여 데이터를 다양한 클러스터레 중복 소속시켜, 다수 클래스와 소수 클래스 간의 구조적 차이를 세밀하게 표현하였으며, 데이터 크기를 1200개로 줄ㄹ여 소수 클래스의 특성을 정교하게 보존할 수 있었다. 기존 DBSCAN 보다 도욱 향상된 소수 클래스 예측 성능을 보여주었고, 모델 학습에서 높은 Macro F1-score와 AUC를 기록하였다. 

![Clustering_4](https://i.esdrop.com/d/f/roqIf5Zmhy/wxnfpW44K7.png)

-> 이 외에도 Z-Score를 활용하여 이상치를 탐지하고 제거했으며, 결측값은 적절한 방법(예: 평균 대체, 회귀 기반 예측 등)을 사용해 처리했습니다. 또한, 거래 시간, 금액 등 다양한 특성을 고려하여 Feature Engineering을 수행해 데이터셋을 모델 학습에 최적화했습니다.

**[모델]**

이상금융거래를 탐지하기 위해 **CTGAN**을 활용해 데이터 증강을 진행하고, **XGBoost 모델**을 사용하여 분류 모델을 구축했습니다. 다양한 머신러닝 모델을 비교 분석하고 가장 적합한 모델을 선정하여 훈련했으며 모델의 하이퍼파라미터를 최적화하기 위해 **Optuna**를 사용했습니다.

- 데이터 증강
  
    - 생성 AI 모델 관련 인증 기준 (VAE(M), TableGAN, CTGAN, CTAB-GAN 등 만을 허용)
      
      - 이 중 CTGAN이 제일 성능이 좋았기에 CTGAN으로 선정함
      
    - **CTGAN**을 이용하여 'a ~ l' 의 데이터는 1000개씩 증강시켰으며 'm'의 데이터는 기존 FCM Clustering을 통해 1200개로 만든 상태에서 1000개를 선택하게 하여, 모든 Fraud_Type의 갯수를 맞춰주었다.

![CTGAN](https://i.esdrop.com/d/f/roqIf5Zmhy/WUH324ihb5.png)

- 분류 모델 구축
  - XGBoost, LightGBM, Catboost 등 다양한 분류 모델을 사용해보았고, 앙상블 과정을 통해서도 결과를 얻어봤지만 이 중 XGBoost 모델이 제일 성능이 좋아 이에 모델의 하이퍼파라미터를 최적화하기 위해 **Optuna**를 사용하였다.
    - ![XGBoost_Optuna](https://i.esdrop.com/d/f/roqIf5Zmhy/BnACLb87oU.png)
    - Best hyperparameters: {'learning_rate': 0.03500842897354163, 'n_estimators': 224, 'max_depth': 4, 'min_child_weight': 2.1605378743201147, 'gamma': 0.22804306127508525, 'subsample': 0.955950194857886, 'colsample_bytree': 0.8227035006806621, 'reg_lambda': 0.3490933288294228, 'reg_alpha': 0.21227891390474513, 'scale_pos_weight': 4.547887415904099, 'max_delta_step': 1.588355137064108} 최적의 하이퍼파라미터로 학습을 시켰습니다.
   - 얻은 결과물에서 'm'을 제외한 다른 데이터에서의 상황을 보았는데 'i' 데이터가 없는 것을 인지하였다.
     !['i'의 데이터 손실](https://i.esdrop.com/d/f/roqIf5Zmhy/S9Yj40cibk.png)
     - 여기서 'i' 를 어떻게 분류할 수 있을까 고민을 하다가 'i' 만 분류하는 모델을 만들어 기존의 모델과 앙상블을 하면 어떨까 하는 아이디어를 내었습니다. 그리하여 기존의 모델과 'i' 모델의 앙상블을 하였고, 모델의 성능을 평가하기 위해 **Macro F1-Score**, **TCAP**의 지표를 활용했습니다. 이를 통해 모델의 공정성과 정확도를 평가하고 모델 성능을 75%까지 향상시키며 금융 거래 이상 탐지 문제를 해결할 수 있는 모델을 개발했습니다.


