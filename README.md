# National Pathology Health Datathon 2021

- [대회 Link](http://nphd2021.co.kr/)

## 팀소개

- 황원상
- 홍요한
- 박준혁
- 박범수

## 대회 소개

- 디지털 병리 이미지 분류하는 딥러닝 모델 개발
- 소화기 병리 이미지를 활용하여 암세포인지 정상 세포인지 Classification Task
- 대회 기간 : 2021.11.18 ~ 2021.11.19 (무박 2일)
- Leader Board는 대회 중간에 1회 확인가능
- Model 크기 제한 300MB

## Data 소개

![image](https://user-images.githubusercontent.com/77658029/142766180-a26e61b5-9b9a-4fba-b8b5-2f864306c00e.png)

- 특징 : 핵의 비정상적인 단백질(DNA)분열로 인하여 세포핵이 커지며고 세포질이 작아지는 형상으로 나타남

## 평가 방법

- 지표 6가지의 평균을 평가 지표로 활용

![image](https://user-images.githubusercontent.com/77658029/142765905-21112d9d-30d5-46e2-8ab7-ad5b7d9de71c.png)

## 활동 내용

0. EDA 진행
    - 하얗게 반이상 빈공간으로 이루어진 Image 존재
    - 대부분은 보라색 or 붉은 색으로 이루어짐
    - 비정형적인 모습(세포질이 무너진 형태)가 많음

1. K-Fold를 활용한 Validation Set 구축 (8:2)

2. 과적합을 피하기 위한 작은 Model 시도(EfficientNet, MobileNet, Non bottleNeck 1D)

![image](https://user-images.githubusercontent.com/77658029/142766726-7f904f13-23ab-4c5a-ac0c-20721858ceff.png)

- NB1D(Non bottleNeck 1D) : Model 용량은 작지만 간혹 튀는 현상이 있어 사용 어려움

3. Augmentation 선정

- 심하게 왜곡되는 Augmentation 제외
- 중간에 비어있는 공간들이 있어 Cutout 추가(제거된 공간은 실제 빈공간처럼 White로 입력)
- 소화기의 조직세포이기 때문에 큰 Image의 일부만 떼어온 Image의 형상
    - Rotate, RandomResizedCrop, ShiftScaleRotate, HorizontalFlip, VerticalFlip를 통하여 여러 지역에서 찍은 효과를 줌
- H&E 염색으로 인한 보라색 빛이 돌지만, 탈색이 덜된 곳은 양성/음성에 관계없이 붉은 빛을 도는 세포들도 존재 - HueSaturationValue로 색을 조정하여 양성/음성에서의 색에 대한 부분도 학습을 시켜줌
- 비정형적인 세포의 모습을 전반적으로 학습하기 위해 ElasticTransform를 넣어줌
- 모델의 과적합이 쉽게 일어나는 것으로 생각되어 Epoch을 줄이고 전반적으로 높은 확률로 Augmentation을 적용시킴

    ```python
    train_transform = A.Compose([
                    A.Cutout(num_holes=8, max_h_size=8, max_w_size=8, fill_value=1, always_apply=False, p=0.05),
                    A.Rotate(always_apply=False, p=1.0, limit=(-90, 90), interpolation=0, border_mode=0, value=(255, 255, 255), mask_value=None),
                    A.RandomResizedCrop(always_apply=False, p=0.1, height=512, width=512, scale=(0.5, 1.0), ratio=(0.75, 1.3333333730697632), interpolation=0),
                    A.HorizontalFlip(p=0.1),
                    A.VerticalFlip(p=0.1),
                    A.HueSaturationValue(always_apply=False, p=0.2, hue_shift_limit=(-20, 20), sat_shift_limit=(-30, 30), val_shift_limit=(-20, 20)),
                    A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=10, border_mode=4, p=0.1),
                    A.ElasticTransform(p=0.2),
                    ToTensorV2()
    ])
    ```
4. ensemble

- 각각의 모델은 대부분 좋은 효과를 보여주었지만, 성능이 비슷한 Model에서도 다른 결과를 찍는 것을 확인함
- Model의 안정성을 높이기 위해서 여러 Model과 Kfold의 특징을 담을 필요가 있음
- EfficientNet(Kfold 2,3,4,5), MixNet(Kfold 1,2,5) Ensemble 진행

5. Result



## Reference

<p><span style="background-color:#EEEEEE;">NPHD2021 - 소화기 병리 / CC BY 2.0<br/>
http://nphd2021.co.kr/
</span></p>

- 대장암 종양 분류를 위한 딥러닝 모델 연구([https://eochodevlog.tistory.com/76](https://eochodevlog.tistory.com/76))
- 다단계 Seg-Unet 모델을 이용한 방사선 사진에서의 End-to-end 골 종양 분할 및 분류([https://www.kci.go.kr/kciportal/ci/sereArticleSearch/ciSereArtiView.kci?sereArticleSearchBean.artiId=ART002557254](https://www.kci.go.kr/kciportal/ci/sereArticleSearch/ciSereArtiView.kci?sereArticleSearchBean.artiId=ART002557254)**)**
- 교모세포종 환자의 영역 분할과 암 분류를 위한 듀얼태스크 심층신경망 모델([http://koreascience.or.kr/article/CFKO202121751079208.pdf](http://koreascience.or.kr/article/CFKO202121751079208.pdf))
- 딥러닝 기반 암세포 사진 분류 알고리즘([https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE07540025](https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE07540025))
- CNN 기반 세포 이미지 분류 알고리즘에 관한 연구([https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE10490504](https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE10490504))



