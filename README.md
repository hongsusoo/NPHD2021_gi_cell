# National Pathology Health Datathon 2021 

- [대회 Link](http://nphd2021.co.kr/)

 <center><img src="https://user-images.githubusercontent.com/77658029/143607271-7deb6023-a48a-4e92-b722-3618f8e48034.png"  width="90%" height="90%"/></center>

## 팀소개 🙋‍♂️

### 팀명뭘로하조A

<table>
  <tr>
    <td align="center">
      <a href="https://github.com/hanlyang0522">
        <img src="https://avatars.githubusercontent.com/u/67934041?v=4" width="150px;" alt=""/>
        <br />
        <sub>박범수</sub>
    <td align="center">
      <a href="https://github.com/WonsangHwang">
        <img src="https://avatars.githubusercontent.com/u/49892621?v=4" width="150px;" alt=""/>
        <br />
        <sub>황원상</sub>
      </a>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/hongsusoo">
        <img src="https://avatars.githubusercontent.com/u/77658029?v=4" width="150px;" alt=""/>
        <br />
        <sub>홍요한</sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/Junhyuk93">
        <img src="https://avatars.githubusercontent.com/u/61610411?v=4" width="150px;" alt=""/>
        <br />
        <sub>박준혁</sub>
      </a>
  </tr>
  <tr>
    </td>
  </tr>
</table>
<br>

## 대회 소개

- 암조직(세포)과 암이 아닌 조직(세포)으로 구성된 병리이미지를 효율적으로 분류하는 딥러닝 모델 개발
- 대회 기간 : 2021.11.18 ~ 2021.11.19 (무박 2일로 진행하는 해커톤)
- Leader Board는 대회 중간에 2일중 정해진 시간에 1회 확인가능
- Model 크기 제한 300MB

## Data 소개

 <center><img src="https://images.velog.io/images/hanlyang0522/post/cb47523c-4529-4da2-8e3a-21073cd7940d/image.png" width="90%" height="90%"/></center>
 
*<center>보안상 데이터를 공개할 수 없기 때문에, 관련 이미지로 대체함</center>*
 
- 핵의 비정상적인 단백질(DNA)분열로 인하여 세포핵이 커지며 세포질이 작아지는 형상으로 나타남
- 512 x 512 png/jpg positive 5000장, negative 5000장 (각 20%는 test data)

## 평가 방법

### 제한 사항 
- 모델 용량 300MB 이하
- 제공 플랫폼 내 러닝타임 2시간 초과하면 실격
- Pre-trained model 사용가능

### 평가 지표 - 아래 지표 6가지의 평균으로 순위를 결정
- Accuracy = ( TP + TN ) / ALL
- Specificity = TN / ( TN + FP )
- Sensitivity ( Recall ) = TP / ( TP + FN )
- Precision = TP / ( TP + FP )
- Negative predicable value = TN / ( TN + FN )
- F1 score = (2 x ( Precision x Recall ) / ( Precision + Recall )
- 동일 정확도를 보일 경우 모델 크기가 작은 팀이 우선
## 프로젝트 내용

### 1. EDA

- 반 이상이 하얗게 빈 공간(으로 이루어진 Image 존재
- 대부분은 보라색/붉은 색으로 이루어짐
- 비정형적인 모습(세포질이 무너진 형태)가 많음
- 눈으로 음성/양성 판정이 어려움

### 2. K-Fold

- Stratified K-Fold를 활용한 Validation Set 구축 (8:2 비율)

### 3. Model 선정
- 과적합을 피하기 위한 작은 Model 시도(EfficientNet, MixNet, Non bottleNeck 1D)
    - 대회 규정 상 총 모델 용량 제한(300MB)이 있어, 가벼운 모델들을 탐색
    - 앙상블 할 것을 고려하여, 30 MB 내외이면서 테스트시 성능이 좋은 모델을 선택
    - 비슷한 이미지들이 많아 뽑아야할 특징이 많지 않을 것이라고 판단
    - 모델별 테스트 결과
    <img src="https://user-images.githubusercontent.com/77658029/145201861-7775425b-ca12-4d16-af64-f5efce04080d.png"  width="70%" height="70%"/>
        - EfficientNet-b0, MixNet을 선정
        - NB1D(Non bottleNeck 1D)는 모델별 초기 테스트시 안정적인 학습이 되지 않는 현상이 발견되어 후보에서 제외함 
        
### 4. Augmentation

- 이미지를 심하게 왜곡하는 Augmentation은 제외함
- 중간에 비어있는 공간들이 있어 Cutout 추가(제거된 공간은 실제 빈공간처럼 White로 입력)
- 소화기의 조직세포이기 때문에 큰 Image의 일부만 떼어온 Image의 형상
- Rotate, RandomResizedCrop, ShiftScaleRotate, HorizontalFlip, VerticalFlip를 통하여 소화기의 다양한 부분에서 뽑은 효과를 줌
- H&E 염색으로 인한 보라색 빛이 돌지만, 염색이 덜된 곳은 양성/음성에 관계없이 붉은 빛을 도는 세포들도 존재해 HueSaturationValue을 활용해 색을 조정하고 양성/음성에서의 색에 대한 부분도 학습을 진행함
- 비정형적인 세포의 모습을 전반적으로 학습하기 위해 ElasticTransform를 활용함
- 모델의 과적합이 쉽게 일어나는 것으로 생각되어 Epoch을 줄이고 전반적으로 높은 확률로 Augmentation을 적용시킴
    
### 5. Ensemble

- 각각의 모델은 대부분 좋은 효과를 보여주었지만, 성능이 비슷한 Model에서도 다른 결과를 찍는 것을 확인함
- Model의 안정성을 높이기 위해서 여러 Model과 K-fold의 특징을 담을 필요가 있음
- EfficientNet (K-fold 2,3,4,5번의 data set), MixNet(K-fold 1,2,5번의 data set) hard-voting ensemble 진행

## 대회 결과 🏆

- Final score : 0.9850 (**1등**/15팀) 🥇


## 팀원들의 한마디 ✨

박준혁 : 
홍요한 : 
박범수 : 
황원상 : 

## Reference

<p><span style="background-color:#EEEEEE;">NPHD2021 - 소화기 병리 / CC BY 2.0<br/>
http://nphd2021.co.kr/
</span></p>

- 대장암 종양 분류를 위한 딥러닝 모델 연구([https://eochodevlog.tistory.com/76](https://eochodevlog.tistory.com/76))
- 다단계 Seg-Unet 모델을 이용한 방사선 사진에서의 End-to-end 골 종양 분할 및 분류([https://www.kci.go.kr/kciportal/ci/sereArticleSearch/ciSereArtiView.kci?sereArticleSearchBean.artiId=ART002557254](https://www.kci.go.kr/kciportal/ci/sereArticleSearch/ciSereArtiView.kci?sereArticleSearchBean.artiId=ART002557254)**)**
- 교모세포종 환자의 영역 분할과 암 분류를 위한 듀얼태스크 심층신경망 모델([http://koreascience.or.kr/article/CFKO202121751079208.pdf](http://koreascience.or.kr/article/CFKO202121751079208.pdf))
- 딥러닝 기반 암세포 사진 분류 알고리즘([https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE07540025](https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE07540025))
- CNN 기반 세포 이미지 분류 알고리즘에 관한 연구([https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE10490504](https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE10490504))
- Data 소개 관련 이미지 출처(http://www.doctorstimes.com/news/articleView.html?idxno=150488)