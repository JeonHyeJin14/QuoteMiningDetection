# A Contrastive Learning Framework for Detecting Contextomized News Quotes (QuoteCSE)

## Task: Contextomized Quote Detection
Let a given news article be *X:(T, B)*, where *T* is the news title, and *B* is the body text. Our task is to predict a binary label *Y* indicating whether the headline quote in *T* is either modified or contextomized by referring to the body-text quotes.

## Method: QuoteCSE

We present QuoteCSE, a contrastive learning framework that represents the embedding of news quotes. In addition, we implemented a classifier to classify whether the title quote *T* is contextomized or not using embedding obtained by QuoteCSE. 
![image](https://user-images.githubusercontent.com/80903024/217503832-ef78155c-f4bc-4b51-bcee-5a32e02ddf60.png)
QuoteCSE is a contrastive quote embedding framework that is designed based on journalism ethics. This figure illustrates the central idea of QuoteCSE. QuoteCSE maximizes the semantic similarity between the headline quote and the matched quote in the body text while minimizing the similarity for other unmatched quotes in the same or other articles.

We obtain embeddings of headline quote and body quotes from QuoteCSE. The headline quote embedding is **u**, and the body quote embedding most similar to the **u** is **v**. To detect the contextomized quote, We implemented a binary MLP classifier with **u**, **v**, **|u-v|**, **u*v** as input.


## Datasets

### Pretraining corpus
```
data/modified_sample.pkl
data/verbatim_sample.pkl
```
We present a sampled dataset of unlabeled corpus used for the QuoteCSE pretraining. Each data instance consists of title quote, positive sample, and negative sample. The positive and negative samples were selected by SentenceBERT, and the assignments are updated during training.

### Contextomized quote detection
```
data/contextomized_quote.pkl
```
We introduce a dataset of 1,600 news articles for detecting contextomized news quotes.
- Label 1: The headline quote is *contextomized*, which refers to the excerpt of words with semantic changes from the original statement.
- Label 0: The headline quote is *modified*. It keeps the semantics of original expression but it is a different phrase or sentence.

**Examples**
|headline quote|body quotes|label|
|------|---|:---:|
|"이대론 그리스처럼 파탄"(A debt crisis, like Greece, is on the horizon)|건강할 때 재정을 지키지 못하면 그리스처럼 될 수도 있다"(If we do not maintain our fiscal health, we may end up like Greece) <br/> "강력한 ‘지출 구조조정’을 통해 허투루 쓰이는 예산을 아껴 필요한 곳에 투입해야 한다"(Wasted budgets should be reallocated to areas in need through the reconstruction of public expenditure)|Contextomized <br/> (1)|
|"불필요한 모임 일절 자제"(Avoid unnecessary gatherings altogether)|"저도 백신을 맞고 해서 여름에 어디 여행이라도 한번 갈 계획을 했었는데..."(Since being vaccinated, I had planned to travel somewhere in the summer, but...) <br/> "어떤 행위는 금지하고 어떤 행위는 허용한다는 개념이 아니라 불필요한 모임과 약속, 외출을 일제 자제하고…."(It is not a matter of prohibiting or permitting specific activities, but of avoiding unnecessary gatherings, appointments, and going out altogether...)|Modified <br/> (0)|


## Usage

### QuoteCSE pretraining
```python
python train.py 
```
You can obtain the pretrained QuoteCSE checkpoints [here](https://drive.google.com/drive/folders/1F-e4JXWI_EXpybJ86SfclCg9SV-mci1N?usp=sharing).

### QuoteCSE-based detection
```python
python contextomized_quote_detection.py 
```

## Reference

For more details and background about the task and method, please check [our paper](https://aclanthology.org/2023.findings-eacl.52.pdf).

```bibtex
@inproceedings{song2023detecting,
  title={Detecting Contextomized Quotes in News Headlines by Contrastive Learning},
  author={Song, Seonyeong and Song, Hyeonho and Park, Kunwoo and Han, Jiyoung and Cha, Meeyoung},
  booktitle={Findings of the Association for Computational Linguistics: EACL 2023},
  pages={685--692},
  year={2023}
}
```

[Download Pre-trained Model] 모델을 실행하려면 가중치 파일이 필요합니다. 아래 링크에서 다운로드 받아 프로젝트 폴더에 넣어주세요. 👉 [https://drive.google.com/file/d/1jlv7PBu06YOY83SAx1gDq2WTWf9cB1nq/view?usp=drive_link]
