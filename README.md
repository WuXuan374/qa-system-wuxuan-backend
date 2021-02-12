# qa-system-wuxuan-backend
## version 8 
### 重要改动
- 在初步检索之后，进行重排序
- 重排序基于ngram-matching, 见论文P32
- alpha1* 初步检索 + alpha2 * 得分重排序
- alpha1 和 alpha2 通过Logistic Regression得到 