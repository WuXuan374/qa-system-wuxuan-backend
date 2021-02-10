# qa-system-wuxuan-backend
## version 7 
### 重要改动
- 基于nlpcc 2016 test data, 进行模型评估和问答
- 修改了数据集后，系统准确率大幅提升，达到了MRR 62%, Accuracy 71%.
- 基于sklearn 的tfidfVectorizer 和 scipy的distance.cosine完成了问题的相似度计算
  感悟：这两者对数据格式有一定要求（需要预处理），但用起来较方便。

### 整体思路
- Question Retrieval: 用户输入问题A, 通过计算与语料库中问题B的cosine similarity，对相似问题进行筛选
- Sentence Retrieval: 问题A与问题B对应的选项之间计算cosine similarity, 对答案进行排序筛选
- 返回top3 答案给用户
