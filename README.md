# qa-system-wuxuan-backend
## version 5 
### 思路
- 将问题和答案利用预训练的word embedding 进行表示
- 计算问题和答案的embedding之间的cosine similarity, 根据相似度对答案进行排序

### 实验结果
- 准确率和MRR比之前基于TFIDF的检索要差不少

### 下一步思路
- 再试试Document Embedding/ Sentence Embedding?
- 基于CNN模型进行问答
