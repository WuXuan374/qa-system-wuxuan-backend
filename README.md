# qa-system-wuxuan-backend
## version1: compute TFIDF Date: 2021/01/26
### 流程描述：
- 用户提出一个问题A
- 提取问题A的关键词
- 通过关键词匹配的方式，查找语料库中相似的问题B(可以有多个)
- 问题B对应的文章(多个）就是我们所需要检索的文本(Document/Passage)
- 问题A对应一个question vector, 基于TFIDF计算出question distance; 文本中的每个选项同样计算出一个distance, 以及这两个distance 计算cosine similarity, 对每个候选答案进行评分（每个候选答案是一个句子、段落），根据评分对所有答案排序