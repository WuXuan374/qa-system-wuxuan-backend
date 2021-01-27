# qa-system-wuxuan-backend
## version 2 Date 2021/01/26
### 与version 1 的主要区别
- sentence retrieval 中采用 ngram similarity

### 功能和使用说明
- 在问题和文本的分词过程中，支持ngram
    * 例如ngram=1, 分词结果都是单个单词["吴轩"]
    * ngram=2, 分词结果中会包含二元词组["吴轩作品"]
- 使用说明
    * run_QA: 调用ReadDocumentContent构造函数时，给出ngram参数（默认为1）
    * Evaluation： 调用Evaluation构造函数时，给出ngram参数（默认为1）