# 汽车行业用户观点主题及情感识别
## 基本思路：  
由于一个 content可能对应多个主题，每个主题有对应的情感值。将主题和情感单独考虑是行不通的，还考虑过将问题分成10个分类器，每个分类器代表一个主题，并做4分类问题（未提及、负向、中立、正向），效果也不是很好。最后考虑将主题和情感结合，构成10*3 分三十个标签，再将题目建模成一个多标签问题。
**使用的词向量**:  
https://github.com/Embedding/Chinese-Word-Vectors
**实现参考的baseline**: 
https://raw.githubusercontent.com/312shan/Subject-and-Sentiment-Analysis/master/README.md
**使用的模型**
text_CNN, bi_gru等