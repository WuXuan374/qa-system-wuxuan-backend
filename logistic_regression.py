import torch
from torch import optim
import torch.nn as nn


class Logistic_Regression(nn.Module):
    def __init__(self, n_in, n_out):
        super(Logistic_Regression, self).__init__()
        self.lr = nn.Sequential(nn.Linear(n_in, n_out),
                                nn.Softmax(dim=1))

    def forward(self, inputs):
        pred_prop = self.lr(inputs)
        # dim=1: 按行取最大值
        _, pred_value = torch.max(pred_prop, dim=1)
        return pred_prop, pred_value


if __name__ == "__main__":
    # prepare inputs and outputs data
    # source_file = "./data/TFIDF_input/train_2016_new.json"
    # reader = ReadDocumentContent(source_file, ngram=1)
    # inputs = []
    # outputs = []
    # for question_str in reader.content.keys():
    #     right_answer = reader.content[question_str]["right answer"]
    #     print(right_answer)
    #     predicted_answers = reader.get_question_answer(
    #         question_str, reader.content[question_str]["options"], stop_word_path="./data/stopwords.txt")
    #     for answer in predicted_answers:
    #         if not answer:
    #             continue
    #         inputs.append([answer["first_score"], answer["second_score"]])
    #         if answer["answer"] == "".join(right_answer):
    #             outputs.append(1)
    #         else:
    #             outputs.append(0)
    # torch.save(inputs, "./inputs.pt")
    # torch.save(outputs, "./outputs.pt")
    inputs = torch.load("./Logistic_regression/inputs.pt")
    outputs = torch.load("./Logistic_regression/outputs.pt")
    # [14844, 2]
    inputs = torch.tensor(inputs, dtype=torch.float)
    # [14844]
    outputs = torch.tensor(outputs, dtype=torch.long)
    # 模型参数
    n_in = inputs.size(1)
    n_out = 2
    lr = 0.5
    epochs = 300
    loss_func = nn.CrossEntropyLoss()
    model = Logistic_Regression(n_in, n_out)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 模型训练
    for epoch in range(epochs):
        y_prop, y_pred = model(inputs)
        loss = loss_func(y_prop.float(), outputs)
        print(epoch, loss)
        # 根据loss进行反向传播
        loss.backward()
        # 更新所有的参数
        optimizer.step()
        # 梯度置0
        optimizer.zero_grad()

    torch.save(model.state_dict(), './parameter2.pkl')

    # 模型使用
    # model.load_state_dict(torch.load('./parameter.pkl'))
    # y_prop, y_pred = model(inputs)
    # result = y_pred == outputs
