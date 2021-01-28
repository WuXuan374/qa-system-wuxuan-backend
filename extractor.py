# 使用正则表达式，从问题、答案中提取响应内容
import re


regExp1 = "\d+年"
regExp2 = "\d+年\d+月"
regExp3 = "\d+年\d+月\d+日"

reg1 = re.compile(regExp1)
reg2 = re.compile(regExp2)
reg3 = re.compile(regExp3)


def extract_date(text):
    result = reg3.findall(text)
    if result:
        date = result
    elif reg2.findall(text):
        date = reg2.findall(text)
    else:
        date = reg1.findall(text)
    return date

