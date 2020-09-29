import jieba
import random


def ner(ques):
    if "叫" in str(ques) and str(ques)[str(ques).index("叫") + 1] != "我":
        return ques[ques.index("叫") + 1:]
    elif "是" in str(ques):
        return ques[ques.index("是") + 1:]
    elif "姓名" in str(ques):
        return ques[ques.index("姓名") + 2:]
    elif "名字" in str(ques):
        return ques[ques.index("名字") + 2:]
    elif "叫我" in str(ques):
        return ques[ques.index("叫我") + 2:]
    elif "喊我" in str(ques):
        return ques[ques.index("叫我") + 2:]
    elif "称呼我" in str(ques):
        return ques[ques.index("称呼我") + 3:]


def Levenshtein_Distance(str1, str2):
    """
    计算字符串 str1 和 str2 的编辑距离
    :param str1
    :param str2
    :return:
    """
    matrix = [[i + j for j in range(len(str2) + 1)] for i in range(len(str1) + 1)]

    for i in range(1, len(str1) + 1):
        for j in range(1, len(str2) + 1):
            if (str1[i - 1] == str2[j - 1]):
                d = 0
            else:
                d = 1

            matrix[i][j] = min(matrix[i - 1][j] + 1, matrix[i][j - 1] + 1, matrix[i - 1][j - 1] + d)

    return matrix[len(str1)][len(str2)]


def question_review(ques, sensitive):
    """
    审核用户对话，过滤敏感内容
    :param ques: 用户说的话
    :param sensitive: 敏感词库列表
    :return:
    """
    results = jieba.cut(ques)
    _words = list(results)
    flag = False
    count = 0
    for word in _words:
        if word in sensitive:
            if word in ["台湾", "香港", "澳门", "西藏", "新疆", "共产党"]:
                flag = True
            else:
                count += 1
    if flag and count != 0:
        return random.choice(["我们还是聊点别的吧", "听不大懂耶", "我们聊点别的吧", "听不大懂哎"])
    elif (flag == False) and count != 0:
        return random.choice(["我们还是聊点别的吧", "听不大懂耶", "我们聊点别的吧", "听不大懂哎"])
    elif flag and count == 0:
        return "approved"
    else:
        return "approved"


def load_sensitive():
    """
    加载敏感词库
    :return:
    """
    with open("sensitive/keywords", 'r', encoding="utf-8") as file:
        sensitive_words = file.readlines()
        sensitive_words = [word.strip() for word in sensitive_words]
    return sensitive_words
