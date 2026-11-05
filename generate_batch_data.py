import pandas as pd
import random
from generate_data import positive_words, negative_words, neutral_words

data = []
# 模拟 5000 条真实的业务数据
for i in range(5000):
    score = random.random()
    if score < 0.6:
        text = "，".join(random.sample(positive_words, 2))
    elif score < 0.85:
        text = "，".join(random.sample(negative_words, 2))
    else:
        text = "，".join(random.sample(neutral_words, 2))
    
    data.append([f"ORDER_{10000+i}", text])

df = pd.DataFrame(data, columns=["订单号", "评论内容"])
# 故意把表头写成中文，模拟商家的真实业务表
df.to_csv("商家评论数据集.csv", index=False, encoding="utf-8-sig")
print("成功生成 5000 条商家测试数据：商家评论数据集.csv")