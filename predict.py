import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# 1. 加载数据并训练
df = pd.read_csv("comments.csv", encoding="utf-8-sig")
vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(1, 2))
X = vectorizer.fit_transform(df["评论内容"])
y = df["情感标签"]

model = MultinomialNB()
model.fit(X, y)

# 2. 预测函数
def analyze_new_comment(comment_text):
    vec = vectorizer.transform([comment_text])
    prediction = model.predict(vec)[0]
    probabilities = model.predict_proba(vec)[0]
    
    # 映射结果：0=负面, 1=中评, 2=正面
    result_map = {0: "负面 👎", 1: "中评 😐", 2: "正面 👍"}
    result = result_map.get(prediction, "未知")
    
    print(f"\n评论内容：{comment_text}")
    print(f"AI 诊断结果：{result}")
    # 打印三类各自的概率 (注意 probabilities 数组的顺序通常按标签升序 0, 1, 2 排列)
    print(f"AI 相信度 -> 负面: {probabilities[0]:.2f}, 中评: {probabilities[1]:.2f}, 正面: {probabilities[2]:.2f}")

# 3. 测试
print("=== 开始测试新评论 ===")
analyze_new_comment("这衣服穿上真好看，下次还来买！")     # 期待：正面
analyze_new_comment("发的什么破快递，等了一个星期，差评！") # 期待：负面
analyze_new_comment("还可以吧，凑合能用。")             # 期待：中评
analyze_new_comment("质量一般般，没有想象中好，但也没坏。") # 期待：中评