import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from wordcloud import WordCloud
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("1. 正在读取和清洗数据...")
df = pd.read_csv("comments.csv", encoding="utf-8-sig")
df.dropna(subset=["评论内容"], inplace=True)
df.drop_duplicates(subset=["评论内容"], inplace=True)
print(f"清洗后剩余数据量: {len(df)} 条")

print("2. 正在提取字符级特征 (跳过中文分词，直接认字)...")
vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(1, 2))
X = vectorizer.fit_transform(df["评论内容"])
y = df["情感标签"]

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("3. 正在训练三分类朴素贝叶斯模型...")
model = MultinomialNB()
model.fit(X_train, y_train)

# 评估（注意 target_names 的顺序必须对应 0, 1, 2）
y_pred = model.predict(X_test)
print("\n===== 模型评估报告 =====")
print(classification_report(y_test, y_pred, target_names=["负面", "中评", "正面"]))

print("4. 正在生成正面评论词云图...")
pos_text = " ".join(df[df["情感标签"] == 2]["评论内容"])
wc = WordCloud(font_path="msyh.ttc", width=800, height=400, background_color="white").generate(pos_text)

plt.figure(figsize=(10, 5))
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")
plt.title("正面评论高频词云")
plt.savefig("positive_wordcloud.png")
print("词云图已保存为 positive_wordcloud.png")