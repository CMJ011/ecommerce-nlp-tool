from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import jieba.analyse
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

app = FastAPI(title="电商评论批量分析 API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. 启动加载模型
print("正在加载模型...")
df_train = pd.read_csv("comments.csv", encoding="utf-8-sig")
vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(1, 2))
X_train = vectorizer.fit_transform(df_train["评论内容"])
y_train = df_train["情感标签"]
model = MultinomialNB()
model.fit(X_train, y_train)

# 提取关键词的辅助函数
# 修改 app.py 中的 get_keywords 函数
def get_keywords(text_list, top_n=30): # 修改为30
    all_text = "".join(text_list)
    # 提取 60 个词，然后过滤掉一半
    tags = jieba.analyse.extract_tags(all_text, topK=top_n*2)
    stopwords = ["的", "了", "是", "我", "就", "也", "在", "和", "有", "不", "很", "都", "这", "个", "等", "太", "啊", "吧"]
    filtered_tags = [t for t in tags if len(t) > 1 and t not in stopwords]
    return filtered_tags[:top_n]

@app.post("/analyze_batch")
async def analyze_batch(file: UploadFile = File(...)):
    # 1. 读取上传的 CSV 文件
    contents = await file.read()
    import io
    df = pd.read_csv(io.StringIO(contents.decode("utf-8-sig")))
    
    # 假设商家的数据列名包含“评论内容”
    if "评论内容" not in df.columns:
        return {"error": "CSV 中必须包含 '评论内容' 列"}
    
    # 2. 批量预测
    texts = df["评论内容"].astype(str).tolist()
    vecs = vectorizer.transform(texts)
    predictions = model.predict(vecs)
    
    # 3. 统计数据
    total = len(predictions)
    # 0:负面, 1:中评, 2:正面
    counts = Counter(predictions)
    pos_count = counts.get(2, 0)
    neg_count = counts.get(0, 0)
    neu_count = counts.get(1, 0)
    
    # 4. 提取各分类的高频关键词
    pos_texts = [texts[i] for i in range(total) if predictions[i] == 2]
    neg_texts = [texts[i] for i in range(total) if predictions[i] == 0]
    neu_texts = [texts[i] for i in range(total) if predictions[i] == 1]
    
    return {
        "total": total,
        "distribution": [
            {"name": "正面", "value": pos_count},
            {"name": "中评", "value": neu_count},
            {"name": "负面", "value": neg_count}
        ],
        "keywords": {
            "正面": get_keywords(pos_texts) if pos_texts else [],
            "负面": get_keywords(neg_texts) if neg_texts else [],
            "中评": get_keywords(neu_texts) if neu_texts else []
        }
    }