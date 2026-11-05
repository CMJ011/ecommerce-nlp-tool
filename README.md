# 📊 电商评论情感分析与商业洞察系统 (E-commerce NLP Dashboard)

> 一个面向 B 端商家的商业级 AI 数据看板。基于 Python、Scikit-learn 与 FastAPI 构建，支持海量电商评论的批量情感多分类（正/中/负）与关键词智能提取。

## 📖 业务背景

在电商行业，商家每天面临成千上万条用户反馈。传统的人工阅读不仅效率低下，且难以快速提炼核心痛点。
本项目从真实商业场景出发，提供了一套**端到端（End-to-End）**的解决方案：
- **数据接入**：商家一键上传包含评论内容的 CSV 文件。
- **AI 批量分析**：后端瞬间完成情感三分类（正面、中评、负面）。
- **商业洞察**：直观展示情感占比饼图、关键指标进度条，并自动提取各分类下的 Top 30 高频关键词，帮助运营团队快速锁定“核心优势”与“改进痛点”。

## ✨ 核心功能

- **多分类 NLP 模型**：区别于简单的正负二分类，本系统支持 `0:负面 / 1:中评 / 2:正面` 的三分类模型，更贴近真实业务。
- **高级特征工程**：跳过传统的 Jieba 词级分词，采用 **Char-level (字符级) N-gram** 提取特征，极大降低了未登录词（OOV）和网络新词对模型准确率的干扰。
- **商业化数据大屏**：纯原生 HTML/CSS/JS + ECharts 构建，提供 KPI 核心数据卡、动态进度条、环形占比图与词云标签云。
- **高性能后端 API**：基于 FastAPI 构建，包含单条预测接口（`/predict`）与批量文件分析接口（`/analyze_batch`）。

## 🛠️ 技术栈

- **编程语言**：Python 3.10+
- **数据处理与 NLP**：Pandas, Jieba (TF-IDF 关键词提取)
- **机器学习**：Scikit-learn (TfidfVectorizer, MultinomialNB)
- **后端服务**：FastAPI, Uvicorn
- **前端可视化**：HTML/CSS/JS, ECharts 5.0

## 📸 效果展示

### 1. 商业级数据看板 (前端)
*(请在此处拖入你的前端大屏截图)*
![数据大屏展示](demo1.png)
<img width="1913" height="958" alt="demo1" src="https://github.com/user-attachments/assets/159d371f-5fe6-49bd-9b1e-7aea2921c844" />


### 2. API 接口文档 (FastAPI)
![API接口展示](demo2.png)
<img width="1903" height="965" alt="demo2" src="https://github.com/user-attachments/assets/60357bba-2dba-4253-925a-cdadb496e7ce" />


## 🚀 快速开始![Uploading demo1.png…]()


### 1. 环境准备
克隆项目并在本地创建虚拟环境：
```bash
git clone https://github.com/你的用户名/你的仓库名.git
cd 你的仓库名
python -m venv venv
# Windows 激活
.\venv\Scripts\activate
# Mac/Linux 激活
source venv/bin/activate
