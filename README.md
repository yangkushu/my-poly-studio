# PolyStudio

PolyStudio 是一个前后端分离的创作工作台服务：前端使用 React + Vite，后端使用 FastAPI。

## 目录结构

- `frontend/`：React 前端
- `backend/`：FastAPI 后端

## 本地运行

### 后端

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp env.example .env
uvicorn app.main:app --reload --port 8000
```

### 前端

```bash
cd frontend
npm install
npm run dev
```

前端开发服务器默认运行在 `http://localhost:3000`，API 请求代理到 `http://localhost:8000`。

## 配置

复制 `backend/env.example` 为 `backend/.env`，再填写所选 LLM provider 的 API 配置。不要将真实密钥提交到仓库。

## 检查

```bash
cd frontend
npm run build
```

后端可通过 `GET /health` 检查服务状态。
