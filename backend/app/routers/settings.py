"""
PolyStudio 设置 API 路由
支持 Skills 配置、MCP 服务器配置、环境变量配置
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from pathlib import Path
from typing import Any
import json
import re
import logging
# from app.services import skill_service
# from app.services import workspace_service

logger = logging.getLogger(__name__)

router = APIRouter()

# 存储路径
BASE_DIR = Path(__file__).parent.parent.parent
STORAGE_DIR = BASE_DIR / "storage"
SETTINGS_FILE = STORAGE_DIR / "settings.json"
ENV_FILE = BASE_DIR / ".env"