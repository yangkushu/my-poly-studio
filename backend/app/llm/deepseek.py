"""
DeepSeek LLM 实现
"""
import os
import logging
from langchain_openai import ChatOpenAI
from app.llm.base import BaseLLMProvider

logger = logging.getLogger(__name__)

class DeepSeekLLMProvider(BaseLLMProvider):
    """DeepSeek LLM 提供商"""

    def __init__(self):
            # 从环境变量获取配置
            self.api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
            self.base_url = "https://api.deepseek.com"
            self.model_name = "deepseek-v4-flash"

            if not self.api_key:
                raise RuntimeError(
                    "未配置 DEEPSEEK_API_KEY。请在 backend/.env 中设置，"
                    "可参考 env.example（cp env.example .env）。"
                )

    def create_model(self) -> ChatOpenAI:
        """创建 SiliconFlow ChatModel 实例"""
        logger.info(f"🔷 创建 SiliconFlow LLM: model={self.model_name}, base_url={self.base_url}")

        model = ChatOpenAI(
            model=self.model_name,
            api_key=self.api_key,
            base_url=self.base_url,
            temperature=0.7,
            streaming=True,  # 启用流式输出
            max_tokens=2048,
            # 关键：禁止并行工具调用，强制"一次调用一个工具 -> 等结果 -> 再下一次"
            model_kwargs={"parallel_tool_calls": False},
        )

        return model

    def get_provider_name(self) -> str:
            return "deepseek"