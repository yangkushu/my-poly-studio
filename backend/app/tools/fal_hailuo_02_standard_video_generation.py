"""fal.ai MiniMax Hailuo 02 Standard 视频生成 Tool。"""
import argparse
import json
import logging
from typing import Literal, Optional

from langchain_core.tools import tool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

try:
    from app.tools.fal_video_generation import FAL_USD_CNY_RATE, create_fal_client, generate_fal_video, resolve_input_url
except ModuleNotFoundError:  # 支持 cd backend 后直接执行 python app/tools/<tool>.py
    from fal_video_generation import FAL_USD_CNY_RATE, create_fal_client, generate_fal_video, resolve_input_url

HAILUO_ENDPOINTS = {
    "text": "fal-ai/minimax/hailuo-02/standard/text-to-video",
    "image": "fal-ai/minimax/hailuo-02/standard/image-to-video",
}


class GenerateFalHailuo02StandardVideoInput(BaseModel):
    """MiniMax Hailuo 02 Standard 视频生成工具的输入参数。"""

    prompt: str = Field(description="必填。描述画面、主体动作、镜头和风格；image 模式同样需要用它说明希望图片如何运动。")
    mode: Literal["text", "image"] = Field(
        default="text",
        description="生成模式：text=仅凭提示词生成，不能传 image_url 或 end_image_url；image=使用一张首帧生成，可选尾帧。",
    )
    image_url: Optional[str] = Field(
        default=None,
        description="首帧图片。仅 image 模式必填；可为公网 URL、data URI 或本项目的 /storage/... 图片路径。",
    )
    end_image_url: Optional[str] = Field(
        default=None,
        description="可选尾帧图片，仅 image 模式可用；视频将以该图作为最后一帧。可为公网 URL、data URI 或本项目的 /storage/... 图片路径。",
    )
    duration: Literal["6", "10"] = Field(default="6", description="视频时长（秒），仅可选 6 或 10，默认 6。")
    resolution: Literal["512P", "768P"] = Field(
        default="768P",
        description="image 模式的输出分辨率：512P 或 768P，默认 768P。text 端点不提供该参数，本工具会忽略它。",
    )
    prompt_optimizer: bool = Field(default=True, description="是否启用模型的提示词优化，默认启用。")


@tool("generate_fal_hailuo_02_standard_video", args_schema=GenerateFalHailuo02StandardVideoInput)
def generate_fal_hailuo_02_standard_video_tool(
    prompt: str,
    mode: Literal["text", "image"] = "text",
    image_url: Optional[str] = None,
    end_image_url: Optional[str] = None,
    duration: Literal["6", "10"] = "6",
    resolution: Literal["512P", "768P"] = "768P",
    prompt_optimizer: bool = True,
) -> str:
    """
    使用 fal.ai 的 MiniMax Hailuo 02 Standard 生成视频。

    - text：纯文生视频；只提供 prompt，不传图片。
    - image：首帧图生视频；必须提供 image_url，可选 end_image_url 指定尾帧。

    视频时长只能为 6 或 10 秒。工具完成后返回包含本地 video_url 的 JSON 结果。
    """
    if mode == "image" and not image_url:
        logger.warning("Hailuo 视频请求被拒绝: image 模式缺少首帧")
        return json.dumps({"error": "image 模式必须提供 image_url"}, ensure_ascii=False)
    if mode == "text" and (image_url or end_image_url):
        logger.warning("Hailuo 视频请求被拒绝: text 模式包含图片参数")
        return json.dumps({"error": "text 模式不接受 image_url 或 end_image_url"}, ensure_ascii=False)

    try:
        logger.info(
            "开始 Hailuo 视频生成: mode=%s duration=%ss resolution=%s prompt_optimizer=%s has_end_frame=%s",
            mode, duration, resolution if mode == "image" else "n/a", prompt_optimizer, bool(end_image_url),
        )
        client = create_fal_client()
        payload = {"prompt": prompt, "duration": duration, "prompt_optimizer": prompt_optimizer}
        if mode == "image":
            payload["resolution"] = resolution
            payload["image_url"] = resolve_input_url(image_url, client)
            if end_image_url:
                payload["end_image_url"] = resolve_input_url(end_image_url, client)
        result = generate_fal_video(HAILUO_ENDPOINTS[mode], payload, prompt, "hailuo02")
        logger.info("Hailuo 视频生成完成: task_id=%s local_path=%s", result.get("task_id"), result.get("local_path"))
        price_usd = 0.017 if resolution == "512P" else 0.045
        return json.dumps({
            **result,
            "provider": "fal.ai", "model": "MiniMax Hailuo 02 Standard", "mode": mode,
            "endpoint": HAILUO_ENDPOINTS[mode], "prompt": prompt, "resolution": resolution,
            "price_usd_per_second": price_usd, "price_yuan_per_second_estimate": round(price_usd * FAL_USD_CNY_RATE, 4),
            "usd_cny_rate_used": FAL_USD_CNY_RATE, "message": "视频已生成并保存到本地",
        }, ensure_ascii=False)
    except Exception as error:
        logger.exception("Hailuo 视频生成失败: mode=%s", mode)
        return json.dumps({"error": f"生成视频时出错: {error}"}, ensure_ascii=False)


def main() -> None:
    """命令行测试入口；使用有效 FAL_KEY 会产生 fal 视频生成费用。"""
    parser = argparse.ArgumentParser(description="测试 fal MiniMax Hailuo 02 Standard 视频 Tool")
    parser.add_argument("--prompt", default="一只小狗在阳光下的草地上奔跑", help="视频提示词")
    parser.add_argument("--mode", choices=("text", "image"), default="text")
    parser.add_argument("--image-url", help="image 模式的首帧")
    parser.add_argument("--end-image-url", help="image 模式可选尾帧")
    parser.add_argument("--duration", choices=("6", "10"), default="6")
    parser.add_argument("--resolution", choices=("512P", "768P"), default="768P")
    parser.add_argument("--no-prompt-optimizer", action="store_true")
    args = parser.parse_args()
    if args.mode == "image" and not args.image_url:
        parser.error("--mode image 时必须提供 --image-url")
    logging.basicConfig(level=logging.INFO)
    print(generate_fal_hailuo_02_standard_video_tool.invoke({
        "prompt": args.prompt, "mode": args.mode, "image_url": args.image_url, "end_image_url": args.end_image_url,
        "duration": args.duration, "resolution": args.resolution, "prompt_optimizer": not args.no_prompt_optimizer,
    }))


if __name__ == "__main__":
    main()
