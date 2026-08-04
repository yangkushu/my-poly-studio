"""
聊天路由 - 处理对话请求
"""
from fastapi import APIRouter, HTTPException, Request, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import json
import os
import uuid
import asyncio
import logging
from datetime import datetime
from pathlib import Path
from app.services.agent_service import process_chat_stream
from app.services.history_service import history_service
from app.services.connection_manager import manager

logger = logging.getLogger(__name__)

router = APIRouter()


class ChatRequest(BaseModel):
    message: str
    messages: Optional[List[Dict[str, Any]]] = []
    session_id: Optional[str] = None
    canvas_id: Optional[str] = None



@router.get("/canvases")
async def get_canvases():
    """获取所有画布历史"""
    return history_service.get_canvases()


@router.post("/canvases")
async def save_canvas(request: Request):
    """保存或更新画布（项目）

    注意：前端会携带 Excalidraw 的 data(elements/appState/files)，
    用 Pydantic 模型解析容易因 extra 字段处理/热重载不同步而丢字段。
    这里直接存原始 JSON，避免 data 被过滤导致刷新后画布空白。
    """
    payload = await request.json()
    return history_service.save_canvas(payload)


@router.post("/chat")
async def chat(request: ChatRequest):
    """
    处理聊天请求，返回流式响应
    支持OpenAI格式的流式输出
    若请求包含 canvas_id，同时将每个 SSE 事件广播给订阅该 canvas 的 WebSocket 客户端
    """
    try:
        # 构建消息历史
        messages =  request.messages.copy() if request.messages else []
        messages.append({
            "role":"user",
            "content": request.message
        })

        canvas_id  = request.canvas_id

        async def stream_and_save():
            assistant_content = ""
            # 收集工具调用结果（图片/视频等媒体 URL）
            tool_results: list[dict] = []
            # 广播用户消息给前端（await 直接调用，不用 create_task）
            user_event = json.dumps({"type": "user_message", "content":request.message}, ensure_ascii=False)
            if canvas_id :
                await manager.broadcast(canvas_id, user_event)
            else:
                await manager.broadcast_all(user_event)

            async for chunk in process_chat_stream(messages, request.session_id):
                yield chunk
                if chunk.startswith("data:"):
                    data_str = chunk[len("data: "):].strip()
                    if data_str and data_str != "[DONE]":
                        # 广播给前端
                        if canvas_id:
                            await manager.broadcast(canvas_id, data_str)
                        else:
                            await manager.broadcast_all(data_str)
                        # 收集 assistant 文本内容与工具调用结果
                        try:
                            ev = json.loads(data_str)
                            ev_type = ev.get("type")
                            if ev_type == "delta" and ev.get("content"):
                                assistant_content += ev["content"]
                            elif ev_type == "tool_result":
                                # 保留工具调用结果（含图片/视频 URL 等）
                                tool_results.append({
                                    "tool_call_id": ev.get("tool_call_id"),
                                    "content": ev.get("content"),
                                })
                        except Exception:
                            pass
                elif data_str == "[DONE]":
                        done_event = json.dumps({"type": "done"}, ensure_ascii=False)
                        if canvas_id:
                            await manager.broadcast(canvas_id, done_event)
                        else:
                            await manager.broadcast_all(done_event)
            # 流结束后保存历史
            try:
                import time
                ts = int(time.time() * 1000)

                # 构建本轮新增的消息（用户 + 助手）
                new_messages = [
                    {"role": "user", "content": request.message},
                    {
                            "role": "assistant",
                            "content": assistant_content,
                            # 若有工具调用结果（图片等），附加到 assistant 消息
                            **({"tool_results": tool_results} if tool_results else {}),
                        },
                ]

                if canvas_id:
                    # 已有项目：加载原有 canvas，追加本轮消息，不新建项目
                    existing_canvases = history_service.get_canvases()
                    existing = next((c for c in existing_canvases if c.get("id") == canvas_id ),None)
                    if existing:
                        existing_messages = existing.get("messages",[])
                        existing["messages"] = existing_messages + new_messages
                        await asyncio.to_thread(history_service.save_canvas, existing)
                    else:
                        # canvas_id 在历史中找不到（可能已删除），退化为新建
                        canvas_to_save = {
                                "id": canvas_id,
                                "name": request.message[:30],
                                "createdAt": ts,
                                "messages": new_messages,
                                "data": {"elements": [], "appState": {}, "files": {}},
                            }
                        await asyncio.to_thread(history_service.save_canvas, canvas_to_save)
                else:
                        # 没有 canvas_id：新建项目
                        new_canvas = {
                            "id": f"canvas-{ts}",
                            "name": request.message[:30],
                            "createdAt": ts,
                            "messages": new_messages,
                            "data": {"elements": [], "appState": {}, "files": {}},
                        }
                        await asyncio.to_thread(history_service.save_canvas, new_canvas)
            except Exception as e:
                    logger.warning(f"保存对话历史失败: {e}")

        # 返回流式响应 - 确保立即发送，不缓冲
        return StreamingResponse(
            stream_and_save(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
                "Content-Type": "text/event-stream; charset=utf-8"
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
