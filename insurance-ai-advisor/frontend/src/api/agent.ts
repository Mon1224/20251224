/** 流式对话。用 fetch + ReadableStream 而不是 EventSource：
 *  EventSource 只能发 GET，既不能带请求体也不能自定义 Authorization 头。 */

import { getToken } from "@/api/client";
import type { ChatDonePayload } from "@/api/types";

const API_BASE = import.meta.env.VITE_API_BASE ?? "";

export type StreamEvent =
  | { type: "meta"; conversationId: number }
  | { type: "status"; stage: string; text: string }
  | { type: "delta"; text: string }
  | { type: "done"; payload: ChatDonePayload }
  | { type: "error"; code: string; message: string };

export async function streamChat(
  params: { question: string; conversationId?: number },
  onEvent: (event: StreamEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  const response = await fetch(`${API_BASE}/api/v1/agent/chat/stream`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${getToken() ?? ""}`,
    },
    body: JSON.stringify({
      question: params.question,
      conversation_id: params.conversationId,
    }),
    signal,
  });

  if (!response.ok || !response.body) {
    onEvent({
      type: "error",
      code: `HTTP_${response.status}`,
      message: "连接 AI 服务失败，请稍后重试",
    });
    return;
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let eventName = "";
  let dataLines: string[] = [];

  // SSE 的规则：一个空行代表"这条事件结束"，此时才把累积的 data 拼起来解析。
  // 网络分片不会按事件边界切分，所以必须先缓冲再拆行。
  const flush = () => {
    if (!eventName) return;
    const raw = dataLines.join("\n");
    const name = eventName;
    eventName = "";
    dataLines = [];
    if (!raw) return;

    try {
      const data = JSON.parse(raw) as Record<string, unknown>;
      switch (name) {
        case "meta":
          onEvent({ type: "meta", conversationId: Number(data.conversation_id) });
          break;
        case "status":
          onEvent({ type: "status", stage: String(data.stage), text: String(data.text) });
          break;
        case "delta":
          onEvent({ type: "delta", text: String(data.text) });
          break;
        case "done":
          onEvent({ type: "done", payload: data as unknown as ChatDonePayload });
          break;
        case "error":
          onEvent({
            type: "error",
            code: String(data.code ?? "AGENT_ERROR"),
            message: String(data.message ?? "AI 服务异常"),
          });
          break;
        default:
          break;
      }
    } catch {
      // 单个事件解析失败就跳过，不要让整条流中断
    }
  };

  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      let index = buffer.indexOf("\n");
      while (index >= 0) {
        const line = buffer.slice(0, index).replace(/\r$/, "");
        buffer = buffer.slice(index + 1);
        if (line === "") {
          flush();
        } else if (line.startsWith("event:")) {
          eventName = line.slice(6).trim();
        } else if (line.startsWith("data:")) {
          dataLines.push(line.slice(5).trim());
        }
        index = buffer.indexOf("\n");
      }
    }
    flush();
  } catch (error) {
    // 用户主动取消不算错误
    if (!(error instanceof DOMException && error.name === "AbortError")) {
      onEvent({ type: "error", code: "NETWORK_ERROR", message: "网络中断，回答未生成完" });
    }
  }
}
