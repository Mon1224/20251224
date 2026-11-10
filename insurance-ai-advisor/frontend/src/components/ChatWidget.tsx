/** 悬浮的 AI 顾问对话窗。所有页面都能唤起——这是"AI 顾问"的体验关键。 */

import { CloseOutlined, SendOutlined, StopOutlined } from "@ant-design/icons";
import { Avatar, Button, Drawer, Empty, FloatButton, Input, Space, Tag, Tooltip } from "antd";
import { useCallback, useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";

import { streamChat } from "@/api/agent";
import type { Citation } from "@/api/types";
import { useAuth } from "@/store/auth";

interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  citations?: Citation[];
}

const QUICK_QUESTIONS = [
  "重疾险的等待期是多久？",
  "百万医疗险的免赔额怎么算？",
  "我的保单有哪些？",
  "30 岁买 50 万保额大概多少钱？",
];

const newId = () => crypto.randomUUID();

export default function ChatWidget() {
  const { user } = useAuth();
  const navigate = useNavigate();

  const [open, setOpen] = useState(false);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [status, setStatus] = useState("");
  const [sending, setSending] = useState(false);
  const [conversationId, setConversationId] = useState<number>();

  const abortRef = useRef<AbortController | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);

  // 有新消息或正在流式输出时，自动滚到底部
  useEffect(() => {
    const node = scrollRef.current;
    if (node) node.scrollTop = node.scrollHeight;
  }, [messages, status]);

  const send = useCallback(
    async (question: string) => {
      const text = question.trim();
      if (!text || sending) return;

      setInput("");
      setSending(true);
      setStatus("正在连接…");

      const replyId = newId();
      setMessages((prev) => [
        ...prev,
        { id: newId(), role: "user", content: text },
        { id: replyId, role: "assistant", content: "" },
      ]);

      const controller = new AbortController();
      abortRef.current = controller;

      const patchReply = (patch: Partial<ChatMessage>) => {
        setMessages((prev) =>
          prev.map((item) => (item.id === replyId ? { ...item, ...patch } : item)),
        );
      };

      await streamChat(
        { question: text, conversationId },
        (event) => {
          switch (event.type) {
            case "meta":
              setConversationId(event.conversationId);
              break;
            case "status":
              setStatus(event.text);
              break;
            case "delta":
              setStatus("");
              setMessages((prev) =>
                prev.map((item) =>
                  item.id === replyId
                    ? { ...item, content: item.content + event.text }
                    : item,
                ),
              );
              break;
            case "done":
              setStatus("");
              patchReply({ citations: event.payload.citations });
              break;
            case "error":
              setStatus("");
              setMessages((prev) =>
                prev.map((item) =>
                  item.id === replyId
                    ? { ...item, content: item.content || `⚠️ ${event.message}` }
                    : item,
                ),
              );
              break;
          }
        },
        controller.signal,
      );

      abortRef.current = null;
      setSending(false);
    },
    [conversationId, sending],
  );

  const stop = () => {
    abortRef.current?.abort();
    abortRef.current = null;
    setSending(false);
    setStatus("");
  };

  const reset = () => {
    stop();
    setMessages([]);
    setConversationId(undefined);
  };

  return (
    <>
      <FloatButton
        type="primary"
        icon={<span style={{ fontSize: 18 }}>AI</span>}
        tooltip="AI 保险顾问"
        badge={{ dot: true }}
        onClick={() => setOpen(true)}
      />

      <Drawer
        title={
          <Space>
            <Avatar style={{ background: "#0f766e" }}>安</Avatar>
            <span>小安 · AI 保险顾问</span>
          </Space>
        }
        placement="right"
        width={430}
        open={open}
        onClose={() => setOpen(false)}
        extra={
          <Tooltip title="清空对话">
            <Button type="text" icon={<CloseOutlined />} onClick={reset} />
          </Tooltip>
        }
        styles={{ body: { padding: 0, display: "flex", flexDirection: "column" } }}
      >
        {!user ? (
          <div style={{ padding: 40, textAlign: "center" }}>
            <Empty description="登录后即可咨询保险问题" />
            <Button
              type="primary"
              style={{ marginTop: 16 }}
              onClick={() => {
                setOpen(false);
                navigate("/login");
              }}
            >
              去登录
            </Button>
          </div>
        ) : (
          <>
            <div ref={scrollRef} style={{ flex: 1, overflowY: "auto", padding: 16 }}>
              {messages.length === 0 ? (
                <div>
                  <div style={{ color: "#6b7280", fontSize: 13, marginBottom: 12 }}>
                    你好，我是小安。可以问我保险条款、产品对比，也可以查你自己的保单和理赔进度。
                  </div>
                  <Space direction="vertical" style={{ width: "100%" }}>
                    {QUICK_QUESTIONS.map((question) => (
                      <Button
                        key={question}
                        block
                        style={{ textAlign: "left" }}
                        onClick={() => void send(question)}
                      >
                        {question}
                      </Button>
                    ))}
                  </Space>
                </div>
              ) : (
                <Space direction="vertical" size={12} style={{ width: "100%" }}>
                  {messages.map((message) => (
                    <div
                      key={message.id}
                      style={{
                        display: "flex",
                        justifyContent: message.role === "user" ? "flex-end" : "flex-start",
                      }}
                    >
                      <div
                        className={
                          message.role === "user"
                            ? "chat-bubble chat-bubble-user"
                            : "chat-bubble chat-bubble-assistant"
                        }
                        style={{ maxWidth: "88%" }}
                      >
                        {message.content}
                        {message.role === "assistant" &&
                          sending &&
                          message.id === messages[messages.length - 1]?.id && (
                            <span className="chat-cursor" />
                          )}
                        {message.citations && message.citations.length > 0 && (
                          <div className="chat-citation">
                            <div style={{ marginBottom: 4 }}>依据条款：</div>
                            {message.citations.map((citation) => (
                              <div key={citation.clause_id}>
                                · {citation.product_name} 第 {citation.clause_no} 条
                                {citation.title ? ` ${citation.title}` : ""}
                              </div>
                            ))}
                          </div>
                        )}
                      </div>
                    </div>
                  ))}
                </Space>
              )}

              {status && (
                <div className="chat-status" style={{ marginTop: 12 }}>
                  <Tag color="processing">{status}</Tag>
                </div>
              )}
            </div>

            <div style={{ borderTop: "1px solid #e5e7eb", padding: 12 }}>
              <Space.Compact style={{ width: "100%" }}>
                <Input.TextArea
                  value={input}
                  onChange={(event) => setInput(event.target.value)}
                  placeholder="描述你的问题，例如：等待期是多久？"
                  autoSize={{ minRows: 1, maxRows: 4 }}
                  disabled={sending}
                  onPressEnter={(event) => {
                    if (!event.shiftKey) {
                      event.preventDefault();
                      void send(input);
                    }
                  }}
                />
                {sending ? (
                  <Button danger icon={<StopOutlined />} onClick={stop}>
                    停止
                  </Button>
                ) : (
                  <Button type="primary" icon={<SendOutlined />} onClick={() => void send(input)}>
                    发送
                  </Button>
                )}
              </Space.Compact>
              <div style={{ color: "#9ca3af", fontSize: 11, marginTop: 6 }}>
                回答依据保险条款生成，仅供参考，实际以合同为准
              </div>
            </div>
          </>
        )}
      </Drawer>
    </>
  );
}
