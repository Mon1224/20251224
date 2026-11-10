-- PostgreSQL 容器首次启动时自动执行
-- 业务库：传统保险业务数据
CREATE DATABASE insurance_biz;

-- 智能库：会话、消息、Agent 运行记录、checkpointer 状态
CREATE DATABASE insurance_agent;
