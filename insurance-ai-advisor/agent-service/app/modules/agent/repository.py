"""Agent 运行记录的写库。

和 checkpointer 的区别要分清：
  checkpointer 存的是"让 LangGraph 能恢复执行"的内部状态（机器要的）
  这里存的是"给人看的运行日志"（排查、算账、优化要的）
"""

from sqlalchemy.ext.asyncio import AsyncSession

from app.modules.agent.models import AgentRun, ToolCall


class AgentRunRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create_run(self, **kwargs) -> AgentRun:
        run = AgentRun(**kwargs)
        self.session.add(run)
        await self.session.flush()
        return run

    async def add_tool_call(self, run_id: int, **kwargs) -> ToolCall:
        call = ToolCall(run_id=run_id, **kwargs)
        self.session.add(call)
        await self.session.flush()
        return call
