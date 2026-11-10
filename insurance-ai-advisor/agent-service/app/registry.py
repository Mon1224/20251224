"""集中导入所有 ORM 模型，确保它们注册到各自的 Base.metadata 上。

Alembic 的 autogenerate 和 create_all 都依赖这里的导入。
**新增模型文件后，必须在这里补一行导入**，否则迁移脚本会漏表。
"""

from app.infrastructure.database import AgentBase, BizBase
from app.modules.agent import models as _agent_models  # noqa: F401
from app.modules.claim import models as _claim_models  # noqa: F401
from app.modules.conversation import models as _conversation_models  # noqa: F401
from app.modules.order import models as _order_models  # noqa: F401
from app.modules.policy import models as _policy_models  # noqa: F401
from app.modules.product import models as _product_models  # noqa: F401
from app.modules.user import models as _user_models  # noqa: F401

BIZ_METADATA = BizBase.metadata
AGENT_METADATA = AgentBase.metadata

__all__ = ["AGENT_METADATA", "BIZ_METADATA"]
