from core.custom_logging import get_logger
from db.session import check_conn_psql

logger = get_logger()


async def check_startup_dependencies() -> bool:
    database_ok = await check_conn_psql()
    if not database_ok:
        logger.error("Startup dependency check failed")
    return database_ok


__all__ = ("check_startup_dependencies",)
