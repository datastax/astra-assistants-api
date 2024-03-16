import asyncio
import logging

logger = logging.getLogger(__name__)
background_task_set = set()

event_loop = asyncio.new_event_loop()

async def add_background_task(function, run_id, thread_id, astradb):
    logger.debug("Creating background task")
    task = asyncio.create_task(
        function, name=run_id
    )
    background_task_set.add(task)
    task.add_done_callback(lambda t: on_task_completion(t, astradb=astradb, run_id=run_id, thread_id=thread_id))


def on_task_completion(task, astradb, run_id, thread_id):
    background_task_set.remove(task)
    logger.debug(f"Task stopped for run_id: {run_id} and thread_id: {thread_id}")

    if task.cancelled():
        logger.warning(f"Task cancelled, setting status to failed for run_id: {run_id} and thread_id: {thread_id}")
        astradb.update_run_status(id=run_id, thread_id=thread_id, status="failed");
        return
    try:
        exception = task.exception()
        if exception is not None:
            logger.warning(f"Task raised an exception, setting status to failed for run_id: {run_id} and thread_id: {thread_id}")
            logger.error(exception)
            astradb.update_run_status(id=run_id, thread_id=thread_id, status="failed");
            raise exception
        else:
            logger.debug(f"Task completed successfully for run_id: {run_id} and thread_id: {thread_id}")
    except asyncio.CancelledError:
        logger.warning(f"why wasn't this caught in task.cancelled()")
        logger.debug(f"Task cancelled, setting status to failed for run_id: {run_id} and thread_id: {thread_id}")
        astradb.update_run_status(id=run_id, thread_id=thread_id, status="failed");
