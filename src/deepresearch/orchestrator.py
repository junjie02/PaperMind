import asyncio
import logging
import math
from pathlib import Path

from deepresearch.artifact_writer import ArtifactWriter
from deepresearch.config import Config
from deepresearch.database import Database
from deepresearch.models import PaperRecord, WorkerTask
from deepresearch.search_client import SearchClient
from deepresearch.worker import Worker

logger = logging.getLogger(__name__)


class Orchestrator:
    def __init__(self, config: Config):
        self.config = config
        self.db = Database(config.db_path)
        self.searcher = SearchClient(
            max_timeout=config.search_timeout,
            model=config.worker_model,
        )
        self.artifact_writer = ArtifactWriter(
            artifact_dir=Path(config.pdf_dir),
            config=config,
        )

    async def close(self):
        await self.db.close()
        await self.artifact_writer.close()

    async def run(
        self,
        research_question: str,
        target_count: int | None = None,
        workers_per_round: int | None = None,
        resume: bool = False,
        skip_pdf: bool = False,
    ) -> list[PaperRecord]:
        target = target_count or self.config.default_target
        workers = workers_per_round or self.config.workers_per_round
        per_round = self.config.papers_per_round

        await self.db.initialize()

        existing_ids = await self.db.get_all_ids() if resume else set()
        paper_set: dict[str, PaperRecord] = {}

        if resume and existing_ids:
            for p_dict in await self.db.get_papers():
                p = PaperRecord(**p_dict)
                paper_set[p.paper_id] = p

        planned_rounds = math.ceil(target / per_round)
        logger.info(
            "开始调研: '%s'  目标=%d  每轮=%d  计划轮数=%d  Worker/轮=%d",
            research_question, target, per_round, planned_rounds, workers,
        )

        round_num = 0
        while True:
            round_num += 1
            is_planned = round_num <= planned_rounds
            all_exclude = set(paper_set.keys()) | existing_ids

            if len(paper_set) >= target and is_planned:
                logger.info("目标已满足（%d/%d），跳过剩余计划轮次", len(paper_set), target)
                break

            logger.info(
                "=== 第 %d 轮 (%s) | 已有 %d 篇 | 排除 %d 个 ID ===",
                round_num,
                "计划内" if is_planned else "补充",
                len(paper_set),
                len(all_exclude),
            )

            quotas = self._allocate_worker_targets(per_round, workers)
            tasks = []
            exclude_list = sorted(all_exclude)
            exclude_titles = [
                paper_set[pid].title if pid in paper_set else ""
                for pid in exclude_list
            ]
            for i in range(workers):
                task = WorkerTask(
                    research_question=research_question,
                    exclude_ids=exclude_list,
                    exclude_titles=exclude_titles,
                    worker_index=i,
                    target_papers=quotas[i],
                )
                tasks.append(Worker(task, self.searcher).run())

            logger.info("启动 %d 个 Worker，配额: %s", len(tasks), quotas)
            results = await asyncio.gather(*tasks, return_exceptions=True)

            new_count = 0
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.warning("Worker %d 异常: %s", i, result)
                    continue
                for paper in result:
                    pid = paper.paper_id
                    if not pid:
                        logger.warning("论文缺少 paper_id，跳过: %r", paper.title)
                        continue
                    if pid in paper_set or pid in existing_ids:
                        continue
                    paper_set[pid] = paper
                    try:
                        await self.db.upsert(paper)
                    except Exception as e:
                        logger.error("入库失败 %s: %s", pid, e)
                        continue
                    new_count += 1

            logger.info("第 %d 轮完成: +%d 篇, 累计 %d/%d", round_num, new_count, len(paper_set), target)

            if not is_planned and new_count == 0:
                logger.info("补充轮无新增，停止搜索")
                break

            if not is_planned and len(paper_set) >= target:
                logger.info("目标已满足，停止搜索")
                break

        logger.info("调研结束: %d 篇论文（共 %d 轮）", len(paper_set), round_num)
        self._log_direction_summary(paper_set.values())
        papers = list(paper_set.values())
        await self._maybe_write_artifacts(papers, skip_pdf)
        return papers

    def _allocate_worker_targets(self, remaining: int, worker_count: int) -> list[int]:
        if worker_count <= 0:
            return []
        if remaining <= 0:
            return [0] * worker_count

        base = remaining // worker_count
        extra = remaining % worker_count
        return [base + (1 if i < extra else 0) for i in range(worker_count)]

    async def _maybe_write_artifacts(
        self, papers: list[PaperRecord], skip_pdf: bool
    ) -> None:
        if skip_pdf:
            logger.info("跳过 PDF / Markdown 写入 (--skip-pdf)")
            return
        if not papers:
            return
        pdf_ok, md_ok, fail = await self.artifact_writer.write_all(papers, self.db)
        logger.info("产物写入: arxiv PDF %d / 非 arxiv MD %d / 失败 %d", pdf_ok, md_ok, fail)

    def _log_direction_summary(self, papers) -> None:
        groups: dict[str, list[PaperRecord]] = {}
        for p in papers:
            groups.setdefault(p.search_direction or "(未分类)", []).append(p)
        logger.info("按方向分组: %d 个方向", len(groups))
        for direction, items in groups.items():
            logger.info("  · [%s] %d 篇", direction, len(items))
