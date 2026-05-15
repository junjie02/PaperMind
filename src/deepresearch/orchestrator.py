import asyncio
import logging

from deepresearch.arxiv_search import ArxivSearchEngine
from deepresearch.config import Config
from deepresearch.database import Database
from deepresearch.dedup import DedupEngine
from deepresearch.diversifier import SearchDiversifier
from deepresearch.llm_client import LLMClient
from deepresearch.models import PaperRecord, WorkerTask
from deepresearch.search_client import SearchClient
from deepresearch.worker import Worker

logger = logging.getLogger(__name__)


class Orchestrator:
    def __init__(self, config: Config):
        self.config = config
        self.db = Database(config.db_path)
        self.dedup = DedupEngine()
        self.llm: LLMClient | None = None
        self.searcher: SearchClient | None = None
        self.arxiv_searcher: ArxivSearchEngine | None = None

        if config.search_backend == "arxiv":
            self.arxiv_searcher = ArxivSearchEngine(
                proxy=config.http_proxy,
                timeout=config.arxiv_timeout,
                retries=config.arxiv_retries,
            )
        elif config.search_backend == "claude":
            self.llm = LLMClient(config)
            self.searcher = SearchClient(max_timeout=config.search_timeout)
        else:
            raise ValueError(f"Unsupported search backend: {config.search_backend}")

    async def close(self):
        await self.db.close()
        if self.arxiv_searcher:
            await self.arxiv_searcher.close()

    async def run(
        self,
        research_question: str,
        target_count: int | None = None,
        workers_per_round: int | None = None,
        resume: bool = False,
    ) -> list[PaperRecord]:
        target = target_count or self.config.default_target
        workers = workers_per_round or self.config.workers_per_round

        await self.db.initialize()

        existing_ids = await self.db.get_all_ids() if resume else set()
        paper_set: dict[str, PaperRecord] = {}

        if resume and existing_ids:
            for p_dict in await self.db.get_papers():
                p = PaperRecord(**p_dict)
                paper_set[p.arxiv_id] = p

        if self.config.search_backend == "arxiv":
            return await self._run_arxiv(research_question, target, paper_set, existing_ids)

        if self.llm is None or self.searcher is None:
            raise RuntimeError(f"Unsupported search backend: {self.config.search_backend}")

        diversifier = SearchDiversifier(self.llm)
        round_num = 0
        stall_rounds = 0

        logger.info(f"开始调研: '{research_question}'  目标={target}  每轮Worker={workers}")

        while len(paper_set) < target and stall_rounds < self.config.stall_rounds_limit and round_num < self.config.max_rounds:
            round_num += 1
            all_exclude = set(paper_set.keys()) | existing_ids
            remaining = target - len(paper_set)
            workers_this_round = min(workers, remaining)

            # Step 1: LLM 生成搜索方向
            directions = await diversifier.generate(
                question=research_question,
                n=workers_this_round,
                exclude_ids=all_exclude,
            )

            if not directions:
                logger.warning("未生成搜索方向，跳过本轮")
                stall_rounds += 1
                continue

            quotas = self._allocate_worker_targets(remaining, len(directions))

            # Step 2: 并行启动 Claude Code worker
            tasks = []
            for i, d in enumerate(directions):
                task = WorkerTask(
                    research_question=research_question,
                    search_direction=d.direction,
                    exclude_ids=sorted(all_exclude),
                    round_num=round_num,
                    worker_index=i,
                    target_papers=quotas[i],
                )
                tasks.append(Worker(task, self.searcher).run())

            logger.info(f"--- 第 {round_num} 轮: 启动 {len(tasks)} 个 Worker ---")
            logger.info(f"第 {round_num} 轮剩余目标 {remaining} 篇，worker 配额: {quotas}")
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Step 3: 去重 & 入库
            new_count = 0
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.warning(f"Worker {i} 异常: {result}")
                    continue
                for paper in result:
                    norm_id = self.dedup.normalize(paper.arxiv_id)
                    if not norm_id:
                        logger.warning(f"无法标准化 arXiv ID: {paper.arxiv_id}")
                        continue
                    if norm_id not in paper_set and norm_id not in existing_ids:
                        paper.arxiv_id = norm_id
                        paper_set[norm_id] = paper
                        try:
                            await self.db.upsert(paper)
                        except Exception as e:
                            logger.error(f"入库失败 {norm_id}: {e}")
                            continue
                        new_count += 1
                        if len(paper_set) >= target:
                            break
                if len(paper_set) >= target:
                    break

            logger.info(f"第 {round_num} 轮完成: +{new_count} 篇, 累计 {len(paper_set)}/{target}")
            for i, result in enumerate(results):
                if isinstance(result, list):
                    ids = [p.arxiv_id for p in result]
                    logger.info(f"  Worker {i}: {len(result)} 篇 → {ids}")

            if new_count == 0:
                stall_rounds += 1
                logger.info(f"本轮无新论文。连续停滞: {stall_rounds}/{self.config.stall_rounds_limit}")
            else:
                stall_rounds = 0

        logger.info(f"调研结束: {len(paper_set)} 篇论文, {round_num} 轮")
        return list(paper_set.values())

    def _allocate_worker_targets(self, remaining: int, worker_count: int) -> list[int]:
        if worker_count <= 0:
            return []
        if remaining <= 0:
            return [0] * worker_count

        base = remaining // worker_count
        extra = remaining % worker_count
        return [base + (1 if i < extra else 0) for i in range(worker_count)]

    async def _run_arxiv(
        self,
        research_question: str,
        target: int,
        paper_set: dict[str, PaperRecord],
        existing_ids: set[str],
    ) -> list[PaperRecord]:
        if self.arxiv_searcher is None:
            raise RuntimeError("arXiv searcher is not initialized")

        logger.info("开始 arXiv 直连检索: %r 目标=%d", research_question, target)
        exclude_ids = set(paper_set.keys()) | existing_ids
        remaining = max(target - len(paper_set), 0)
        if remaining == 0:
            return list(paper_set.values())

        new_papers = await self.arxiv_searcher.search(
            question=research_question,
            target_count=remaining,
            exclude_ids=exclude_ids,
        )

        new_count = 0
        for paper in new_papers:
            norm_id = self.dedup.normalize(paper.arxiv_id)
            if not norm_id or norm_id in paper_set or norm_id in existing_ids:
                continue

            paper.arxiv_id = norm_id
            paper_set[norm_id] = paper
            try:
                await self.db.upsert(paper)
                new_count += 1
            except Exception as exc:
                logger.error("入库失败 %s: %s", norm_id, exc)

        logger.info("arXiv 检索完成: +%d 篇, 累计 %d/%d", new_count, len(paper_set), target)
        return list(paper_set.values())
