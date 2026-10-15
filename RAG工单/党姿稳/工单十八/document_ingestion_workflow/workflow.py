# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-18-实现文档质量评估Skill并集成至智能体工作流
# 模块职责：document_ingestion_workflow 专用质检工作流。核心链路为
#           “触发质检 → 调用 DocumentQualityAssessmentSkill → 按分类标签路由到不同解析器”，
#           并对需要人工复核的文档设置人工分支。

import os                                                          # 路径
import json                                                        # 清单落盘
import time                                                        # 计时

import document_quality_assessment as dqa                          # 质检 Skill
from document_ingestion_workflow import parsers                    # 解析器注册表

WORKFLOW_NAME = "document_ingestion_workflow"                      # 工作流名称


class DocumentIngestionWorkflow(object):
    """文档入库质检工作流：质检 → 决策 → 路由执行。"""

    def __init__(self, config_file=None, work_dir=None):
        self.cfg, self.note = dqa.load_config(config_file)         # 配置
        self.work_dir = work_dir or os.path.join(                  # 工作目录
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
        os.makedirs(self.work_dir, exist_ok=True)                  # 确保存在

    # ---------------- 节点 1：触发质检 ----------------
    def inspect(self, target, output_dir=None):
        """调用质检 Skill，返回结构化报告。"""
        out = output_dir or os.path.join(self.work_dir, "workflow_reports")
        return dqa.run_assessment(target=target, config_file=self.cfg.source,
                                  output_dir=out)

    # ---------------- 节点 2：决策（按分类标签路由） ----------------
    def decide(self, report):
        """决策节点：把质检报告转成“文档 → 解析器”的路由计划与人工复核清单。"""
        plan, review, stats = [], [], {}                           # 计划 / 复核 / 统计
        for d in report["classification"]["documents"]:            # 逐文档
            route = d.get("route") or "passthrough_parser"         # 路由目标
            stats[route] = stats.get(route, 0) + 1                 # 路由分布
            item = {"path": d["path"], "name": d.get("name"),      # 计划项
                    "primary_label": d.get("primary_label"),       # 主标签
                    "quality_tags": d.get("quality_tags", []),     # 质量标签
                    "route": route,                                # 目标解析器
                    "reason": self._reason(d)}                     # 判定依据
            plan.append(item)
            if route == "human_review":                            # 人工分支
                review.append(item)
        return {"workflow": WORKFLOW_NAME, "node": "decide",
                "total": len(plan), "route_distribution": stats,
                "plan": plan, "human_review": review}

    def _reason(self, doc):
        """给出该文档被路由到当前解析器的原因说明。"""
        label = doc.get("primary_label")                           # 主标签
        if doc.get("route") == "human_review":                     # 人工分支
            return "命中人工复核标签 %s" % "、".join(doc.get("quality_tags", []))
        if label == "Scan_PDF":                                    # 扫描型
            return "扫描型 PDF（无文本层），路由至 OCR 解析流程"
        if label == "Mixed_PDF":                                   # 混合型
            return "混合型 PDF，逐页分流：有文本层直接抽取，无文本层做 OCR"
        if label == "Text_PDF":                                    # 文字型
            return "文字型 PDF，可直接抽取文本层"
        return "按主标签 %s 选择对应解析器" % label

    # ---------------- 节点 3：执行路由 ----------------
    def execute(self, report, plan=None, limit=None, max_pages=1):
        """按路由计划执行解析，返回执行结果（可用 limit 限制条数以控制耗时）。"""
        plan = plan or self.decide(report)                         # 未给定则现算
        targets = plan["plan"]                                     # 待执行清单
        if limit:                                                  # 限制条数
            targets = targets[:limit]
        results, t0 = [], time.time()                              # 结果与计时
        for item in targets:                                       # 逐文档执行
            if item["route"] == "human_review":                    # 人工分支不自动解析
                results.append(dict(item, parser_result={
                    "parser": "human_review", "ok": False,
                    "note": "需人工确认后再解析"}))
                continue
            parser = parsers.get_parser(item["route"])             # 取解析器
            try:
                res = parser(item["path"], self.cfg, max_pages=max_pages)
            except Exception as exc:                               # 单个失败不中断
                res = {"parser": item["route"], "ok": False,
                       "error": "%s: %s" % (type(exc).__name__, exc)}
            results.append(dict(item, parser_result=res))
        ok = sum(1 for r in results if r["parser_result"].get("ok"))  # 成功数
        return {"workflow": WORKFLOW_NAME, "node": "execute",
                "executed": len(results), "succeeded": ok,          # 统计
                "elapsed_seconds": round(time.time() - t0, 2),      # 耗时
                "results": results}

    # ---------------- 完整链路 ----------------
    def run(self, target, output_dir=None, limit=None, max_pages=1, save=True):
        """跑完整工作流：质检 → 决策 → 路由执行，并落盘执行清单。"""
        report = self.inspect(target, output_dir)                  # 节点 1
        plan = self.decide(report)                                 # 节点 2
        execution = self.execute(report, plan, limit=limit,        # 节点 3
                                 max_pages=max_pages)
        out = {"workflow": WORKFLOW_NAME,                          # 汇总输出
               "config_note": self.note,
               "route_distribution": plan["route_distribution"],   # 路由分布
               "human_review_count": len(plan["human_review"]),    # 人工复核数
               "executed": execution["executed"],                  # 执行数
               "succeeded": execution["succeeded"],                # 成功数
               "elapsed_seconds": execution["elapsed_seconds"],    # 执行耗时
               "plan": plan["plan"],                               # 路由计划
               "results": execution["results"]}                    # 执行明细
        if save:                                                   # 落盘
            path = os.path.join(self.work_dir, "workflow_run.json")
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(out, fh, ensure_ascii=False, indent=2)
            out["saved_to"] = path
        return out


def register_skill(registry_file=None):
    """把本 Skill 注册到智能体技能库：写出技能清单 JSON。

    RAGFlow 的 Agent 框架按技能名在技能库中查找可调用能力；本函数产出符合该约定的
    清单（名称/版本/描述/入口），决策节点据此调用 run_assessment。
    """
    path = registry_file or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "skill_registry.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    registry = {"skills": [dict(dqa.SKILL_META, enabled=True,       # 技能条目
                                workflow=WORKFLOW_NAME)]}
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(registry, fh, ensure_ascii=False, indent=2)
    return path
