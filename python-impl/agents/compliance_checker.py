"""
合规审查 Agent，负责校园客服场景下的隐私、安全与越权承诺检查。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from tracing.otel_config import trace_agent_call


@dataclass
class ComplianceResult:
    """合规审查结果。"""

    passed: bool
    risk_level: str
    violations: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    sanitized_content: str = ""


SENSITIVE_PATTERNS = {
    "phone": r"1[3-9]\d{9}",
    "id_card": r"\d{17}[\dXx]",
    "student_id": r"\b20\d{6,10}\b",
    "email": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
}

FORBIDDEN_TERMS = [
    "保证录取",
    "保证通过",
    "代抢课",
    "出售学生信息",
    "泄露成绩",
    "私下转账",
]

COMPLIANCE_SYSTEM_PROMPT = """你是校园客服合规审查 Agent，负责检查回复内容是否存在风险。

审查维度：
1. 是否泄露学生个人隐私，如手机号、身份证号、学号、邮箱
2. 是否包含越权承诺，如保证录取、保证奖学金结果、直接修改成绩、私下代办
3. 是否给出危险、违规或不当指引
4. 是否存在歧视、侮辱或不适合校园服务场景的表达
5. 是否建议用户绕过正式流程、绕开学校制度

请以 JSON 返回：
{
  "passed": true,
  "risk_level": "low|medium|high|critical",
  "violations": ["问题描述"],
  "suggestions": ["修正建议"]
}
"""


class ComplianceCheckerAgent:
    """合规审查 Agent。"""

    def __init__(self, llm: ChatOpenAI):
        self.llm = llm

    def _rule_based_check(self, content: str) -> list[str]:
        violations = []

        for term in FORBIDDEN_TERMS:
            if term in content:
                violations.append(f"包含违规或越权表达: '{term}'")

        for pii_type, pattern in SENSITIVE_PATTERNS.items():
            if re.search(pattern, content):
                label = {
                    "phone": "手机号",
                    "id_card": "身份证号",
                    "student_id": "学号",
                    "email": "邮箱地址",
                }.get(pii_type, pii_type)
                violations.append(f"检测到隐私信息暴露: {label}")

        return violations

    def _mask_pii(self, content: str) -> str:
        masked = content

        for pattern in SENSITIVE_PATTERNS.values():
            def _mask_match(match: re.Match) -> str:
                text = match.group()
                if len(text) <= 4:
                    return "****"
                return text[:3] + "*" * (len(text) - 6) + text[-3:]

            masked = re.sub(pattern, _mask_match, masked)

        return masked

    @trace_agent_call("compliance_rule_check")
    async def rule_check(self, content: str) -> ComplianceResult:
        violations = self._rule_based_check(content)
        sanitized = self._mask_pii(content)

        if not violations:
            return ComplianceResult(
                passed=True,
                risk_level="low",
                sanitized_content=sanitized,
            )

        has_pii = any("隐私信息" in item for item in violations)
        has_forbidden = any("违规或越权" in item for item in violations)

        if has_pii and has_forbidden:
            risk_level = "critical"
        elif has_pii or has_forbidden:
            risk_level = "high"
        else:
            risk_level = "medium"

        return ComplianceResult(
            passed=False,
            risk_level=risk_level,
            violations=violations,
            sanitized_content=sanitized,
        )

    @trace_agent_call("compliance_llm_check")
    async def llm_check(self, content: str) -> ComplianceResult:
        messages = [
            SystemMessage(content=COMPLIANCE_SYSTEM_PROMPT),
            HumanMessage(content=f"请审查以下校园客服回复内容：\n\n{content}"),
        ]
        response = await self.llm.ainvoke(messages)

        import json

        try:
            result = json.loads(response.content)
        except json.JSONDecodeError:
            return ComplianceResult(
                passed=True,
                risk_level="low",
                sanitized_content=self._mask_pii(content),
            )

        return ComplianceResult(
            passed=result.get("passed", True),
            risk_level=result.get("risk_level", "low"),
            violations=result.get("violations", []),
            suggestions=result.get("suggestions", []),
            sanitized_content=self._mask_pii(content),
        )

    @trace_agent_call("compliance_full_check")
    async def full_check(self, content: str) -> ComplianceResult:
        rule_result = await self.rule_check(content)
        if not rule_result.passed and rule_result.risk_level in ("high", "critical"):
            return rule_result

        llm_result = await self.llm_check(content)
        all_violations = rule_result.violations + llm_result.violations
        final_passed = rule_result.passed and llm_result.passed
        risk_priority = {"low": 0, "medium": 1, "high": 2, "critical": 3}
        final_risk = max(
            rule_result.risk_level,
            llm_result.risk_level,
            key=lambda item: risk_priority.get(item, 0),
        )

        return ComplianceResult(
            passed=final_passed,
            risk_level=final_risk,
            violations=all_violations,
            suggestions=llm_result.suggestions,
            sanitized_content=rule_result.sanitized_content,
        )

    @trace_agent_call("compliance_process")
    async def process(self, state: dict[str, Any]) -> dict[str, Any]:
        sub_results = state.get("sub_results", {})

        content_to_check = "\n".join(
            result for result in sub_results.values() if isinstance(result, str)
        )
        if not content_to_check.strip():
            return {**state, "compliance_passed": True}

        compliance_result = await self.full_check(content_to_check)
        if not compliance_result.passed:
            for key, value in list(sub_results.items()):
                if isinstance(value, str):
                    sub_results[key] = compliance_result.sanitized_content

        return {
            **state,
            "compliance_passed": compliance_result.passed,
            "sub_results": {
                **sub_results,
                "compliance": {
                    "passed": compliance_result.passed,
                    "risk_level": compliance_result.risk_level,
                    "violations": compliance_result.violations,
                },
            },
        }
