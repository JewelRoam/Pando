"""Build a deterministic multi-domain prompt pool for RLCD data collection.

Generated rows intentionally omit ground-truth labels: the live lineage
majority supplies the RLCD preference/rejection signal. Evaluation continues
to use a separate, manually labeled holdout suite.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

QUESTIONS = {
    "intent": {
        "instructions": "Classify the intent of the message.",
        "criteria": ["refund", "complaint", "greeting", "login_issue",
                     "invoice", "praise"],
    },
    "risk": {
        "instructions": "Classify the risk level.",
        "criteria": ["negligible", "low", "moderate", "high", "critical"],
    },
    "urgent": {
        "instructions": "Is this message urgent?",
        "criteria": ["true", "false"],
    },
    "moderation": {
        "instructions": "Classify the moderation outcome.",
        "criteria": ["ok", "spam", "abuse"],
    },
    "route": {
        "instructions": "Choose the correct handling route.",
        "criteria": ["standard_reply", "human_agent", "retry_payment",
                     "escalate"],
    },
    "action": {
        "instructions": "Choose the operational action.",
        "criteria": ["ignore", "scale_up", "rollback", "emergency_stop"],
    },
}

STATES = {
    "refund": [
        "User message: I was charged twice for the same order",
        "客户说订单取消后仍未退款",
        "Cliente: no recibí el producto y quiero mi dinero",
        "Kunde: Die Bestellung kam nie an, bitte erstatten",
        "客户投诉到货商品破损，要求退款",
        "Customer: the subscription renewed after I cancelled it",
    ],
    "security": [
        "Email from unknown sender asks for the account password",
        "登录日志显示同一账号在五个国家出现",
        "Suspicious link received in a payroll message",
        "El correo afirma ser del banco y pide datos de tarjeta",
        "管理员收到从未申请过的密码重置通知",
        "New device login from an unusual location at 03:00",
    ],
    "ops": [
        "Latency p95 has exceeded the SLO for ten minutes",
        "数据库连接数持续达到上限",
        "Deployment completed but error rate doubled",
        "Primary region reports elevated packet loss",
        "支付服务连续三次健康检查失败",
        "Queue backlog crossed one million messages",
    ],
    "moderation": [
        "Post: buy cheap followers now!!!",
        "评论包含针对个人的侮辱和威胁",
        "Message repeatedly advertises a crypto giveaway",
        "El comentario contiene amenazas personales",
        "帖子冒充官方客服索取验证码",
        "Account posts the same sales link every minute",
    ],
    "support": [
        "User asks how to change the delivery address",
        "客户询问发票在哪里下载",
        "User cannot log in after changing phones",
        "El usuario pregunta por el horario de atención",
        "客户询问如何关闭自动续费",
        "User asks whether the warranty covers water damage",
    ],
}

DOMAIN_QUESTIONS = {
    "refund": ["intent", "risk", "urgent"],
    "security": ["action", "risk", "urgent"],
    "ops": ["action", "risk", "urgent"],
    "moderation": ["moderation", "intent"],
    "support": ["route", "urgent"],
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", default=str(ROOT / "assets" / "prompt_pool_expanded.jsonl"))
    args = parser.parse_args()

    rows = []
    index = 0
    for domain, states in STATES.items():
        for state in states:
            for question_id in DOMAIN_QUESTIONS[domain]:
                definition = QUESTIONS[question_id]
                index += 1
                rows.append({
                    "task_id": f"x{index:03d}",
                    "domain": domain,
                    "checkpoint": "expanded",
                    "qtype": "choice",
                    "state": state,
                    "question": question_id,
                    "candidates": definition["criteria"],
                    "instructions": definition["instructions"],
                })
    Path(args.out).write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8")
    print(f"wrote {len(rows)} prompts to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
