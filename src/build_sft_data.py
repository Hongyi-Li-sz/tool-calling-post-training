"""
构造 SFT 训练数据

从 tools_schema.json 读取工具定义，基于模板 + 随机参数填充生成训练数据。
覆盖 5 种场景：
  1. 单工具调用 (~300 条)
  2. 不需要工具 / 闲聊 (~80 条)
  3. 参数缺失需要追问 (~60 条)
  4. 多样表达 / 相似意图 (~80 条)
  5. 干扰问题 / 复合请求 (~60 条)

输出格式: Alpaca 格式 (instruction/input/output)
"""
import argparse
import json
import os
import random
import sys
from datetime import datetime, timedelta

# ── 随机种子，保证可复现 ──
random.seed(42)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_schema() -> dict:
    """加载工具 Schema"""
    path = os.path.join(PROJECT_ROOT, "tools_schema.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ══════════════════════════════════════════════════════════════════
#  参数值生成器 — 为每个工具生成逼真的参数值
# ══════════════════════════════════════════════════════════════════

SEARCH_QUERIES = [
    "Python 多线程教程", "深度学习入门", "Transformer 架构详解",
    "大模型微调方法", "CUDA 编程指南", "PyTorch 基础",
    "如何优化 SQL 查询", "Docker 部署 Flask", "Redis 缓存策略",
    "Git 分支管理", "Linux 常用命令", "设计模式入门",
    "RESTful API 设计规范", "分布式系统一致性", "MySQL 索引优化",
    "Nginx 反向代理配置", "Kubernetes 入门教程", "推荐系统算法",
    "自然语言处理综述", "强化学习原理", "时间序列预测方法",
]

CALC_EXPRESSIONS = [
    "123 * 456", "15 + 28 / 2", "(100 - 25) * 3", "1024 / 8",
    "3.14 * 10 * 10", "256 + 512", "1000 - 345", "17 * 19",
    "sqrt(144)", "2 ** 10", "88 / 4 + 6", "100 * 0.85",
    "365 * 24", "9999 / 3", "45 + 55 - 20", "7 * 8 * 9",
    "1000000 / 1024", "18.5 * 2", "256 >> 2", "1024 % 128",
]

ORDER_IDS = [
    "A10293", "B20456", "C37890", "D45678", "E56789",
    "F12345", "G98765", "H34567", "I11111", "J22222",
    "ORD-2024-001", "ORD-2024-0892", "SO-240315-001", "PO-20240315-0056",
    "TBA1029384756", "X20240315001",
]

MEETING_TOPICS = [
    "项目评审", "Sprint 回顾", "技术方案讨论", "需求评审",
    "季度规划", "1v1 沟通", "产品发布会", "团队建设",
    "代码审查", "架构设计讨论", "预算评审", "客户演示",
    "年度总结", "新员工入职培训", "安全合规审查", "性能优化讨论",
]

EMAIL_RECIPIENTS = [
    "admin@company.com", "zhangsan@example.com", "hr@corp.cn",
    "manager@team.org", "support@service.com", "lisi@tech.cn",
    "wangwu@dev.io", "ceo@startup.com", "finance@corp.cn",
    "all@team.org",
]

EMAIL_SUBJECTS = [
    "系统升级通知", "会议纪要", "项目进度汇报", "请假申请",
    "预算审批", "技术方案 v2.0", "客户反馈汇总", "周报",
    "活动通知", "安全提醒",
]

EMAIL_CONTENTS = [
    "您好，系统将于今晚 22:00 进行例行升级维护，预计持续 2 小时，期间服务可能会短暂中断，请提前做好安排。",
    "各位同事，附件是本月的项目进度报告，请大家查阅。如有问题请及时反馈。",
    "尊敬的用户，您的账户将于下个月到期，请及时续费以保证服务不中断。",
    "您好，关于上次讨论的技术方案，我整理了新版本，请查看附件并提供反馈。",
    "通知：本周五下午 3 点在公司大会议室召开全员大会，请准时参加。",
]

MEETING_DATES = []
for i in range(30):
    d = datetime.now() + timedelta(days=i + 1)
    MEETING_DATES.append(d.strftime("%Y-%m-%d"))

MEETING_TIMES = [
    "09:00", "09:30", "10:00", "10:30", "11:00", "14:00",
    "14:30", "15:00", "15:30", "16:00", "16:30", "17:00",
]

# ══════════════════════════════════════════════════════════════════
#  模板定义
# ══════════════════════════════════════════════════════════════════

# ── 类型 1：单工具调用模板 ──

TOOL_TEMPLATES = {
    "search_docs": [
        # 直接检索
        "帮我搜索一下{query}",
        "查一下关于{query}的资料",
        "搜索：{query}",
        "帮我找一下{query}相关的内容",
        "检索知识库，关键词是{query}",
        "看看有没有{query}的文档",
        "在知识库里查{query}",
        "搜索文档：{query}",
        "帮我查查{query}",
        "找一下{query}的教程",
        "有没有{query}方面的资料？",
        "帮我搜{query}",
        "检索 {query}",
        "我想了解{query}，帮我搜索一下",
        "能不能帮我搜一下{query}",
        "请帮我检索{query}的相关信息",
    ],

    "calculator": [
        "计算 {expression} 等于多少",
        "{expression} 的结果是多少",
        "帮我算一下 {expression}",
        "计算：{expression}",
        "{expression} = ?",
        "求 {expression} 的值",
        "算算 {expression}",
        "帮我计算 {expression} 的结果",
        "{expression}，等于几？",
        "请计算 {expression}",
        "算一下这个表达式：{expression}",
        "{expression} 怎么算",
        "帮我求解 {expression}",
        "计算器算一下 {expression}",
        "{expression} 给我算出来",
        "把 {expression} 算出来",
    ],

    "query_order": [
        "帮我查一下订单 {order_id} 的状态",
        "查询订单 {order_id}",
        "订单 {order_id} 到哪了",
        "{order_id} 这个订单现在什么状态",
        "帮我看看 {order_id} 的物流",
        "查订单：{order_id}",
        "我想知道 {order_id} 的订单情况",
        "帮我跟踪一下 {order_id}",
        "{order_id} 订单状态是什么",
        "这个订单 {order_id} 发货了吗",
        "查一下 {order_id} 的配送进度",
        "{order_id} 什么时候能到",
        "帮我查查 {order_id} 卖家发货了没",
        "看一下 {order_id} 是不是已签收",
        "查询 {order_id} 是否已完成",
        "{order_id} 现在在哪",
    ],

    "book_meeting": [
        "帮我预约 {date} {time} 的会议，主题是{topic}",
        "定一个 {date} {time} 的{topic}会议",
        "预约会议：{date} {time}，{topic}",
        "帮我在 {date} {time} 安排{topic}",
        "我要预约一个会议，{date} {time}，关于{topic}",
        "请帮我预定 {date} 下午的{topic}会议",
        "帮我约一下 {date} {time} {topic}",
        "在 {date} {time} 帮我安排一场{topic}的会议",
        "帮我定个会议室，{date} {time}，讨论{topic}",
        "预约 {date} 的{topic}，时间{time}",
        "帮我安排 {date}{time} 的{topic}",
        "预定会议，{date}{time}开{topic}",
        "帮我约{topic}，{date}{time}",
        "帮我创建{topic}会议，{date}{time}",
        "{date} 帮我约个{topic}，{time}",
    ],

    "send_email": [
        "给 {recipient} 发一封关于{subject}的邮件",
        "发邮件给 {recipient}，主题是{subject}",
        "帮我写邮件给 {recipient}，{subject}",
        "发送邮件：收件人 {recipient}，{subject}",
        "帮我发邮件，给{recipient}，说{subject}",
        "请给 {recipient} 发{subject}的邮件",
        "向 {recipient} 发送{subject}邮件",
        "帮我给 {recipient} 发邮件，主题{subject}",
        "给 {recipient} 发{subject}",
        "写封邮件发给{recipient}，关于{subject}",
        "请帮我发送{subject}到 {recipient}",
        "给 {recipient} 去邮件：{subject}",
        "麻烦给 {recipient} 发个邮件，{subject}",
        "发邮件通知{recipient}：{subject}",
        "帮我起草邮件给 {recipient}，{subject}",
    ],
}

# ── 类型 2：不需要工具（闲聊/常识问答） ──

NO_TOOL_SAMPLES = [
    {"instruction": "你是谁？", "output": {"tool": "none", "arguments": {}}},
    {"instruction": "你好", "output": {"tool": "none", "arguments": {}}},
    {"instruction": "今天天气怎么样", "output": {"tool": "none", "arguments": {}}},
    {"instruction": "讲个笑话", "output": {"tool": "none", "arguments": {}}},
    {"instruction": "你有什么功能", "output": {"tool": "none", "arguments": {}}},
    {"instruction": "谢谢你的帮助", "output": {"tool": "none", "arguments": {}}},
    {"instruction": "Python 是什么", "output": {"tool": "none", "arguments": {}}},
    {"instruction": "北京在哪里", "output": {"tool": "none", "arguments": {}}},
    {"instruction": "解释一下什么是机器学习", "output": {"tool": "none", "arguments": {}}},
    {"instruction": "1+1等于几", "output": {"tool": "none", "arguments": {}}},
    {"instruction": "现在几点了", "output": {"tool": "none", "arguments": {}}},
    {"instruction": "推荐一本好书", "output": {"tool": "none", "arguments": {}}},
    {"instruction": "怎么学好英语", "output": {"tool": "none", "arguments": {}}},
    {"instruction": "你觉得人工智能会取代人类吗", "output": {"tool": "none", "arguments": {}}},
    {"instruction": "给我讲个睡前故事", "output": {"tool": "none", "arguments": {}}},
    {"instruction": "中国首都是哪", "output": {"tool": "none", "arguments": {}}},
    {"instruction": "请用中文回答", "output": {"tool": "none", "arguments": {}}},
    {"instruction": "给我一个建议", "output": {"tool": "none", "arguments": {}}},
    {"instruction": "你叫什么名字", "output": {"tool": "none", "arguments": {}}},
    {"instruction": "今天星期几", "output": {"tool": "none", "arguments": {}}},
    {"instruction": "你能做什么", "output": {"tool": "none", "arguments": {}}},
    {"instruction": "什么是深度学习", "output": {"tool": "none", "arguments": {}}},
    {"instruction": "中午吃什么好", "output": {"tool": "none", "arguments": {}}},
    {"instruction": "如何保持健康", "output": {"tool": "none", "arguments": {}}},
    {"instruction": "怎么学好编程", "output": {"tool": "none", "arguments": {}}},
    {"instruction": "推荐一部电影", "output": {"tool": "none", "arguments": {}}},
    {"instruction": "给我个鼓励", "output": {"tool": "none", "arguments": {}}},
    {"instruction": "周末去哪玩", "output": {"tool": "none", "arguments": {}}},
    {"instruction": "人生有什么意义", "output": {"tool": "none", "arguments": {}}},
    {"instruction": "帮我写一首诗", "output": {"tool": "none", "arguments": {}}},
    {"instruction": "你觉得哪个语言最好学", "output": {"tool": "none", "arguments": {}}},
    {"instruction": "请介绍一下自己", "output": {"tool": "none", "arguments": {}}},
    {"instruction": "地球绕太阳转一圈多久", "output": {"tool": "none", "arguments": {}}},
    {"instruction": "煮鸡蛋要多久", "output": {"tool": "none", "arguments": {}}},
    {"instruction": "怎么才能变聪明", "output": {"tool": "none", "arguments": {}}},
    {"instruction": "你有感情吗", "output": {"tool": "none", "arguments": {}}},
    {"instruction": "帮我做个决定", "output": {"tool": "none", "arguments": {}}},
    {"instruction": "新年是什么时候", "output": {"tool": "none", "arguments": {}}},
    {"instruction": "你懂音乐吗", "output": {"tool": "none", "arguments": {}}},
    {"instruction": "给我翻译成英文：你好世界", "output": {"tool": "none", "arguments": {}}},
    {"instruction": "怎么睡得好", "output": {"tool": "none", "arguments": {}}},
    {"instruction": "为什么天空是蓝色的", "output": {"tool": "none", "arguments": {}}},
    {"instruction": "帮我解释一下什么是 API", "output": {"tool": "none", "arguments": {}}},
    {"instruction": "苹果和梨哪个好吃", "output": {"tool": "none", "arguments": {}}},
    {"instruction": "如何提高工作效率", "output": {"tool": "none", "arguments": {}}},
    {"instruction": "给我打个招呼", "output": {"tool": "none", "arguments": {}}},
    {"instruction": "你能写代码吗", "output": {"tool": "none", "arguments": {}}},
    {"instruction": "有没有外星人", "output": {"tool": "none", "arguments": {}}},
    {"instruction": "推荐一首好听的歌", "output": {"tool": "none", "arguments": {}}},
    {"instruction": "比特币是什么", "output": {"tool": "none", "arguments": {}}},
    {"instruction": "怎么减肥", "output": {"tool": "none", "arguments": {}}},
    {"instruction": "请讲个故事", "output": {"tool": "none", "arguments": {}}},
    {"instruction": "你的生日是哪天", "output": {"tool": "none", "arguments": {}}},
    {"instruction": "天上有多少颗星星", "output": {"tool": "none", "arguments": {}}},
    {"instruction": "什么叫人工智能", "output": {"tool": "none", "arguments": {}}},
    {"instruction": "人为什么要吃饭", "output": {"tool": "none", "arguments": {}}},
    {"instruction": "我心情不好怎么办", "output": {"tool": "none", "arguments": {}}},
    {"instruction": "怎么网购", "output": {"tool": "none", "arguments": {}}},
    {"instruction": "做个自我介绍", "output": {"tool": "none", "arguments": {}}},
    {"instruction": "怎么学好数学", "output": {"tool": "none", "arguments": {}}},
    {"instruction": "给我推荐一本书", "output": {"tool": "none", "arguments": {}}},
    {"instruction": "你可靠吗", "output": {"tool": "none", "arguments": {}}},
    {"instruction": "帮我起个名字", "output": {"tool": "none", "arguments": {}}},
    {"instruction": "人生的意义是什么", "output": {"tool": "none", "arguments": {}}},
    {"instruction": "你能听见我说话吗", "output": {"tool": "none", "arguments": {}}},
    {"instruction": "中午了有点困", "output": {"tool": "none", "arguments": {}}},
    {"instruction": "帮我写句祝福的话", "output": {"tool": "none", "arguments": {}}},
    {"instruction": "请问怎么联系客服", "output": {"tool": "none", "arguments": {}}},
    {"instruction": "你能帮我干什么", "output": {"tool": "none", "arguments": {}}},
    {"instruction": "怎么学微积分", "output": {"tool": "none", "arguments": {}}},
    {"instruction": "有哪些好的学习资源", "output": {"tool": "none", "arguments": {}}},
    {"instruction": "什么是云计算", "output": {"tool": "none", "arguments": {}}},
    {"instruction": "帮我分析一下", "output": {"tool": "none", "arguments": {}}},
    {"instruction": "怎么提高记忆力", "output": {"tool": "none", "arguments": {}}},
    {"instruction": "推荐一个好用的笔记软件", "output": {"tool": "none", "arguments": {}}},
    {"instruction": "怎么缓解压力", "output": {"tool": "none", "arguments": {}}},
    {"instruction": "介绍一个旅游景点", "output": {"tool": "none", "arguments": {}}},
    {"instruction": "有什么好吃的推荐", "output": {"tool": "none", "arguments": {}}},
    {"instruction": "怎么学做饭", "output": {"tool": "none", "arguments": {}}},
    {"instruction": "如何理财", "output": {"tool": "none", "arguments": {}}},
    {"instruction": "可不可以帮我", "output": {"tool": "none", "arguments": {}}},
    {"instruction": "OK", "output": {"tool": "none", "arguments": {}}},
    {"instruction": "在吗", "output": {"tool": "none", "arguments": {}}},
    {"instruction": "hello", "output": {"tool": "none", "arguments": {}}},
    {"instruction": "早上好", "output": {"tool": "none", "arguments": {}}},
    {"instruction": "你能识别图片吗", "output": {"tool": "none", "arguments": {}}},
    {"instruction": "帮我解释一下相对论", "output": {"tool": "none", "arguments": {}}},
]

# ── 类型 3：参数缺失需要追问 ──

MISSING_PARAM_SAMPLES = [
    # book_meeting — 缺参数
    {"instruction": "帮我预约个会议",
     "output": {"tool": "none", "arguments": {}, "need_clarification": True,
                "question": "请提供会议日期、时间和主题。"}},
    {"instruction": "帮我定个会",
     "output": {"tool": "none", "arguments": {}, "need_clarification": True,
                "question": "请提供会议日期、时间和主题。"}},
    {"instruction": "我想预约会议",
     "output": {"tool": "none", "arguments": {}, "need_clarification": True,
                "question": "请提供会议日期、时间和主题。"}},
    {"instruction": "安排一个会议",
     "output": {"tool": "none", "arguments": {}, "need_clarification": True,
                "question": "请提供会议日期、时间和主题。"}},
    {"instruction": "下周三下午有空吗？帮我约个产品讨论",
     "output": {"tool": "none", "arguments": {}, "need_clarification": True,
                "question": "请提供会议的完整日期（YYYY-MM-DD 格式）和具体时间。"}},
    {"instruction": "明天上午10点开会",
     "output": {"tool": "none", "arguments": {}, "need_clarification": True,
                "question": "请提供会议主题。"}},
    {"instruction": "帮我约个需求评审的会",
     "output": {"tool": "none", "arguments": {}, "need_clarification": True,
                "question": "请提供会议日期和时间。"}},
    {"instruction": "约个周五的会",
     "output": {"tool": "none", "arguments": {}, "need_clarification": True,
                "question": "请提供会议的具体日期（YYYY-MM-DD 格式）、时间和主题。"}},
    {"instruction": "下午3点帮我订会议室",
     "output": {"tool": "none", "arguments": {}, "need_clarification": True,
                "question": "请提供会议日期和主题。"}},

    # send_email — 缺参数
    {"instruction": "帮我发封邮件",
     "output": {"tool": "none", "arguments": {}, "need_clarification": True,
                "question": "请提供收件人邮箱、邮件主题和内容。"}},
    {"instruction": "帮我给经理发个邮件",
     "output": {"tool": "none", "arguments": {}, "need_clarification": True,
                "question": "请提供收件人的完整邮箱地址、邮件主题和内容。"}},
    {"instruction": "发一封关于请假申请的邮件",
     "output": {"tool": "none", "arguments": {}, "need_clarification": True,
                "question": "请提供收件人邮箱和邮件内容。"}},
    {"instruction": "帮我通知一下团队",
     "output": {"tool": "none", "arguments": {}, "need_clarification": True,
                "question": "请提供收件人邮箱、通知主题和具体内容。"}},
    {"instruction": "发邮件",
     "output": {"tool": "none", "arguments": {}, "need_clarification": True,
                "question": "请提供收件人邮箱、邮件主题和内容。"}},
    {"instruction": "帮我写个邮件给 HR",
     "output": {"tool": "none", "arguments": {}, "need_clarification": True,
                "question": "请提供收件人的完整邮箱地址、邮件主题和内容。"}},
    {"instruction": "发周报给领导",
     "output": {"tool": "none", "arguments": {}, "need_clarification": True,
                "question": "请提供收件人邮箱和邮件内容。"}},
    {"instruction": "邮件通知全组",
     "output": {"tool": "none", "arguments": {}, "need_clarification": True,
                "question": "请提供收件人邮箱、邮件主题和内容。"}},

    # query_order — 缺参数
    {"instruction": "帮我查一下订单",
     "output": {"tool": "none", "arguments": {}, "need_clarification": True,
                "question": "请提供订单号。"}},
    {"instruction": "查订单状态",
     "output": {"tool": "none", "arguments": {}, "need_clarification": True,
                "question": "请提供订单号。"}},
    {"instruction": "我的订单到哪了",
     "output": {"tool": "none", "arguments": {}, "need_clarification": True,
                "question": "请提供订单号。"}},
    {"instruction": "看看我的快递",
     "output": {"tool": "none", "arguments": {}, "need_clarification": True,
                "question": "请提供订单号。"}},
    {"instruction": "物流跟踪",
     "output": {"tool": "none", "arguments": {}, "need_clarification": True,
                "question": "请提供订单号。"}},
    {"instruction": "订单怎么样了",
     "output": {"tool": "none", "arguments": {}, "need_clarification": True,
                "question": "请提供订单号。"}},
    {"instruction": "我买的那个东西发货了吗",
     "output": {"tool": "none", "arguments": {}, "need_clarification": True,
                "question": "请提供订单号。"}},
    {"instruction": "查询我的购买记录",
     "output": {"tool": "none", "arguments": {}, "need_clarification": True,
                "question": "请提供订单号。如果您不记得，可以告诉我大致的时间和商品名称帮您查找。"}},

    # search_docs — 缺参数
    {"instruction": "帮我搜索一下",
     "output": {"tool": "none", "arguments": {}, "need_clarification": True,
                "question": "请提供搜索关键词。"}},
    {"instruction": "检索知识库",
     "output": {"tool": "none", "arguments": {}, "need_clarification": True,
                "question": "请提供搜索关键词或问题。"}},
    {"instruction": "搜点东西",
     "output": {"tool": "none", "arguments": {}, "need_clarification": True,
                "question": "请提供搜索关键词。"}},
    {"instruction": "查文档",
     "output": {"tool": "none", "arguments": {}, "need_clarification": True,
                "question": "请提供搜索关键词。"}},

    # calculator — 缺参数
    {"instruction": "帮我算一下",
     "output": {"tool": "none", "arguments": {}, "need_clarification": True,
                "question": "请提供需要计算的数学表达式。"}},
    {"instruction": "算个数",
     "output": {"tool": "none", "arguments": {}, "need_clarification": True,
                "question": "请提供需要计算的数学表达式。"}},
    {"instruction": "计算器",
     "output": {"tool": "none", "arguments": {}, "need_clarification": True,
                "question": "请提供需要计算的数学表达式。"}},
    {"instruction": "做一道计算题",
     "output": {"tool": "none", "arguments": {}, "need_clarification": True,
                "question": "请提供需要计算的数学表达式。"}},
    # 更多追问场景
    {"instruction": "帮我订个下周三的会",
     "output": {"tool": "none", "arguments": {}, "need_clarification": True,
                "question": "请提供会议的具体时间（HH:MM 格式）和主题。"}},
    {"instruction": "通知一下大家",
     "output": {"tool": "none", "arguments": {}, "need_clarification": True,
                "question": "请提供收件人邮箱、通知主题和内容。"}},
    {"instruction": "查个东西",
     "output": {"tool": "none", "arguments": {}, "need_clarification": True,
                "question": "请问您要查什么？是资料搜索还是订单查询？"}},
    {"instruction": "那个表达式帮我算算",
     "output": {"tool": "none", "arguments": {}, "need_clarification": True,
                "question": "请提供需要计算的数学表达式。"}},
    {"instruction": "发个通知邮件",
     "output": {"tool": "none", "arguments": {}, "need_clarification": True,
                "question": "请提供收件人邮箱、通知主题和内容。"}},
    {"instruction": "帮忙预约",
     "output": {"tool": "none", "arguments": {}, "need_clarification": True,
                "question": "请告诉我您要预约什么类型的会面，以及日期、时间等信息。"}},
    {"instruction": "给我查",
     "output": {"tool": "none", "arguments": {}, "need_clarification": True,
                "question": "请问您要查什么？"}},
]

# ── 类型 5：干扰问题 / 复合请求 ──

DISTRACTING_SAMPLES = [
    # 两件事混在一起说 → 选第一个/最主要的
    {"instruction": "帮我发邮件顺便查订单 A10293",
     "output": {"tool": "send_email", "arguments": {
         "recipient": "manager@team.org", "subject": "通知",
         "content": "请查收通知内容。"}}},
    {"instruction": "搜一下 Python 教程，然后帮我发个邮件",
     "output": {"tool": "search_docs", "arguments": {"query": "Python 教程"}}},
    {"instruction": "先帮我算 100*50，然后订个会议室",
     "output": {"tool": "calculator", "arguments": {"expression": "100 * 50"}}},
    {"instruction": "查订单 B20456 的同时帮我检索一下退货政策",
     "output": {"tool": "query_order", "arguments": {"order_id": "B20456"}}},
    {"instruction": "帮我查一下有关深度学习的资料，顺便算个 999/3",
     "output": {"tool": "search_docs", "arguments": {"query": "深度学习"}}},
    {"instruction": "我有两件事：订一个明天的项目评审会，还有查订单 C37890",
     "output": {"tool": "book_meeting", "arguments": {
         "date": (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d"),
         "time": "14:00", "topic": "项目评审"}}},
    # 模糊意图
    {"instruction": "帮我查东西",
     "output": {"tool": "none", "arguments": {}, "need_clarification": True,
                "question": "请问您要查什么？是订单、知识库资料，还是其他？"}},
    {"instruction": "帮我处理一下",
     "output": {"tool": "none", "arguments": {}, "need_clarification": True,
                "question": "请问您具体需要什么帮助？我可以帮您搜索资料、查订单、预约会议、发邮件或做计算。"}},
    {"instruction": "有件事要你帮忙",
     "output": {"tool": "none", "arguments": {}, "need_clarification": True,
                "question": "请告诉我您需要什么帮助。"}},
    {"instruction": "我需要帮助",
     "output": {"tool": "none", "arguments": {}, "need_clarification": True,
                "question": "请问您具体需要什么帮助？"}},
    {"instruction": "帮我搜之前先确认一下你是不是最新的版本",
     "output": {"tool": "none", "arguments": {}}},
    # 说了工具名但不完整
    {"instruction": "用 send_email 功能发通知",
     "output": {"tool": "none", "arguments": {}, "need_clarification": True,
                "question": "请提供收件人邮箱、邮件主题和内容。"}},
    {"instruction": "调用 search_docs",
     "output": {"tool": "none", "arguments": {}, "need_clarification": True,
                "question": "请提供搜索关键词。"}},
    {"instruction": "用 calculator 算东西",
     "output": {"tool": "none", "arguments": {}, "need_clarification": True,
                "question": "请提供需要计算的数学表达式。"}},
    # 更多复合/模糊请求
    {"instruction": "我想查订单但是不太记得单号了",
     "output": {"tool": "none", "arguments": {}, "need_clarification": True,
                "question": "您可以尝试在订单列表里查找，或者提供一些订单的关键信息（如购买时间、商品名称），我来帮您匹配。"}},
    {"instruction": "帮我查一下最近的知识库更新",
     "output": {"tool": "search_docs", "arguments": {"query": "知识库更新"}}},
    {"instruction": "计算完 500+200 后再发邮件",
     "output": {"tool": "calculator", "arguments": {"expression": "500 + 200"}}},
    {"instruction": "你除了查订单还能干啥",
     "output": {"tool": "none", "arguments": {}}},
    {"instruction": "我不知道该用什么功能",
     "output": {"tool": "none", "arguments": {}}},
    {"instruction": "帮我干个活",
     "output": {"tool": "none", "arguments": {}, "need_clarification": True,
                "question": "请问您具体需要什么帮助？"}},
    # 工具名直接出现在指令中，但可能有歧义
    {"instruction": "用 search 查一下 Docker 教程",
     "output": {"tool": "search_docs", "arguments": {"query": "Docker 教程"}}},
    {"instruction": "计算 3*5+7",
     "output": {"tool": "calculator", "arguments": {"expression": "3 * 5 + 7"}}},
    {"instruction": "订单查询：A10293",
     "output": {"tool": "query_order", "arguments": {"order_id": "A10293"}}},
    {"instruction": "email zhangsan@example.com 关于项目进度",
     "output": {"tool": "send_email", "arguments": {
         "recipient": "zhangsan@example.com", "subject": "项目进度",
         "content": "请查收项目进度汇报。"}}},
    {"instruction": "meeting 明天项目讨论",
     "output": {"tool": "book_meeting", "arguments": {
         "date": (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d"),
         "time": "10:00", "topic": "项目讨论"}}},
    {"instruction": "docs Python 异步编程",
     "output": {"tool": "search_docs", "arguments": {"query": "Python 异步编程"}}},
]

# ══════════════════════════════════════════════════════════════════
#  数据生成函数
# ══════════════════════════════════════════════════════════════════


def pick_value(pool: list, exclude: set = None) -> str:
    """从候选池中随机选值，可选排除已选中的"""
    if exclude and len(pool) > len(exclude):
        available = [v for v in pool if v not in exclude]
        return random.choice(available)
    return random.choice(pool)


def generate_single_tool_samples(tool_name: str, num: int = 80) -> list[dict]:
    """为单个工具生成多样化的单工具调用样本"""
    templates = TOOL_TEMPLATES[tool_name]
    samples = []

    for _ in range(num):
        template = random.choice(templates)

        if tool_name == "search_docs":
            query = pick_value(SEARCH_QUERIES)
            instruction = template.format(query=query)
            output = {"tool": "search_docs", "arguments": {"query": query}}

        elif tool_name == "calculator":
            expression = pick_value(CALC_EXPRESSIONS)
            instruction = template.format(expression=expression)
            output = {"tool": "calculator", "arguments": {"expression": expression}}

        elif tool_name == "query_order":
            order_id = pick_value(ORDER_IDS)
            instruction = template.format(order_id=order_id)
            output = {"tool": "query_order", "arguments": {"order_id": order_id}}

        elif tool_name == "book_meeting":
            date = pick_value(MEETING_DATES)
            time = pick_value(MEETING_TIMES)
            topic = pick_value(MEETING_TOPICS)
            instruction = template.format(date=date, time=time, topic=topic)
            output = {"tool": "book_meeting", "arguments": {
                "date": date, "time": time, "topic": topic}}

        elif tool_name == "send_email":
            recipient = pick_value(EMAIL_RECIPIENTS)
            subject = pick_value(EMAIL_SUBJECTS)
            content = pick_value(EMAIL_CONTENTS)
            instruction = template.format(recipient=recipient, subject=subject)
            output = {"tool": "send_email", "arguments": {
                "recipient": recipient, "subject": subject, "content": content}}

        samples.append({
            "instruction": instruction,
            "input": "",
            "output": json.dumps(output, ensure_ascii=False),
        })

    return samples


def generate_all_data() -> list[dict]:
    """生成全部 SFT 数据，覆盖 5 种场景"""
    schema = load_schema()
    tool_names = [t["name"] for t in schema["tools"]]
    all_data = []

    # ── 类型 1：单工具调用 (5 × 60 = 300 条) ──
    print("生成单工具调用数据...")
    for tool_name in tool_names:
        samples = generate_single_tool_samples(tool_name, num=80)
        all_data.extend(samples)
        print(f"  {tool_name}: {len(samples)} 条")

    # ── 类型 2：不需要工具 (80 条) ──
    print(f"生成不需要工具的数据: {len(NO_TOOL_SAMPLES)} 条")
    for item in NO_TOOL_SAMPLES:
        all_data.append({
            "instruction": item["instruction"],
            "input": "",
            "output": json.dumps(item["output"], ensure_ascii=False),
        })

    # ── 类型 3：参数缺失需要追问 (60 条) ──
    print(f"生成参数缺失追问数据: {len(MISSING_PARAM_SAMPLES)} 条")
    for item in MISSING_PARAM_SAMPLES:
        all_data.append({
            "instruction": item["instruction"],
            "input": "",
            "output": json.dumps(item["output"], ensure_ascii=False),
        })

    # ── 类型 4：纯 JSON 格式（额外新增 20 条确定性的格式样本）─
    #  这些样本专门强化模型对 JSON 输出格式的学习
    format_reinforce = [
        ("帮我查一下订单 D45678", {"tool": "query_order", "arguments": {"order_id": "D45678"}}),
        ("计算 256 + 512", {"tool": "calculator", "arguments": {"expression": "256 + 512"}}),
        ("搜索 Transformer 架构", {"tool": "search_docs", "arguments": {"query": "Transformer 架构"}}),
        ("预约 2025-01-15 10:00 技术方案讨论",
         {"tool": "book_meeting", "arguments": {"date": "2025-01-15", "time": "10:00", "topic": "技术方案讨论"}}),
        ("给 hr@corp.cn 发请假申请",
         {"tool": "send_email", "arguments": {"recipient": "hr@corp.cn", "subject": "请假申请",
                                                "content": "您好，我因身体不适需要请假一天。"}}),
        ("你是 AI 吗", {"tool": "none", "arguments": {}}),
        ("搜一下大模型微调", {"tool": "search_docs", "arguments": {"query": "大模型微调"}}),
        ("3.14 * 10 等于多少", {"tool": "calculator", "arguments": {"expression": "3.14 * 10"}}),
        ("查物流 ORD-2024-0892", {"tool": "query_order", "arguments": {"order_id": "ORD-2024-0892"}}),
        ("安排周五下午 Sprint 回顾", {"tool": "book_meeting", "arguments": {
            "date": "2025-02-28", "time": "14:00", "topic": "Sprint 回顾"}}),
        ("通知 all@team.org 周末团建",
         {"tool": "send_email", "arguments": {"recipient": "all@team.org", "subject": "周末团建",
                                                "content": "各位同事，本周六组织团建活动，请大家报名参加。"}}),
        ("给我讲个笑话", {"tool": "none", "arguments": {}}),
        ("Python 多线程怎么用", {"tool": "search_docs", "arguments": {"query": "Python 多线程"}}),
        ("1024 除以 8 是多少", {"tool": "calculator", "arguments": {"expression": "1024 / 8"}}),
        ("订单 PO-20240315-0056 怎样了", {"tool": "query_order", "arguments": {"order_id": "PO-20240315-0056"}}),
        ("下周一早上九点约预算评审", {"tool": "book_meeting", "arguments": {
            "date": "2025-03-03", "time": "09:00", "topic": "预算评审"}}),
        ("发一封系统维护通知给 support@service.com",
         {"tool": "send_email", "arguments": {"recipient": "support@service.com", "subject": "系统维护通知",
                                                "content": "系统将于本周六凌晨2点进行维护，预计4小时完成。"}}),
        ("你好啊", {"tool": "none", "arguments": {}}),
        ("查资料：深度学习入门", {"tool": "search_docs", "arguments": {"query": "深度学习入门"}}),
        ("帮我解 17 * 19", {"tool": "calculator", "arguments": {"expression": "17 * 19"}}),
    ]
    for instruction, output in format_reinforce:
        all_data.append({
            "instruction": instruction,
            "input": "",
            "output": json.dumps(output, ensure_ascii=False),
        })

    # ── 类型 5：干扰问题 / 复合请求 ──
    print(f"生成干扰/复合请求数据: {len(DISTRACTING_SAMPLES)} 条")
    for item in DISTRACTING_SAMPLES:
        all_data.append({
            "instruction": item["instruction"],
            "input": "",
            "output": json.dumps(item["output"], ensure_ascii=False),
        })

    # ── 打乱顺序 ──
    random.shuffle(all_data)

    return all_data


def validate_sample(sample: dict, index: int) -> tuple[bool, str]:
    """验证单条数据"""
    # 必须有三个字段
    for field in ["instruction", "input", "output"]:
        if field not in sample:
            return False, f"#{index}: 缺少字段 {field}"

    # output 必须是合法 JSON
    try:
        parsed = json.loads(sample["output"])
    except json.JSONDecodeError as e:
        return False, f"#{index}: output 不是合法 JSON — {e}"

    # 必须有 tool 字段
    if "tool" not in parsed:
        return False, f"#{index}: output 缺少 tool 字段"

    # 必须有 arguments 字段
    if "arguments" not in parsed:
        return False, f"#{index}: output 缺少 arguments 字段"

    return True, f"#{index}: ✅ ok"


def main():
    parser = argparse.ArgumentParser(description="构造 SFT 训练数据")
    parser.add_argument("--output", type=str, default=None,
                        help="输出文件路径（默认 data/sft/train.json）")
    parser.add_argument("--validate", action="store_true", default=True,
                        help="生成后验证数据")
    parser.add_argument("--sample_count", type=int, default=50,
                        help="抽查条数")
    args = parser.parse_args()

    # 生成数据
    print("=" * 60)
    print("开始构造 SFT 训练数据")
    print("=" * 60)
    all_data = generate_all_data()

    output_path = args.output or os.path.join(
        PROJECT_ROOT, "data", "sft", "train.json")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 数据已保存到: {output_path}")
    print(f"   总计: {len(all_data)} 条")

    # ── 统计分布 ──
    tool_counts = {}
    for s in all_data:
        try:
            tool = json.loads(s["output"]).get("tool", "unknown")
        except Exception:
            tool = "json_parse_error"
        tool_counts[tool] = tool_counts.get(tool, 0) + 1

    print(f"\n📊 工具分布:")
    for tool, count in sorted(tool_counts.items(), key=lambda x: -x[1]):
        print(f"   {tool}: {count} 条 ({count/len(all_data)*100:.1f}%)")

    # ── 验证 ──
    if args.validate:
        print(f"\n{'=' * 60}")
        print(f"随机抽查 {args.sample_count} 条数据验证...")
        print(f"{'=' * 60}")

        indices = random.sample(range(len(all_data)), min(args.sample_count, len(all_data)))
        errors = []
        for idx in indices:
            ok, msg = validate_sample(all_data[idx], idx)
            if not ok:
                errors.append(msg)

        if errors:
            print(f"\n❌ 发现 {len(errors)} 条错误:")
            for e in errors:
                print(f"   {e}")
        else:
            print(f"\n✅ 抽查 {len(indices)} 条全部通过！")

    return all_data


if __name__ == "__main__":
    main()
