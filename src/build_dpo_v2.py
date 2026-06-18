"""
构造 DPO-v2 偏好数据

围绕 SFT-v2 的 Bad Cases 精准构造：
  1. 抑制过度追问 (~140对) — 修复 74.6% 过度追问率
  2. 参数缺失必须追问 (~100对) — 进一步提升 42.2% 追问准确率
  3. 保持 JSON 格式 (~60对) — 防止 DPO 格式回退
  4. 语义干扰拒调用 (~80对) — 提升 23.3% 语义干扰准确率
  5. 工具选择正确 (~40对) — 保持单工具调用能力

目标 400-500 对，保存到 data/dpo/train_v2.json
"""
import argparse
import json
import os
import random
import sys

random.seed(42)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ═══════════════════════════════════════════════════════════════
# 类型1: 抑制过度追问 (~140对)
# chosen = 简洁拒绝, rejected = 多余追问
# ═══════════════════════════════════════════════════════════════

ANTI_OVERCLARIFY_PAIRS = [
    # 问候
    ("你好",
     {"tool": "none", "arguments": {}},
     {"tool": "none", "arguments": {}, "need_clarification": True, "question": "请问你是谁？"}),
    ("早上好",
     {"tool": "none", "arguments": {}},
     {"tool": "none", "arguments": {}, "need_clarification": True, "question": "请补充更多信息。"}),
    ("嗨",
     {"tool": "none", "arguments": {}},
     {"tool": "none", "arguments": {}, "need_clarification": True, "question": "请问您需要什么帮助？"}),
    ("哈喽",
     {"tool": "none", "arguments": {}},
     {"tool": "none", "arguments": {}, "need_clarification": True, "question": "请问您是客户还是客服？"}),
    # 闲聊
    ("讲个笑话",
     {"tool": "none", "arguments": {}},
     {"tool": "none", "arguments": {}, "need_clarification": True, "question": "请问您想听什么类型的笑话？"}),
    ("推荐一部电影",
     {"tool": "none", "arguments": {}},
     {"tool": "none", "arguments": {}, "need_clarification": True, "question": "请问您喜欢什么类型的电影？"}),
    ("周末去哪玩",
     {"tool": "none", "arguments": {}},
     {"tool": "none", "arguments": {}, "need_clarification": True, "question": "请提供您所在的城市。"}),
    ("中午吃什么",
     {"tool": "none", "arguments": {}},
     {"tool": "none", "arguments": {}, "need_clarification": True, "question": "请问您的口味偏好是什么？"}),
    ("今天天气怎么样",
     {"tool": "none", "arguments": {}},
     {"tool": "none", "arguments": {}, "need_clarification": True, "question": "请提供城市名称。"}),
    ("怎么减肥",
     {"tool": "none", "arguments": {}},
     {"tool": "none", "arguments": {}, "need_clarification": True, "question": "请提供您的身高体重信息。"}),
    # 常识
    ("北京在哪里",
     {"tool": "none", "arguments": {}},
     {"tool": "none", "arguments": {}, "need_clarification": True, "question": "请问您想知道具体哪个区？"}),
    ("Python是什么",
     {"tool": "none", "arguments": {}},
     {"tool": "none", "arguments": {}, "need_clarification": True, "question": "请问您想了解 Python 的哪些方面？"}),
    ("什么是区块链",
     {"tool": "none", "arguments": {}},
     {"tool": "none", "arguments": {}, "need_clarification": True, "question": "请问您想了解区块链的哪个方面？"}),
    ("地球绕太阳一圈多久",
     {"tool": "none", "arguments": {}},
     {"tool": "none", "arguments": {}, "need_clarification": True, "question": "请问您指的是公转周期还是自转周期？"}),
    # 自我介绍
    ("你是谁",
     {"tool": "none", "arguments": {}},
     {"tool": "none", "arguments": {}, "need_clarification": True, "question": "请问您是想了解我的功能还是我的身份？"}),
    ("介绍一下你自己",
     {"tool": "none", "arguments": {}},
     {"tool": "none", "arguments": {}, "need_clarification": True, "question": "请问您想了解哪方面？"}),
    ("你有什么功能",
     {"tool": "none", "arguments": {}},
     {"tool": "none", "arguments": {}, "need_clarification": True, "question": "请问您想使用什么功能？"}),
    # 礼貌用语
    ("谢谢",
     {"tool": "none", "arguments": {}},
     {"tool": "none", "arguments": {}, "need_clarification": True, "question": "请问还需要什么帮助？"}),
    ("OK",
     {"tool": "none", "arguments": {}},
     {"tool": "none", "arguments": {}, "need_clarification": True, "question": "请问您需要我继续做什么？"}),
    ("好的",
     {"tool": "none", "arguments": {}},
     {"tool": "none", "arguments": {}, "need_clarification": True, "question": "请问还需要其他帮助吗？"}),
    # 写作
    ("帮我写一首诗",
     {"tool": "none", "arguments": {}},
     {"tool": "none", "arguments": {}, "need_clarification": True, "question": "请问您想要什么体裁的诗？"}),
    ("翻译Hello World到中文",
     {"tool": "none", "arguments": {}},
     {"tool": "none", "arguments": {}, "need_clarification": True, "question": "请问您想翻译成简体中文还是繁体中文？"}),
    # 开放式建议
    ("怎么学好英语",
     {"tool": "none", "arguments": {}},
     {"tool": "none", "arguments": {}, "need_clarification": True, "question": "请问您目前的英语水平如何？"}),
    ("如何提高效率",
     {"tool": "none", "arguments": {}},
     {"tool": "none", "arguments": {}, "need_clarification": True, "question": "请问您指的是工作效率还是学习效率？"}),
    # 元问题
    ("AI会取代人类吗",
     {"tool": "none", "arguments": {}},
     {"tool": "none", "arguments": {}, "need_clarification": True, "question": "请问您想了解哪个行业的AI影响？"}),
    ("你有感情吗",
     {"tool": "none", "arguments": {}},
     {"tool": "none", "arguments": {}, "need_clarification": True, "question": "请问您是想了解AI的局限性吗？"}),
]

# 批量生成更多反追问样本
_more_anti_overclarify = [
    "今天星期几", "现在几点了", "晚上好", "再见", "晚安",
    "你真厉害", "你好聪明", "不错", "很有意思",
    "如何在面试中表现出色", "怎么准备技术面试",
    "前端和后端哪个好", "怎么学编程", "推荐一个课程",
    "你叫什么名字", "你能做什么", "你的生日是哪天",
    "给我一个建议", "帮我做个决定", "讲个故事",
    "推荐一本书", "怎么学好数学", "如何理财",
    "比特币值得投资吗", "什么是云计算", "怎么网购",
    "人为什么要工作", "幸福是什么", "什么是成功",
    "怎么才能变聪明", "推荐好用的app", "有没有外星人",
    "你怎么看人类", "你会写代码吗", "你可以上网吗",
    "你有什么缺点", "你的训练数据有什么", "你有记忆吗",
]
for q in _more_anti_overclarify:
    ANTI_OVERCLARIFY_PAIRS.append(
        (q, {"tool": "none", "arguments": {}},
         {"tool": "none", "arguments": {}, "need_clarification": True,
          "question": "请提供更多细节以便我更好地帮助你。"}))

# ═══════════════════════════════════════════════════════════════
# 类型2: 参数缺失必须追问 (~100对)
# chosen = 正确追问, rejected = 直接调用工具（编造参数）
# ═══════════════════════════════════════════════════════════════

MUST_CLARIFY_PAIRS = [
    # search_docs 缺 query
    ("帮我搜索一下",
     {"tool": "none", "arguments": {}, "need_clarification": True, "question": "请提供搜索关键词。"},
     {"tool": "search_docs", "arguments": {"query": "搜索"}}),
    ("检索知识库",
     {"tool": "none", "arguments": {}, "need_clarification": True, "question": "请提供搜索关键词或问题。"},
     {"tool": "search_docs", "arguments": {"query": "知识库"}}),
    ("搜点东西",
     {"tool": "none", "arguments": {}, "need_clarification": True, "question": "请提供搜索关键词。"},
     {"tool": "search_docs", "arguments": {"query": "东西"}}),
    ("查文档",
     {"tool": "none", "arguments": {}, "need_clarification": True, "question": "请提供搜索关键词。"},
     {"tool": "search_docs", "arguments": {"query": "文档"}}),
    # calculator 缺 expression
    ("帮我算一下",
     {"tool": "none", "arguments": {}, "need_clarification": True, "question": "请提供需要计算的数学表达式。"},
     {"tool": "calculator", "arguments": {"expression": "1+1"}}),
    ("算个数",
     {"tool": "none", "arguments": {}, "need_clarification": True, "question": "请提供需要计算的数学表达式。"},
     {"tool": "calculator", "arguments": {"expression": "0"}}),
    ("做一道计算题",
     {"tool": "none", "arguments": {}, "need_clarification": True, "question": "请提供需要计算的数学表达式。"},
     {"tool": "calculator", "arguments": {"expression": "2+2"}}),
    # query_order 缺 order_id
    ("帮我查一下订单",
     {"tool": "none", "arguments": {}, "need_clarification": True, "question": "请提供订单号。"},
     {"tool": "query_order", "arguments": {"order_id": "unknown"}}),
    ("查订单状态",
     {"tool": "none", "arguments": {}, "need_clarification": True, "question": "请提供订单号。"},
     {"tool": "query_order", "arguments": {"order_id": "N/A"}}),
    ("我的订单到哪了",
     {"tool": "none", "arguments": {}, "need_clarification": True, "question": "请提供订单号。"},
     {"tool": "query_order", "arguments": {"order_id": "last_order"}}),
    ("物流跟踪",
     {"tool": "none", "arguments": {}, "need_clarification": True, "question": "请提供订单号。"},
     {"tool": "query_order", "arguments": {"order_id": "tracking"}}),
    # book_meeting 缺参数
    ("帮我预约个会议",
     {"tool": "none", "arguments": {}, "need_clarification": True, "question": "请提供会议日期、时间和主题。"},
     {"tool": "book_meeting", "arguments": {"date": "2026-07-01", "time": "10:00", "topic": "会议"}}),
    ("安排一个会议",
     {"tool": "none", "arguments": {}, "need_clarification": True, "question": "请提供会议日期、时间和主题。"},
     {"tool": "book_meeting", "arguments": {"date": "2026-07-01", "time": "14:00", "topic": "讨论"}}),
    ("帮我定个会",
     {"tool": "none", "arguments": {}, "need_clarification": True, "question": "请提供会议日期、时间和主题。"},
     {"tool": "book_meeting", "arguments": {"date": "2026-07-01", "time": "09:00", "topic": "例会"}}),
    ("约个会",
     {"tool": "none", "arguments": {}, "need_clarification": True, "question": "请提供会议日期、时间和主题。"},
     {"tool": "book_meeting", "arguments": {"date": "2026-06-20", "time": "15:00", "topic": "约会"}}),
    ("明天上午开会",
     {"tool": "none", "arguments": {}, "need_clarification": True, "question": "请提供会议主题和具体时间。"},
     {"tool": "book_meeting", "arguments": {"date": "2026-06-19", "time": "09:00", "topic": "会议"}}),
    # send_email 缺参数
    ("帮我发封邮件",
     {"tool": "none", "arguments": {}, "need_clarification": True, "question": "请提供收件人邮箱、邮件主题和内容。"},
     {"tool": "send_email", "arguments": {"recipient": "unknown@example.com", "subject": "通知", "content": "请查看。"}}),
    ("发邮件",
     {"tool": "none", "arguments": {}, "need_clarification": True, "question": "请提供收件人邮箱、邮件主题和内容。"},
     {"tool": "send_email", "arguments": {"recipient": "admin@company.com", "subject": "邮件", "content": "内容"}}),
    ("帮我写个邮件给HR",
     {"tool": "none", "arguments": {}, "need_clarification": True, "question": "请提供收件人的完整邮箱地址、邮件主题和内容。"},
     {"tool": "send_email", "arguments": {"recipient": "hr@corp.cn", "subject": "申请", "content": "请假申请"}}),
    ("发周报给领导",
     {"tool": "none", "arguments": {}, "need_clarification": True, "question": "请提供收件人邮箱和邮件内容。"},
     {"tool": "send_email", "arguments": {"recipient": "leader@corp.cn", "subject": "周报", "content": "本周工作内容"}}),
]

# 批量生成更多追问对
_extra_must_clarify = [
    ("帮我搜", "请提供搜索关键词。", "search_docs", {"query": "搜索"}),
    ("查查资料", "请提供搜索关键词。", "search_docs", {"query": "资料"}),
    ("算算", "请提供需要计算的数学表达式。", "calculator", {"expression": "1"}),
    ("求解", "请提供需要计算的数学表达式。", "calculator", {"expression": "x"}),
    ("查物流", "请提供订单号。", "query_order", {"order_id": "latest"}),
    ("包裹查一下", "请提供订单号。", "query_order", {"order_id": "package"}),
    ("帮我安排个会", "请提供会议日期、时间和主题。", "book_meeting", {"date": "2026-07-01", "time": "10:00", "topic": "会议"}),
    ("约个讨论", "请提供会议日期、时间和主题。", "book_meeting", {"date": "2026-06-20", "time": "14:00", "topic": "讨论"}),
    ("发个通知", "请提供收件人邮箱、通知主题和内容。", "send_email", {"recipient": "all@corp.cn", "subject": "通知", "content": "通知内容"}),
    ("邮件通知一下", "请提供收件人、主题和内容。", "send_email", {"recipient": "team@corp.cn", "subject": "通知", "content": "请查看。"}),
    ("帮我查个东西", "请问您要查什么？订单还是资料？", "search_docs", {"query": "东西"}),
    ("我需要帮助", "请问您具体需要什么帮助？", "search_docs", {"query": "帮助"}),
    ("帮忙处理一下", "请问您具体需要什么帮助？", "query_order", {"order_id": "help"}),
    ("约个时间讨论", "请提供会议日期、时间和主题。", "book_meeting", {"date": "2026-07-01", "time": "10:00", "topic": "讨论"}),
]
for ins, question, wrong_tool, wrong_args in _extra_must_clarify:
    MUST_CLARIFY_PAIRS.append(
        (ins,
         {"tool": "none", "arguments": {}, "need_clarification": True, "question": question},
         {"tool": wrong_tool, "arguments": wrong_args}))

# ═══════════════════════════════════════════════════════════════
# 类型3: 保持 JSON 格式 (~60对)
# chosen = 合法JSON, rejected = 非法JSON/自然语言
# ═══════════════════════════════════════════════════════════════

JSON_FORMAT_PAIRS = []

_tool_queries = [
    ("查订单 A10293",
     {"tool": "query_order", "arguments": {"order_id": "A10293"}}),
    ("计算 123 * 456",
     {"tool": "calculator", "arguments": {"expression": "123 * 456"}}),
    ("搜索 Docker 教程",
     {"tool": "search_docs", "arguments": {"query": "Docker 教程"}}),
    ("预约 2026-08-01 10:00 代码审查",
     {"tool": "book_meeting", "arguments": {"date": "2026-08-01", "time": "10:00", "topic": "代码审查"}}),
    ("发邮件给 admin@company.com 关于系统升级",
     {"tool": "send_email", "arguments": {"recipient": "admin@company.com", "subject": "系统升级", "content": "请查收系统升级通知。"}}),
    # 更多变体
    ("查下单 B20456",
     {"tool": "query_order", "arguments": {"order_id": "B20456"}}),
    ("算一下 7*8*9",
     {"tool": "calculator", "arguments": {"expression": "7 * 8 * 9"}}),
    ("检索 Nginx 配置",
     {"tool": "search_docs", "arguments": {"query": "Nginx 配置"}}),
    ("预定 2026-09-15 14:30 的需求评审",
     {"tool": "book_meeting", "arguments": {"date": "2026-09-15", "time": "14:30", "topic": "需求评审"}}),
    ("给 hr@corp.cn 发请假邮件",
     {"tool": "send_email", "arguments": {"recipient": "hr@corp.cn", "subject": "请假申请", "content": "您好，我因身体不适需要请假一天。"}}),
]

for query, correct in _tool_queries:
    correct_str = json.dumps(correct, ensure_ascii=False)
    # rejected: 自然语言
    JSON_FORMAT_PAIRS.append(
        (query, correct_str, f"好的，我来帮你处理。{correct_str}"))
    # rejected: 缺引号的 JSON
    bad_json = correct_str.replace('"tool"', 'tool').replace('"arguments"', 'arguments')
    JSON_FORMAT_PAIRS.append(
        (query, correct_str, bad_json))
    # rejected: 纯自然语言
    JSON_FORMAT_PAIRS.append(
        (query, correct_str, "好的，我来帮你处理。"))

# ═══════════════════════════════════════════════════════════════
# 类型4: 语义干扰拒调用 (~80对)
# chosen = none (不调工具), rejected = 错误调用工具
# ═══════════════════════════════════════════════════════════════

SEMANTIC_DPO_PAIRS = [
    # 提到会议但不需
    ("我今天有点累，不想开会",
     {"tool": "none", "arguments": {}},
     {"tool": "book_meeting", "arguments": {"date": "2026-06-19", "time": "09:00", "topic": "会议"}}),
    ("昨天那个会议开了三个小时",
     {"tool": "none", "arguments": {}},
     {"tool": "book_meeting", "arguments": {"date": "2026-06-17", "time": "09:00", "topic": "会议记录"}}),
    ("这个会议取消吧",
     {"tool": "none", "arguments": {}},
     {"tool": "book_meeting", "arguments": {"date": "2026-06-19", "time": "10:00", "topic": "取消会议"}}),
    ("上次开会讨论的方案你还记得吗",
     {"tool": "none", "arguments": {}},
     {"tool": "search_docs", "arguments": {"query": "开会讨论的方案"}}),
    ("下周再约时间",
     {"tool": "none", "arguments": {}},
     {"tool": "book_meeting", "arguments": {"date": "2026-06-26", "time": "10:00", "topic": "约时间"}}),
    # 提到订单但不需
    ("这个订单号看起来像 A10293，但我不是要查询它",
     {"tool": "none", "arguments": {}},
     {"tool": "query_order", "arguments": {"order_id": "A10293"}}),
    ("昨天我下了一个订单，体验还不错",
     {"tool": "none", "arguments": {}},
     {"tool": "query_order", "arguments": {"order_id": "recent"}}),
    ("订单编号的格式一般是怎样的",
     {"tool": "none", "arguments": {}},
     {"tool": "query_order", "arguments": {"order_id": "format"}}),
    ("请问退货流程是什么",
     {"tool": "none", "arguments": {}},
     {"tool": "query_order", "arguments": {"order_id": "return"}}),
    ("这个订单能不能取消",
     {"tool": "none", "arguments": {}},
     {"tool": "query_order", "arguments": {"order_id": "cancel"}}),
    # 提到邮件但不需
    ("邮件收到了吗",
     {"tool": "none", "arguments": {}},
     {"tool": "send_email", "arguments": {"recipient": "unknown@example.com", "subject": "邮件确认", "content": "确认收到邮件。"}}),
    ("我昨天发了一封邮件给经理但没回复",
     {"tool": "none", "arguments": {}},
     {"tool": "send_email", "arguments": {"recipient": "manager@corp.cn", "subject": "跟进", "content": "跟进邮件。"}}),
    ("邮件通知功能好用吗",
     {"tool": "none", "arguments": {}},
     {"tool": "send_email", "arguments": {"recipient": "admin@company.com", "subject": "邮件功能", "content": "询问邮件功能。"}}),
    ("请查收附件中的报告",
     {"tool": "none", "arguments": {}},
     {"tool": "send_email", "arguments": {"recipient": "team@corp.cn", "subject": "报告", "content": "附件报告。"}}),
    ("回复邮件时要注意什么",
     {"tool": "none", "arguments": {}},
     {"tool": "send_email", "arguments": {"recipient": "hr@corp.cn", "subject": "邮件注意", "content": "邮件注意事项。"}}),
    # 提到计算但不需
    ("计算器这个工具怎么用",
     {"tool": "none", "arguments": {}},
     {"tool": "calculator", "arguments": {"expression": "calculator usage"}}),
    ("这个计算结果靠谱吗",
     {"tool": "none", "arguments": {}},
     {"tool": "calculator", "arguments": {"expression": "verify"}}),
    ("我不太会算这个账",
     {"tool": "none", "arguments": {}},
     {"tool": "calculator", "arguments": {"expression": "account"}}),
    # 提到搜索但不需
    ("搜索功能好像不太准",
     {"tool": "none", "arguments": {}},
     {"tool": "search_docs", "arguments": {"query": "搜索功能"}}),
    ("上次搜索的内容我没保存",
     {"tool": "none", "arguments": {}},
     {"tool": "search_docs", "arguments": {"query": "上次搜索的内容"}}),
    ("有没有更快的方法找资料",
     {"tool": "none", "arguments": {}},
     {"tool": "search_docs", "arguments": {"query": "快速找资料的方法"}}),
    # 日期询问
    ("今天是几号",
     {"tool": "none", "arguments": {}},
     {"tool": "query_order", "arguments": {"order_id": "今天的日期"}}),
    ("现在几点了",
     {"tool": "none", "arguments": {}},
     {"tool": "query_order", "arguments": {"order_id": "current_time"}}),
    ("明天星期几",
     {"tool": "none", "arguments": {}},
     {"tool": "book_meeting", "arguments": {"date": "2026-06-19", "time": "00:00", "topic": "星期查询"}}),
    ("这个月还有多少天",
     {"tool": "none", "arguments": {}},
     {"tool": "calculator", "arguments": {"expression": "days_in_month"}}),
]

# 批量生成更多反追问对（目标 ~140）
_more_anti2 = [
    "今天过得怎么样", "有什么新鲜事", "今天心情不错", "最近好吗",
    "好久没聊天了", "你有什么爱好", "你喜欢什么颜色",
    "你喜欢音乐吗", "你最喜欢哪部电影", "你最擅长什么",
    "有什么想聊的吗", "我们来聊天吧", "你能陪我聊聊吗",
    "我心情不好", "我有点难过", "我无聊了", "给我讲点有趣的",
    "今天穿了什么颜色衣服", "你喜欢猫还是狗", "你喜欢运动吗",
    "咖啡和茶哪个好", "早起好还是晚睡好", "你觉得读书有用吗",
    "学什么技术最赚钱", "创业还是打工好", "如何看待加班文化",
    "远程办公效率高吗", "自学的效果好吗", "要不要读研",
    "跳槽好还是内部晋升好", "工作几年再创业合适吗",
    "大城市好还是小城市好", "买房还是租房", "结婚要不要买房",
    "星座靠谱吗", "运气可以改变吗", "努力和天赋哪个重要",
    "人活着是为了什么", "快乐是什么", "成功怎么定义",
    "做事拖延怎么办", "怎么克服懒惰", "如何找到人生目标",
]
for q in _more_anti2:
    ANTI_OVERCLARIFY_PAIRS.append(
        (q, {"tool": "none", "arguments": {}},
         {"tool": "none", "arguments": {}, "need_clarification": True,
          "question": "请提供更多细节以便我更好地帮助你。"}))

# 第三批反追问对
_anti3 = [
    "给我点正能量", "怎么提高专注力", "快给我打气",
    "好无聊", "有点沮丧", "今天运气不错", "分享一个笑话",
    "人生哲理是什么", "怎么变自律", "拖延症怎么破",
    "如何高效学习", "有什么好书", "推荐一个纪录片",
    "怎么练口语", "如何快速入睡", "早上起不来怎么办",
    "怎么让自己更有动力", "学习没动力了", "刷题有用吗",
]
for q in _anti3:
    ANTI_OVERCLARIFY_PAIRS.append(
        (q, {"tool": "none", "arguments": {}},
         {"tool": "none", "arguments": {}, "need_clarification": True,
          "question": "请提供更多上下文信息。"}))

# 补充更多语义干扰
_sem3 = [
    ("下班时间到了", "book_meeting", {"date": "2026-06-19", "time": "18:00", "topic": "下班"}),
    ("午饭时间到了", "book_meeting", {"date": "2026-06-19", "time": "12:00", "topic": "午饭"}),
    ("月底了要写月报", "send_email", {"recipient": "manager@corp.cn", "subject": "月报", "content": "月报内容"}),
    ("年终总结怎么写", "send_email", {"recipient": "hr@corp.cn", "subject": "年终总结", "content": "总结内容"}),
    ("数据怎么分析", "calculator", {"expression": "数据分析"}),
    ("统计一下数量", "calculator", {"expression": "统计数量"}),
    ("搜索引擎的工作原理", "search_docs", {"query": "搜索引擎原理"}),
    ("怎么快速找到文件", "search_docs", {"query": "快速找文件"}),
    ("圣诞节是哪天", "book_meeting", {"date": "2026-12-25", "time": "00:00", "topic": "圣诞节"}),
    ("国庆节放几天", "book_meeting", {"date": "2026-10-01", "time": "00:00", "topic": "国庆节"}),
]
for ins, wt, wa in _sem3:
    SEMANTIC_DPO_PAIRS.append((ins, {"tool": "none", "arguments": {}}, {"tool": wt, "arguments": wa}))

# 批量补充更多追问对（目标 ~100）
_more_clarify2 = [
    ("搜一下那个", "search_docs", {"query": "那个"}),
    ("检索一下", "search_docs", {"query": "检索"}),
    ("帮我找个东西", "search_docs", {"query": "东西"}),
    ("算算看", "calculator", {"expression": "0"}),
    ("求个值", "calculator", {"expression": "x"}),
    ("帮我做计算", "calculator", {"expression": "calc"}),
    ("订单查询", "query_order", {"order_id": "query"}),
    ("查查快递", "query_order", {"order_id": "express"}),
    ("看下物流", "query_order", {"order_id": "logistics"}),
    ("安排会议", "book_meeting", {"date": "2026-07-01", "time": "10:00", "topic": "会议"}),
    ("约个需求讨论", "book_meeting", {"date": "2026-06-20", "time": "14:00", "topic": "需求讨论"}),
    ("帮我定会议室", "book_meeting", {"date": "2026-07-01", "time": "09:00", "topic": "定会议室"}),
    ("帮我预约一下", "book_meeting", {"date": "2026-07-01", "time": "10:00", "topic": "预约"}),
    ("明天开会讨论", "book_meeting", {"date": "2026-06-19", "time": "10:00", "topic": "讨论"}),
    ("发邮件通知", "send_email", {"recipient": "all@corp.cn", "subject": "通知", "content": "内容"}),
    ("帮我起草邮件", "send_email", {"recipient": "draft@corp.cn", "subject": "邮件", "content": "起草"}),
    ("给客户发邮件", "send_email", {"recipient": "client@corp.cn", "subject": "客户邮件", "content": "邮件内容"}),
    ("写个邮件给老板", "send_email", {"recipient": "boss@corp.cn", "subject": "邮件", "content": "内容"}),
    ("邮件发送一下", "send_email", {"recipient": "send@corp.cn", "subject": "发送", "content": "发送内容"}),
    ("帮我通知团队", "send_email", {"recipient": "team@corp.cn", "subject": "通知", "content": "通知内容"}),
]

for ins, wt, wa in _more_clarify2:
    qmap = {"search_docs": "请提供搜索关键词。", "calculator": "请提供需要计算的数学表达式。",
            "query_order": "请提供订单号。",
            "book_meeting": "请提供会议日期、时间和主题。",
            "send_email": "请提供收件人邮箱、邮件主题和内容。"}
    MUST_CLARIFY_PAIRS.append(
        (ins,
         {"tool": "none", "arguments": {}, "need_clarification": True, "question": qmap[wt]},
         {"tool": wt, "arguments": wa}))

# 批量补充更多 JSON 格式对（目标 ~60）
_more_json_tools = [
    ("查订单 ORD-2024-0892", {"tool": "query_order", "arguments": {"order_id": "ORD-2024-0892"}}),
    ("计算 sqrt(144)", {"tool": "calculator", "arguments": {"expression": "sqrt(144)"}}),
    ("搜 PyTorch 基础教程", {"tool": "search_docs", "arguments": {"query": "PyTorch 基础教程"}}),
    ("预约 2026-10-01 的年度总结", {"tool": "book_meeting", "arguments": {"date": "2026-10-01", "time": "14:00", "topic": "年度总结"}}),
    ("给 support@service.com 发服务通知", {"tool": "send_email", "arguments": {"recipient": "support@service.com", "subject": "服务通知", "content": "请查看服务通知内容。"}}),
    ("查 SO-240315-001", {"tool": "query_order", "arguments": {"order_id": "SO-240315-001"}}),
    ("算 17 * 19 + 23", {"tool": "calculator", "arguments": {"expression": "17 * 19 + 23"}}),
    ("搜 Linux 常用命令", {"tool": "search_docs", "arguments": {"query": "Linux 常用命令"}}),
    ("预约 2026-11-15 的性能讨论", {"tool": "book_meeting", "arguments": {"date": "2026-11-15", "time": "11:00", "topic": "性能讨论"}}),
    ("给 wangwu@dev.io 发技术方案", {"tool": "send_email", "arguments": {"recipient": "wangwu@dev.io", "subject": "技术方案", "content": "详见技术方案内容。"}}),
]
for q, c in _more_json_tools:
    cs = json.dumps(c, ensure_ascii=False)
    JSON_FORMAT_PAIRS.append((q, cs, f"好的，我来帮你处理。{cs}"))
    bad = cs.replace('"tool"', 'tool').replace('"arguments"', 'arguments')
    JSON_FORMAT_PAIRS.append((q, cs, bad))

# 批量补充更多语义干扰对（目标 ~80）
_more_semantic2 = [
    ("今天的会议改到什么时候了", "book_meeting", {"date": "2026-06-19", "time": "00:00", "topic": "会议时间"}),
    ("下班后要不要一起去吃饭", "book_meeting", {"date": "2026-06-19", "time": "18:00", "topic": "吃饭"}),
    ("周末加班吗", "book_meeting", {"date": "2026-06-21", "time": "09:00", "topic": "加班"}),
    ("上次订的那个东西到哪了", "query_order", {"order_id": "上次的"}),
    ("我买的东西质量有问题怎么投诉", "query_order", {"order_id": "complaint"}),
    ("为什么我的订单总是延迟发货", "query_order", {"order_id": "delayed"}),
    ("怎么把邮箱里的垃圾邮件过滤掉", "send_email", {"recipient": "spam@filter.com", "subject": "过滤", "content": "filter"}),
    ("邮件太多看不过来", "send_email", {"recipient": "inbox@corp.cn", "subject": "整理", "content": "整理邮件"}),
    ("心算怎么练", "calculator", {"expression": "心算"}),
    ("数学建模怎么学", "calculator", {"expression": "数学建模"}),
    ("为什么我搜的关键词没有结果", "search_docs", {"query": "关键词 没有结果"}),
    ("有什么好的搜索技巧", "search_docs", {"query": "搜索技巧"}),
    ("今天是端午节吗", "query_order", {"order_id": "端午节"}),
    ("春节是哪天", "book_meeting", {"date": "2026-01-29", "time": "00:00", "topic": "春节"}),
    ("2024是不是闰年", "calculator", {"expression": "2024 % 4"})]
for ins, wt, wa in _more_semantic2:
    SEMANTIC_DPO_PAIRS.append((ins, {"tool": "none", "arguments": {}}, {"tool": wt, "arguments": wa}))

# 批量补充更多语义干扰对（目标 ~80）
_more_semantic = [
    ("今天周五了", "book_meeting", {"date": "2026-06-20", "time": "00:00", "topic": "周五"}),
    ("我不想开会", "book_meeting", {"date": "2026-06-19", "time": "09:00", "topic": "会议"}),
    ("订单被拒了怎么办", "query_order", {"order_id": "rejected"}),
    ("邮件发不出去", "send_email", {"recipient": "admin@company.com", "subject": "test", "content": "test"}),
    ("这个算错了", "calculator", {"expression": "error"}),
    ("搜不到东西", "search_docs", {"query": "搜不到"}),
    ("为什么我的订单还没发货", "query_order", {"order_id": "pending"}),
    ("这封邮件是垃圾邮件吗", "send_email", {"recipient": "spam@check.com", "subject": "spam?", "content": "check"}),
    ("数学好难", "calculator", {"expression": "difficulty"}),
    ("搜索引擎怎么优化", "search_docs", {"query": "搜索引擎优化"}),
]
for ins, wrong_tool, wrong_args in _more_semantic:
    SEMANTIC_DPO_PAIRS.append(
        (ins, {"tool": "none", "arguments": {}}, {"tool": wrong_tool, "arguments": wrong_args}))

# ═══════════════════════════════════════════════════════════════
# 类型5: 工具选择正确 (~40对)
# chosen = 正确工具+参数, rejected = 错误工具
# ═══════════════════════════════════════════════════════════════

TOOL_SELECTION_PAIRS = [
    ("查订单 A10293",
     {"tool": "query_order", "arguments": {"order_id": "A10293"}},
     {"tool": "search_docs", "arguments": {"query": "订单 A10293"}}),
    ("计算 123*456",
     {"tool": "calculator", "arguments": {"expression": "123 * 456"}},
     {"tool": "search_docs", "arguments": {"query": "123*456"}}),
    ("搜索 Python 教程",
     {"tool": "search_docs", "arguments": {"query": "Python 教程"}},
     {"tool": "query_order", "arguments": {"order_id": "Python 教程"}}),
    ("帮我预约明天的项目评审",
     {"tool": "book_meeting", "arguments": {"date": "2026-06-19", "time": "14:00", "topic": "项目评审"}},
     {"tool": "send_email", "arguments": {"recipient": "project@corp.cn", "subject": "项目评审", "content": "预约会议。"}}),
    ("给 ceo@startup.com 发年度汇报",
     {"tool": "send_email", "arguments": {"recipient": "ceo@startup.com", "subject": "年度汇报", "content": "详见附件。"}},
     {"tool": "book_meeting", "arguments": {"date": "2026-07-01", "time": "10:00", "topic": "年度汇报"}}),
    # more
    ("搜一下 Docker 怎么部署",
     {"tool": "search_docs", "arguments": {"query": "Docker 部署"}},
     {"tool": "calculator", "arguments": {"expression": "Docker 部署"}}),
    ("查查物流 B20456",
     {"tool": "query_order", "arguments": {"order_id": "B20456"}},
     {"tool": "send_email", "arguments": {"recipient": "logistics@corp.cn", "subject": "物流 B20456", "content": "查询物流"}}),
    ("算下 1024/8",
     {"tool": "calculator", "arguments": {"expression": "1024 / 8"}},
     {"tool": "query_order", "arguments": {"order_id": "1024/8"}}),
]

# 补充更多工具选择对
_extra_tool_pairs = [
    ("帮我预约 Sprint 回顾", "book_meeting",
     {"date": "2026-06-26", "time": "10:00", "topic": "Sprint 回顾"},
     "search_docs", {"query": "Sprint 回顾"}),
    ("发邮件给 team@corp.cn 通知会议变更", "send_email",
     {"recipient": "team@corp.cn", "subject": "会议变更通知", "content": "详见内容。"},
     "book_meeting", {"date": "2026-06-20", "time": "14:00", "topic": "会议变更"}),
    ("计算一下 3.14 * 100", "calculator",
     {"expression": "3.14 * 100"},
     "query_order", {"order_id": "3.14"}),
    ("查一下订单 D45678", "query_order",
     {"order_id": "D45678"},
     "calculator", {"expression": "D45678"}),
    ("搜一搜 大模型微调方法", "search_docs",
     {"query": "大模型微调方法"},
     "book_meeting", {"date": "2026-07-01", "time": "10:00", "topic": "大模型微调"})]
for ins, correct_tool, correct_args, wrong_tool, wrong_args in _extra_tool_pairs:
    TOOL_SELECTION_PAIRS.append(
        (ins, {"tool": correct_tool, "arguments": correct_args},
         {"tool": wrong_tool, "arguments": wrong_args}))

# 补充更多工具选择对（目标 ~40）
_more_tool_pairs = [
    ("查订单 A10293", "query_order", {"order_id": "A10293"}, "book_meeting", {"date": "2026-06-19", "time": "10:00", "topic": "订单 A10293"}),
    ("算 256+512", "calculator", {"expression": "256 + 512"}, "search_docs", {"query": "256+512"}),
    ("搜 Docker 教程", "search_docs", {"query": "Docker 教程"}, "send_email", {"recipient": "docker@corp.cn", "subject": "Docker 教程", "content": "搜索Docker"}),
    ("预约明天上午的需求评审", "book_meeting", {"date": "2026-06-19", "time": "10:00", "topic": "需求评审"}, "query_order", {"order_id": "需求评审"}),
    ("发邮件给 manager@team.org 汇报进度", "send_email", {"recipient": "manager@team.org", "subject": "进度汇报", "content": "汇报内容"}, "calculator", {"expression": "进度汇报"}),
    ("查快递 B20456", "query_order", {"order_id": "B20456"}, "send_email", {"recipient": "express@corp.cn", "subject": "快递 B20456", "content": "查快递"}),
    ("算 1000-345", "calculator", {"expression": "1000 - 345"}, "book_meeting", {"date": "2026-06-19", "time": "10:00", "topic": "1000-345"}),
    ("搜 Nginx 反向代理", "search_docs", {"query": "Nginx 反向代理"}, "calculator", {"expression": "Nginx 反向代理"}),
    ("约下周三的代码审查", "book_meeting", {"date": "2026-06-25", "time": "14:00", "topic": "代码审查"}, "search_docs", {"query": "代码审查"}),
    ("给 hr@corp.cn 发请假邮件", "send_email", {"recipient": "hr@corp.cn", "subject": "请假申请", "content": "请假内容"}, "query_order", {"order_id": "请假"}),
]
for ins, ct, ca, wt, wa in _more_tool_pairs:
    TOOL_SELECTION_PAIRS.append(
        (ins, {"tool": ct, "arguments": ca}, {"tool": wt, "arguments": wa}))


# ═══════════════════════════════════════════════════════════════
# 组装
# ═══════════════════════════════════════════════════════════════

def build_pairs():
    all_pairs = []

    def add(triples, etype):
        for prompt, chosen, rejected in triples:
            chosen_str = json.dumps(chosen, ensure_ascii=False) if isinstance(chosen, dict) else chosen
            rejected_str = json.dumps(rejected, ensure_ascii=False) if isinstance(rejected, dict) else rejected
            if chosen_str.strip() == rejected_str.strip():
                continue
            all_pairs.append({
                "prompt": prompt,
                "chosen": chosen_str,
                "rejected": rejected_str,
                "error_type": etype,
            })

    add(ANTI_OVERCLARIFY_PAIRS, "anti_overclarify")
    add(MUST_CLARIFY_PAIRS, "must_clarify")
    add(JSON_FORMAT_PAIRS, "json_format")
    add(SEMANTIC_DPO_PAIRS, "semantic_interference")
    add(TOOL_SELECTION_PAIRS, "tool_selection")

    random.shuffle(all_pairs)
    return all_pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=os.path.join(PROJECT_ROOT, "data", "dpo", "train_v2.json"))
    args = parser.parse_args()

    pairs = build_pairs()
    print(f"DPO-v2 偏好对总数: {len(pairs)}")

    # 统计
    from collections import Counter
    ec = Counter(p["error_type"] for p in pairs)
    ERROR_LABELS = {
        "anti_overclarify": "抑制过度追问",
        "must_clarify": "参数缺失必须追问",
        "json_format": "保持JSON格式",
        "semantic_interference": "语义干扰拒调用",
        "tool_selection": "工具选择正确",
    }
    for et, count in ec.most_common():
        print(f"  {ERROR_LABELS.get(et, et)}: {count}")

    # 验证
    for i, p in enumerate(pairs):
        for f in ["prompt", "chosen", "rejected"]:
            if f not in p:
                print(f"❌ #{i}: 缺少 {f}")
        if p["chosen"] == p["rejected"]:
            print(f"❌ #{i}: chosen == rejected")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(pairs, f, ensure_ascii=False, indent=2)
    print(f"\n✅ 保存: {args.output}")

    # 样例
    print("\n样例:")
    for i, p in enumerate(pairs[:3]):
        print(f"\n[{i+1}] [{ERROR_LABELS.get(p['error_type'], p['error_type'])}]")
        print(f"  Prompt: {p['prompt'][:50]}")
        print(f"  Chosen: {p['chosen'][:60]}")
        print(f"  Rejected: {p['rejected'][:60]}")


if __name__ == "__main__":
    main()
