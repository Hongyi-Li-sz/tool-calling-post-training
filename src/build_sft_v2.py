"""
构造 SFT-v2 修复型训练数据

Day 9 评测发现 → Day 10 精准修复：
  问题1: 过度追问率 84-96%  → 150条 "无需工具+不追问" 样本
  问题2: 语义干扰 0%        → 120条 语义干扰拒调用样本
  问题3: 参数追问 12.5%     → 200条 参数缺失追问样本
  保持:  单工具调用能力     → 100条 标准调用样本
  补充:  边界情况           →  30条 复合/模糊请求

总计 ≥600 条，保存到 data/sft/train_v2.json
"""
import argparse
import json
import os
import random
import sys
from datetime import datetime, timedelta

random.seed(42)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ═══════════════════════════════════════════════════════════════
# 工具参数池（复用 V1 的丰富参数池）
# ═══════════════════════════════════════════════════════════════

QUERIES = ["Python 多线程教程", "深度学习入门", "Transformer 架构详解",
           "Docker 部署", "Kubernetes 入门", "MySQL 索引优化",
           "Redis 缓存策略", "Git 分支管理", "Linux 常用命令",
           "RESTful API 设计", "Nginx 反向代理", "分布式系统一致性",
           "PyTorch 基础", "自然语言处理综述", "时间序列预测方法",
           "CUDA 编程指南", "设计模式入门", "推荐系统算法"]

EXPRESSIONS = ["123 * 456", "15 + 28 / 2", "(100 - 25) * 3", "1024 / 8",
               "3.14 * 10 * 10", "256 + 512", "1000 - 345", "17 * 19",
               "2 ** 10", "88 / 4 + 6", "100 * 0.85", "7 * 8 * 9",
               "9999 / 3", "45 + 55 - 20", "sqrt(144)", "1024 % 128"]

ORDER_IDS = ["A10293", "B20456", "C37890", "D45678", "E56789",
             "F12345", "G98765", "H34567", "ORD-2024-001", "ORD-2024-0892",
             "SO-240315-001", "PO-20240315-0056", "TBA1029384756"]

MEETING_DATES = [(datetime.now() + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(1, 31)]
MEETING_TIMES = ["09:00", "09:30", "10:00", "10:30", "11:00",
                 "14:00", "14:30", "15:00", "15:30", "16:00"]
MEETING_TOPICS = ["项目评审", "Sprint 回顾", "技术方案讨论", "需求评审",
                  "季度规划", "1v1 沟通", "代码审查", "架构设计讨论",
                  "预算评审", "客户演示", "年度总结", "性能优化讨论"]

EMAIL_RECIPIENTS = ["admin@company.com", "zhangsan@example.com", "hr@corp.cn",
                    "manager@team.org", "support@service.com", "ceo@startup.com"]
EMAIL_SUBJECTS = ["系统升级通知", "项目进度汇报", "请假申请", "预算审批",
                  "会议纪要", "周报", "活动通知", "客户反馈汇总", "安全提醒"]
EMAIL_CONTENTS = ["请查收附件。", "详见正文内容。", "如有问题请及时反馈。",
                  "请于周五前回复。", "以上内容供参考。"]

# ═══════════════════════════════════════════════════════════════
# 类型1: 参数缺失追问 (200条) — 修复12.5%的追问准确率
# ═══════════════════════════════════════════════════════════════

CLARIFICATION_SAMPLES = []

# search_docs 缺 query (30条)
_search_tmpl = [
    "帮我搜索一下", "检索知识库", "搜点东西", "查文档", "帮我搜",
    "帮我在知识库里找找", "搜索功能用一下", "找一下资料",
    "搜一下那个", "帮我查查资料", "有没有相关文档", "文档搜索",
    "帮我检索内容", "找点东西", "搜搜看",
]
for tmpl in _search_tmpl * 2:  # 每个模板2次，变参数
    CLARIFICATION_SAMPLES.append({
        "instruction": tmpl,
        "output": json.dumps({"tool": "none", "arguments": {},
                              "need_clarification": True,
                              "question": "请提供搜索关键词或问题。"},
                             ensure_ascii=False)})

# calculator 缺 expression (30条)
_calc_tmpl = [
    "帮我算一下", "算个数", "计算器", "做一道计算题",
    "那个表达式帮我算算", "帮我计算", "求解", "算算",
    "帮我算个东西", "计算一下", "用计算器", "帮我做数学题",
    "帮我求值", "计算题", "数学计算",
]
for tmpl in _calc_tmpl * 2:
    CLARIFICATION_SAMPLES.append({
        "instruction": tmpl,
        "output": json.dumps({"tool": "none", "arguments": {},
                              "need_clarification": True,
                              "question": "请提供需要计算的数学表达式。"},
                             ensure_ascii=False)})

# query_order 缺 order_id (30条)
_order_tmpl = [
    "帮我查一下订单", "查订单状态", "我的订单到哪了", "物流跟踪",
    "订单怎么样了", "看看我的快递", "追踪包裹", "查物流",
    "快递到哪了", "帮我查订单", "看下我的订单", "查查物流信息",
    "订单状态查询", "查件", "查包裹",
]
for tmpl in _order_tmpl * 2:
    CLARIFICATION_SAMPLES.append({
        "instruction": tmpl,
        "output": json.dumps({"tool": "none", "arguments": {},
                              "need_clarification": True,
                              "question": "请提供订单号。"},
                             ensure_ascii=False)})

# book_meeting 缺参数 (55条)
_meeting_tmpl = [
    ("帮我预约个会议", "请提供会议日期、时间和主题。"),
    ("我想预约会议", "请提供会议日期、时间和主题。"),
    ("安排一个会议", "请提供会议日期、时间和主题。"),
    ("帮我定个会", "请提供会议日期、时间和主题。"),
    ("约个会", "请提供会议日期、时间和主题。"),
    ("帮我约个需求评审", "请提供会议日期和时间。"),
    ("明天上午开会", "请提供会议主题和具体时间。"),
    ("下午3点帮我订会议室", "请提供会议日期和主题。"),
    ("下周三下午约个讨论", "请提供具体日期（YYYY-MM-DD格式）、时间和主题。"),
    ("帮我安排下周五的会议", "请提供具体日期（YYYY-MM-DD格式）、时间和会议主题。"),
    ("约个下周一的会", "请提供具体日期（YYYY-MM-DD格式）、时间和主题。"),
    # 缺两个参数的变体
    ("帮我预约会议讨论项目进度", "请提供会议日期和时间。"),
    ("预约一个产品评审", "请提供会议日期和时间。"),
    ("约个时间讨论技术方案", "请提供会议日期、时间。"),
    ("帮我预定会议室", "请提供会议日期、时间和主题。"),
    ("安排下周的Sprint回顾", "请提供具体日期和时间。"),
    ("帮我约架构讨论", "请提供会议日期和时间。"),
    ("定个明天下午的会", "请提供会议主题和具体时间（如14:00）。"),
    ("下周一上午十点约预算评审", "请提供具体日期（YYYY-MM-DD格式）。"),
    ("约个周五下午的会议", "请提供具体日期、时间和主题。"),
    ("帮我安排团队建设", "请提供会议日期和时间。"),
    # 更多变体
    ("安排个技术分享会", "请提供会议日期、时间和具体主题。"),
    ("帮我定个需求沟通会", "请提供会议日期和时间。"),
    ("约一个客户演示", "请提供会议日期和时间。"),
    ("帮我预约下周的产品发布讨论", "请提供具体日期和时间。"),
    ("预约一个1v1沟通", "请提供日期、时间和参与人信息。"),
    ("帮我安排季度规划会", "请提供会议日期和时间。"),
    ("定个年度总结的会", "请提供会议日期和时间。"),
]
for instruction, question in _meeting_tmpl:
    CLARIFICATION_SAMPLES.append({
        "instruction": instruction,
        "output": json.dumps({"tool": "none", "arguments": {},
                              "need_clarification": True,
                              "question": question},
                             ensure_ascii=False)})
# 补到55条
extra_meeting = [
    ("下周开个会", "请提供具体日期、时间、主题。"),
    ("帮我约个代码审查的会", "请提供会议日期和时间。"),
    ("安排会议讨论一下新功能", "请提供会议日期和时间。"),
    ("定一个明早的站会", "请提供具体日期和精确时间。"),
    ("帮我安排下周二的培训", "请提供具体日期、时间和培训主题。"),
    ("约个会讨论安全问题", "请提供会议日期和时间。"),
    ("帮我定个下周的会议室", "请提供日期、时间和主题。"),
]
for ins, q in extra_meeting:
    CLARIFICATION_SAMPLES.append({
        "instruction": ins,
        "output": json.dumps({"tool": "none", "arguments": {},
                              "need_clarification": True, "question": q},
                             ensure_ascii=False)})

# send_email 缺参数 (40条)
_email_tmpl = [
    ("帮我发封邮件", "请提供收件人邮箱、邮件主题和内容。"),
    ("发邮件", "请提供收件人邮箱、邮件主题和内容。"),
    ("帮我写个邮件给HR", "请提供收件人的完整邮箱地址、邮件主题和内容。"),
    ("发周报给领导", "请提供收件人邮箱和邮件内容。"),
    ("邮件通知全组", "请提供收件人邮箱、邮件主题和内容。"),
    ("帮我通知一下大家", "请提供收件人邮箱、通知主题和具体内容。"),
    ("发个通知邮件", "请提供收件人邮箱、邮件主题和内容。"),
    ("帮我给经理发个邮件", "请提供收件人的完整邮箱地址、邮件主题和内容。"),
    ("邮件通知一下", "请提供收件人、主题和内容。"),
    ("帮我起草一封邮件", "请提供收件人、主题和内容。"),
    ("给客户发个邮件", "请提供收件人的完整邮箱地址、主题和内容。"),
    ("帮我把会议纪要邮件出去", "请提供收件人邮箱。"),
    ("发一封请假邮件", "请提供收件人邮箱和具体请假内容。"),
    ("帮我邮件回复一下", "请提供收件人邮箱、回复主题和内容。"),
    ("给团队发个通知", "请提供收件人邮箱、通知主题和内容。"),
    # 更多场景
    ("邮件发给合作伙伴", "请提供收件人的完整邮箱地址、主题和内容。"),
    ("帮我写邮件通知项目延期", "请提供收件人邮箱和具体延期原因。"),
    ("发邮件提醒一下大家", "请提供收件人邮箱和提醒内容。"),
    ("给老板发个汇报邮件", "请提供收件人的完整邮箱地址、汇报主题和内容。"),
    ("帮我邮件发送周报", "请提供收件人邮箱和具体周报内容。"),
]
for instruction, question in _email_tmpl:
    CLARIFICATION_SAMPLES.append({
        "instruction": instruction,
        "output": json.dumps({"tool": "none", "arguments": {},
                              "need_clarification": True,
                              "question": question},
                             ensure_ascii=False)})
# 补到40条
for _ in range(6):
    CLARIFICATION_SAMPLES.append({
        "instruction": random.choice(["发个邮件通知", "帮我邮件联系", "邮件告知一下",
                                      "帮我发个通知邮件", "写邮件", "帮我群发邮件"]),
        "output": json.dumps({"tool": "none", "arguments": {},
                              "need_clarification": True,
                              "question": "请提供收件人邮箱、邮件主题和内容。"},
                             ensure_ascii=False)})

# 补充：更多追问变体 (34条)
_extra_clarify = [
    ("帮我搜搜", "请提供搜索关键词。"),
    ("算算这个", "请提供需要计算的数学表达式。"),
    ("查物流信息", "请提供订单号。"),
    ("帮我安排个会", "请提供会议日期、时间和主题。"),
    ("发个邮件", "请提供收件人邮箱、邮件主题和内容。"),
    ("帮我检索", "请提供搜索关键词或问题。"),
    ("做一道算术题", "请提供需要计算的数学表达式。"),
    ("快递查一下", "请提供订单号。"),
    ("约个会议", "请提供会议日期、时间和主题。"),
    ("邮件发一下", "请提供收件人邮箱、邮件主题和内容。"),
    ("搜个东西", "请提供搜索关键词。"),
    ("算一道题", "请提供需要计算的数学表达式。"),
    ("帮我追踪订单", "请提供订单号。"),
    ("定个会议", "请提供会议日期、时间和主题。"),
    ("写封邮件", "请提供收件人邮箱、邮件主题和内容。"),
    ("帮忙搜一搜", "请提供搜索关键词。"),
    ("帮我求个值", "请提供需要计算的数学表达式。"),
    ("看下订单信息", "请提供订单号。"),
    ("安排个交流会", "请提供会议日期、时间和主题。"),
    ("帮我邮件回复", "请提供收件人邮箱、回复主题和内容。"),
    ("帮我在文档里找", "请提供搜索关键词。"),
    ("帮忙计算一下", "请提供需要计算的数学表达式。"),
    ("我的快递呢", "请提供订单号。"),
    ("约个讨论会", "请提供会议日期、时间和主题。"),
    ("通知发送一下", "请提供收件人邮箱、通知主题和内容。"),
    ("查查资料", "请提供搜索关键词。"),
    ("求解数学题", "请提供需要计算的数学表达式。"),
    ("包裹查一下", "请提供订单号。"),
    ("安排个评审", "请提供会议日期和时间。"),
    ("给客户发通知", "请提供收件人邮箱和通知内容。"),
    ("搜一下文档库", "请提供搜索关键词。"),
    ("算个数给我", "请提供需要计算的数学表达式。"),
    ("帮我定个会议室讨论", "请提供会议日期、时间和主题。"),
    ("发一封关于请假的通知", "请提供收件人邮箱和请假内容。"),
]
for ins, q in _extra_clarify:
    CLARIFICATION_SAMPLES.append({
        "instruction": ins,
        "output": json.dumps({"tool": "none", "arguments": {},
                              "need_clarification": True, "question": q},
                             ensure_ascii=False)})

# 模糊意图追问 (15条)
_vague_tmpl = [
    "帮我查个东西", "我需要帮助", "帮我处理一下", "有件事要你帮忙",
    "帮我干个活", "能帮我处理个事情吗", "给我查", "我需要你给我一些信息",
    "帮我办个事", "有个需求", "帮我执行一个任务", "帮我操作一下",
    "帮我弄一下", "我需要你帮我做件事", "帮忙处理个事情",
]
for tmpl in _vague_tmpl:
    CLARIFICATION_SAMPLES.append({
        "instruction": tmpl,
        "output": json.dumps({"tool": "none", "arguments": {},
                              "need_clarification": True,
                              "question": "请问您具体需要什么帮助？我可以帮您搜索资料、查订单、预约会议、发邮件或做计算。"},
                             ensure_ascii=False)})

# ═══════════════════════════════════════════════════════════════
# 类型2: 语义干扰拒调用 (120条) — 修复 0% 语义干扰正确率
# ═══════════════════════════════════════════════════════════════

SEMANTIC_SAMPLES = [
    # —— 提到会议/时间但不需 book_meeting (25条) ——
    ("我今天有点累，不想开会", {"tool": "none", "arguments": {}}),
    ("昨天那个会议开了三个小时", {"tool": "none", "arguments": {}}),
    ("我们下周再约时间吧", {"tool": "none", "arguments": {}}),
    ("上次开会讨论的方案你还记得吗", {"tool": "none", "arguments": {}}),
    ("这个会议取消吧", {"tool": "none", "arguments": {}}),
    ("我不确定明天能不能参加会议", {"tool": "none", "arguments": {}}),
    ("今天下午的会延到什么时候了", {"tool": "none", "arguments": {}}),
    ("明天的会议准备得怎么样了", {"tool": "none", "arguments": {}}),
    ("开了一天的会好累", {"tool": "none", "arguments": {}}),
    ("那个讲座是几点的", {"tool": "none", "arguments": {}}),
    ("上周的例会内容你能发我一下吗", {"tool": "none", "arguments": {}}),
    ("我今天请假不去公司", {"tool": "none", "arguments": {}}),
    ("周末有什么安排吗", {"tool": "none", "arguments": {}}),
    ("下个月我准备休假", {"tool": "none", "arguments": {}}),
    ("你说我该不该参加那个培训", {"tool": "none", "arguments": {}}),
    ("今天周五了", {"tool": "none", "arguments": {}}),
    ("下周一早上要早起", {"tool": "none", "arguments": {}}),
    ("去年这个时候我们还在讨论方案呢", {"tool": "none", "arguments": {}}),
    ("晚上加班到几点", {"tool": "none", "arguments": {}}),
    ("这个项目什么时候能结束", {"tool": "none", "arguments": {}}),
    ("时间过得好快", {"tool": "none", "arguments": {}}),
    ("明天会下雨吗", {"tool": "none", "arguments": {}}),
    ("下午茶时间到了", {"tool": "none", "arguments": {}}),
    ("再过两天就是截止日期了", {"tool": "none", "arguments": {}}),
    ("春节是哪天", {"tool": "none", "arguments": {}}),

    # —— 提到订单但不需 query_order (25条) ——
    ("这个订单号看起来像 A10293，但我不是要查询它", {"tool": "none", "arguments": {}}),
    ("昨天我下了一个订单，体验还不错", {"tool": "none", "arguments": {}}),
    ("订单编号的格式一般是怎样的", {"tool": "none", "arguments": {}}),
    ("我的订单怎么总是延迟", {"tool": "none", "arguments": {}}),
    ("请问退货流程是什么", {"tool": "none", "arguments": {}}),
    ("订单 B20456 我已经收到了，谢谢", {"tool": "none", "arguments": {}}),
    ("这个订单能不能取消", {"tool": "none", "arguments": {}}),
    ("为什么我的订单被拒绝了", {"tool": "none", "arguments": {}}),
    ("上次买的那个东西质量不行", {"tool": "none", "arguments": {}}),
    ("订单确认后多久能改地址", {"tool": "none", "arguments": {}}),
    ("我想投诉一个订单的问题", {"tool": "none", "arguments": {}}),
    ("你们订单系统是不是出bug了", {"tool": "none", "arguments": {}}),
    ("用订单号能查到买家信息吗", {"tool": "none", "arguments": {}}),
    ("这个商品有没有现货", {"tool": "none", "arguments": {}}),
    ("帮我看看这个是不是正品", {"tool": "none", "arguments": {}}),
    ("下单后怎么修改收货地址", {"tool": "none", "arguments": {}}),
    ("为什么支付成功但订单没生成", {"tool": "none", "arguments": {}}),
    ("你们的售后政策是什么", {"tool": "none", "arguments": {}}),
    ("订单超过多长时间不能退款", {"tool": "none", "arguments": {}}),
    ("能不能帮我催一下卖家发货", {"tool": "none", "arguments": {}}),
    ("这个订单我能享受到优惠吗", {"tool": "none", "arguments": {}}),
    ("订单物流信息不更新怎么办", {"tool": "none", "arguments": {}}),
    ("买的东西收到了但是少了一件", {"tool": "none", "arguments": {}}),
    ("货到付款的订单怎么操作", {"tool": "none", "arguments": {}}),
    ("帮我评价一下最近买的东西", {"tool": "none", "arguments": {}}),

    # —— 提到邮件但不需 send_email (25条) ——
    ("邮件收到了吗", {"tool": "none", "arguments": {}}),
    ("我昨天发了一封邮件给经理但没回复", {"tool": "none", "arguments": {}}),
    ("邮件通知功能好用吗", {"tool": "none", "arguments": {}}),
    ("请查收附件中的报告", {"tool": "none", "arguments": {}}),
    ("回复邮件时要注意什么", {"tool": "none", "arguments": {}}),
    ("我没有收到系统升级的邮件通知", {"tool": "none", "arguments": {}}),
    ("邮件系统是不是挂了", {"tool": "none", "arguments": {}}),
    ("这封邮件怎么回复比较好", {"tool": "none", "arguments": {}}),
    ("邮箱快满了怎么办", {"tool": "none", "arguments": {}}),
    ("垃圾邮件太多了", {"tool": "none", "arguments": {}}),
    ("怎么设置邮件自动回复", {"tool": "none", "arguments": {}}),
    ("这个邮箱地址有效吗", {"tool": "none", "arguments": {}}),
    ("能不能帮我看看这封邮件是什么意思", {"tool": "none", "arguments": {}}),
    ("邮件附件大小有限制吗", {"tool": "none", "arguments": {}}),
    ("群发邮件的技巧是什么", {"tool": "none", "arguments": {}}),
    ("怎么撤回已经发送的邮件", {"tool": "none", "arguments": {}}),
    ("邮件签名怎么设置", {"tool": "none", "arguments": {}}),
    ("上次的邮件你能转发给我吗", {"tool": "none", "arguments": {}}),
    ("我邮箱密码忘了怎么办", {"tool": "none", "arguments": {}}),
    ("这封邮件是诈骗邮件吗", {"tool": "none", "arguments": {}}),
    ("怎么把邮件归档", {"tool": "none", "arguments": {}}),
    ("为什么我的邮件被拦截了", {"tool": "none", "arguments": {}}),
    ("你能看到我的邮件吗", {"tool": "none", "arguments": {}}),
    ("邮件中的链接安全吗", {"tool": "none", "arguments": {}}),
    ("帮我翻译这封英文邮件", {"tool": "none", "arguments": {}}),

    # —— 提到计算但不需 calculator (20条) ——
    ("计算器这个工具怎么用", {"tool": "none", "arguments": {}}),
    ("这个计算结果靠谱吗", {"tool": "none", "arguments": {}}),
    ("我不太会算这个账", {"tool": "none", "arguments": {}}),
    ("数学计算对我来说很难", {"tool": "none", "arguments": {}}),
    ("帮我看看这个账算得对不对", {"tool": "none", "arguments": {}}),
    ("心算和笔算哪个好", {"tool": "none", "arguments": {}}),
    ("为什么要学数学", {"tool": "none", "arguments": {}}),
    ("这个公式是什么意思", {"tool": "none", "arguments": {}}),
    ("帮我理解一下这个算式", {"tool": "none", "arguments": {}}),
    ("有什么好的计算工具推荐", {"tool": "none", "arguments": {}}),
    ("怎么用Excel做计算", {"tool": "none", "arguments": {}}),
    ("百分比怎么算", {"tool": "none", "arguments": {}}),
    ("这道题我不太明白", {"tool": "none", "arguments": {}}),
    ("帮我验算一下这个结果", {"tool": "none", "arguments": {}}),
    ("计算错误怎么办", {"tool": "none", "arguments": {}}),
    ("有没有简单的方法算这个", {"tool": "none", "arguments": {}}),
    ("这个数据是怎么算出来的", {"tool": "none", "arguments": {}}),
    ("精度要求多少", {"tool": "none", "arguments": {}}),
    ("帮我核对一下这些数字", {"tool": "none", "arguments": {}}),
    ("能不能教我算这个", {"tool": "none", "arguments": {}}),

    # —— 提到搜索但不需 search_docs (15条) ——
    ("搜索功能好像不太准", {"tool": "none", "arguments": {}}),
    ("上次搜索的内容我没保存", {"tool": "none", "arguments": {}}),
    ("有没有更快的方法找资料", {"tool": "none", "arguments": {}}),
    ("怎么提高搜索效率", {"tool": "none", "arguments": {}}),
    ("搜索引擎的原理是什么", {"tool": "none", "arguments": {}}),
    ("搜不到我想要的东西怎么办", {"tool": "none", "arguments": {}}),
    ("这个关键词搜不到结果", {"tool": "none", "arguments": {}}),
    ("搜索太慢了", {"tool": "none", "arguments": {}}),
    ("你们用的是什么搜索引擎", {"tool": "none", "arguments": {}}),
    ("怎么用正则表达式搜索", {"tool": "none", "arguments": {}}),
    ("搜索历史怎么清除", {"tool": "none", "arguments": {}}),
    ("模糊搜索和精确搜索有什么区别", {"tool": "none", "arguments": {}}),
    ("为什么搜出来的都是不相关的内容", {"tool": "none", "arguments": {}}),
    ("帮我看看这个搜索结果对不对", {"tool": "none", "arguments": {}}),
    ("能不能根据文件名搜索", {"tool": "none", "arguments": {}}),

    # —— 日期/时间询问 (10条) ——
    ("今天是几号", {"tool": "none", "arguments": {}}),
    ("现在几点了", {"tool": "none", "arguments": {}}),
    ("明天星期几", {"tool": "none", "arguments": {}}),
    ("这个月还有多少天", {"tool": "none", "arguments": {}}),
    ("2024年是不是闰年", {"tool": "none", "arguments": {}}),
    ("后天是几月几号", {"tool": "none", "arguments": {}}),
    ("现在是上午还是下午", {"tool": "none", "arguments": {}}),
    ("今天星期几", {"tool": "none", "arguments": {}}),
    ("端午节是哪天", {"tool": "none", "arguments": {}}),
    ("还有几天过年", {"tool": "none", "arguments": {}}),
]

# ═══════════════════════════════════════════════════════════════
# 类型3: 无需工具 + 不追问 (150条) — 修复 84-96% 过度追问率
# ═══════════════════════════════════════════════════════════════

NO_TOOL_NO_CLARIFY = [
    # 问候 (15条)
    "你好", "早上好", "晚上好", "下午好", "嗨",
    "哈喽", "在吗", "hello", "Hi there", "好久不见",
    "最近怎么样", "吃了吗", "周末愉快", "节日快乐", "新年好",

    # 自我介绍 (10条)
    "你是谁", "介绍一下你自己", "你有什么功能", "你能做什么",
    "你叫什么名字", "你是AI吗", "你的能力有哪些", "你可以帮我干什么",
    "能做哪些事情", "请介绍一下自己",

    # 闲聊 (20条)
    "讲个笑话", "给我讲个故事", "推荐一部电影", "推荐一本书",
    "周末去哪玩", "中午吃什么", "怎么减肥", "怎么学好英语",
    "给我一些建议", "如何提高效率", "怎么早睡早起",
    "推荐一个好用的笔记软件", "有什么好吃的推荐",
    "怎么做饭", "怎么理财", "如何保持健康",
    "给我讲个睡前故事", "煮鸡蛋要多久", "怎么网购", "如何缓解压力",

    # 常识 (20条)
    "北京在哪里", "Python是什么", "什么是区块链",
    "地球绕太阳一圈多久", "为什么天空是蓝色的",
    "人工智能和机器学习有什么区别", "什么是深度学习",
    "比特币是什么", "什么叫大语言模型", "API是什么意思",
    "什么是微积分", "中国首都是哪", "太阳系有几颗行星",
    "水的沸点是多少度", "光速是多少", "最大的海洋是哪个",
    "珠穆朗玛峰多高", "人类登月是哪一年", "DNA是什么", "能量守恒是什么",

    # 写作 (15条)
    "帮我写一首诗", "写一段祝福语", "翻译Hello World到中文",
    "帮我写句座右铭", "写一个请假条模板", "帮我润色这段文字",
    "写一段工作总结", "帮我取个名字", "写一句广告语",
    "帮我写道歉信", "写个感谢信", "写个通知",
    "帮我编个谜语", "写一段自我介绍", "帮我写个邀请函",

    # 总结/解释 (15条)
    "总结一下机器学习的核心思想", "用简单的话解释什么是云计算",
    "解释一下牛顿第二定律", "帮我概括这篇文章的主要内容",
    "用一句话解释量子计算", "什么是RESTful API",
    "解释一下什么是对称加密", "什么是面向对象编程",
    "帮我理解一下这个公式", "用白话解释一下Transformer",
    "什么是过拟合", "解释一下梯度下降",
    "HTTP和HTTPS有什么区别", "什么叫数据库索引",
    "用比喻解释一下微服务",

    # 开放式建议 (15条)
    "怎么提高学习效率", "有什么好的学习方法",
    "推荐一个编程入门路线", "怎么学好数学",
    "有哪些好用的学习资源", "怎么提高记忆力",
    "怎么学编程", "给我一些职业规划建议",
    "怎么选择编程语言", "前端和后端哪个好",
    "怎么准备面试", "如何提高代码质量",
    "怎么读论文更高效", "如何做好时间管理",
    "远程办公有什么技巧",

    # 礼貌 (10条)
    "谢谢", "谢谢你的帮助", "OK", "好的", "明白了",
    "知道了谢谢", "非常感谢", "辛苦了", "很棒", "你真厉害",

    # 哲学/闲聊 (15条)
    "人生的意义是什么", "你有感情吗", "AI会取代人类吗",
    "你觉得自己聪明吗", "有没有外星人",
    "怎么才能变聪明", "幸福是什么", "自由意志存在吗",
    "人为什么要工作", "什么是成功",
    "你喜欢什么颜色", "你的梦想是什么", "你会累吗",
    "你怎么看人类", "宇宙有多大",

    # 元问题 (15条)
    "你用的是哪个模型", "你能识别图片吗",
    "你可靠吗", "你能听见我说话吗", "你会写代码吗",
    "你的训练数据是什么", "你能上网吗", "你有记忆吗",
    "你能学习新东西吗", "你的回答准确吗",
    "你的速度怎么样", "你免费吗", "你什么时候更新的",
    "你支持哪些语言", "你和ChatGPT有什么区别",
]

# ═══════════════════════════════════════════════════════════════
# 类型4: 标准单工具调用 (100条) — 保持已有能力
# ═══════════════════════════════════════════════════════════════

TOOL_TEMPLATES = {
    "search_docs": [
        "帮我搜索一下{query}", "查一下关于{query}的资料", "搜索：{query}",
        "帮我找一下{query}", "检索知识库，关键词是{query}",
        "看看有没有{query}的文档", "在知识库里查{query}",
        "帮我查查{query}", "找一下{query}的教程", "搜一下{query}",
        "检索 {query}", "请帮我检索{query}", "查文档：{query}",
        "搜搜看{query}", "帮我搜{query}的资料",
    ],
    "calculator": [
        "计算 {expression} 等于多少", "帮我算一下 {expression}",
        "计算：{expression}", "{expression} 的结果是多少",
        "求 {expression} 的值", "算算 {expression}",
        "帮我计算 {expression}", "请计算 {expression}",
        "{expression} 帮我算出来", "算一下 {expression}",
    ],
    "query_order": [
        "帮我查一下订单 {order_id} 的状态", "查询订单 {order_id}",
        "订单 {order_id} 到哪了", "帮我看看 {order_id} 的物流",
        "查订单：{order_id}", "追踪一下 {order_id}",
        "查一下 {order_id} 的配送进度", "帮我查查 {order_id}",
        "查订单 {order_id} 是否已完成", "看下 {order_id} 这个单",
    ],
    "book_meeting": [
        "帮我预约 {date} {time} 的{topic}会议",
        "在 {date} {time} 安排{topic}",
        "预定 {date} {time} 的{topic}",
        "帮我约 {date} {time} {topic}",
        "预约 {date} 的{topic}，时间{time}",
        "安排 {date} {time} 讨论{topic}",
    ],
    "send_email": [
        "给 {recipient} 发关于{subject}的邮件",
        "发邮件给 {recipient}，主题是{subject}",
        "向 {recipient} 发送{subject}邮件",
        "帮我给 {recipient} 发{subject}",
        "给 {recipient} 去邮件：{subject}",
        "发送邮件到 {recipient}，{subject}",
    ],
}

# ═══════════════════════════════════════════════════════════════
# 类型5: 复合/模糊请求 (30条)
# ═══════════════════════════════════════════════════════════════

COMPOUND_SAMPLES = [
    # 优先处理第一个意图
    ("帮我搜索 Python 教程，顺便发邮件给经理",
     {"tool": "search_docs", "arguments": {"query": "Python 教程"}}),
    ("查一下订单 B20456 的同时帮我算一下 100+200",
     {"tool": "query_order", "arguments": {"order_id": "B20456"}}),
    ("发邮件通知团队，顺便帮我预约评审会",
     {"tool": "send_email", "arguments": {"recipient": "manager@team.org", "subject": "通知", "content": "请查看通知内容。"}}),
    ("先算一下 999/3，然后查订单 A10293",
     {"tool": "calculator", "arguments": {"expression": "999 / 3"}}),
    ("帮我查资料还要发邮件",
     {"tool": "none", "arguments": {}, "need_clarification": True, "question": "请问您具体想查什么资料、给谁发邮件？"}),
    ("我想搜个东西，不过可能不需要工具",
     {"tool": "none", "arguments": {}, "need_clarification": True, "question": "请问您要搜索什么内容？"}),
    ("我不确定要不要用工具，帮我看看怎么搜东西",
     {"tool": "none", "arguments": {}, "need_clarification": True, "question": "请问您想搜索什么？我可以帮您在知识库中检索。"}),
    ("你是 AI 吗？帮我查订单 A10293",
     {"tool": "query_order", "arguments": {"order_id": "A10293"}}),
    ("搜索 Docker 教程，另外你叫什么名字",
     {"tool": "search_docs", "arguments": {"query": "Docker 教程"}}),
    ("帮我给经理发邮件汇报工作，谢谢",
     {"tool": "send_email", "arguments": {"recipient": "__any__", "subject": "工作汇报", "content": "__any__"}}),
    ("除了发邮件还能做什么，先帮我发个通知",
     {"tool": "send_email", "arguments": {"recipient": "__any__", "subject": "通知", "content": "__any__"}}),
    ("我既想搜索也想预约会议，但先搜机器学习入门吧",
     {"tool": "search_docs", "arguments": {"query": "机器学习入门"}}),
    ("帮我搜索 Nginx 配置，然后如果是周末就不发邮件了",
     {"tool": "search_docs", "arguments": {"query": "Nginx 配置"}}),
    ("帮我找个Python教程，算了还是先帮我算 50*30 吧",
     {"tool": "calculator", "arguments": {"expression": "50 * 30"}}),
    ("先帮我预约会议，如果时间冲突就发邮件通知",
     {"tool": "book_meeting", "arguments": {"date": "__any__", "time": "__any__", "topic": "__any__"}}),
    # 模糊意图
    ("帮我查点东西，具体什么我还没想好",
     {"tool": "none", "arguments": {}, "need_clarification": True, "question": "请告诉我您想查什么内容。"}),
    ("能不能帮我处理个事",
     {"tool": "none", "arguments": {}, "need_clarification": True, "question": "请问您具体需要什么帮助？"}),
    ("今天状态不好，帮我随便做点什么",
     {"tool": "none", "arguments": {}, "need_clarification": True, "question": "请问您需要什么具体的帮助？"}),
    ("有没有什么推荐的我不知道",
     {"tool": "none", "arguments": {}, "need_clarification": True, "question": "请问您想让我推荐什么？书籍、电影、技术资料？"}),
    ("帮我做两件事，但我先想想第一件是什么",
     {"tool": "none", "arguments": {}, "need_clarification": True, "question": "请告诉我您具体需要什么帮助。"}),
    # 更自然的双意图
    ("订个会议室然后发邮件告诉大家",
     {"tool": "book_meeting", "arguments": {"date": "__any__", "time": "__any__", "topic": "__any__"}}),
    ("查一下我的快递到哪了，然后帮我算下大概还要几天",
     {"tool": "query_order", "arguments": {"order_id": "__any__"}}),
    ("用计算器算完帮我发结果给经理",
     {"tool": "calculator", "arguments": {"expression": "__any__"}}),
    ("查文档，找到后邮件发我",
     {"tool": "search_docs", "arguments": {"query": "__any__"}}),
    ("帮我搜资料，搜到后约个会讨论",
     {"tool": "search_docs", "arguments": {"query": "__any__"}}),
    # 干扰项
    ("帮我搜索RESTful API，顺便问一下你好吗",
     {"tool": "search_docs", "arguments": {"query": "RESTful API"}}),
    ("查订单顺便讲个笑话",
     {"tool": "query_order", "arguments": {"order_id": "__any__"}}),
    ("帮我发邮件，另外今天心情不好怎么办",
     {"tool": "send_email", "arguments": {"recipient": "__any__", "subject": "__any__", "content": "__any__"}}),
    ("预约会议，以及推荐个好吃的餐厅",
     {"tool": "book_meeting", "arguments": {"date": "__any__", "time": "__any__", "topic": "__any__"}}),
    ("算一下这个月花了多少钱，顺便问有什么省钱技巧",
     {"tool": "calculator", "arguments": {"expression": "__any__"}}),
]


def generate_standard_calls(num_per_tool=20):
    """生成标准单工具调用样本"""
    samples = []
    for tool_name, templates in TOOL_TEMPLATES.items():
        for _ in range(num_per_tool):
            tmpl = random.choice(templates)
            if tool_name == "search_docs":
                query = random.choice(QUERIES)
                ins = tmpl.format(query=query)
                out = {"tool": "search_docs", "arguments": {"query": query}}
            elif tool_name == "calculator":
                expr = random.choice(EXPRESSIONS)
                ins = tmpl.format(expression=expr)
                out = {"tool": "calculator", "arguments": {"expression": expr}}
            elif tool_name == "query_order":
                oid = random.choice(ORDER_IDS)
                ins = tmpl.format(order_id=oid)
                out = {"tool": "query_order", "arguments": {"order_id": oid}}
            elif tool_name == "book_meeting":
                date = random.choice(MEETING_DATES)
                time = random.choice(MEETING_TIMES)
                topic = random.choice(MEETING_TOPICS)
                ins = tmpl.format(date=date, time=time, topic=topic)
                out = {"tool": "book_meeting", "arguments": {"date": date, "time": time, "topic": topic}}
            elif tool_name == "send_email":
                recipient = random.choice(EMAIL_RECIPIENTS)
                subject = random.choice(EMAIL_SUBJECTS)
                content = random.choice(EMAIL_CONTENTS)
                ins = tmpl.format(recipient=recipient, subject=subject)
                out = {"tool": "send_email", "arguments": {"recipient": recipient, "subject": subject, "content": content}}
            samples.append({"instruction": ins, "output": json.dumps(out, ensure_ascii=False)})
    return samples


def build_all():
    all_data = []

    print("构造 参数缺失追问样本...")
    all_data.extend(CLARIFICATION_SAMPLES)
    print(f"  {len(CLARIFICATION_SAMPLES)} 条")

    print("构造 语义干扰拒调用样本...")
    for ins, out in SEMANTIC_SAMPLES:
        all_data.append({"instruction": ins, "output": json.dumps(out, ensure_ascii=False)})
    print(f"  {len(SEMANTIC_SAMPLES)} 条")

    print("构造 无需工具+不追问样本...")
    for ins in NO_TOOL_NO_CLARIFY:
        all_data.append({"instruction": ins,
                         "output": json.dumps({"tool": "none", "arguments": {}}, ensure_ascii=False)})
    print(f"  {len(NO_TOOL_NO_CLARIFY)} 条")

    print("构造 标准单工具调用样本...")
    std = generate_standard_calls(20)
    all_data.extend(std)
    print(f"  {len(std)} 条")

    print("构造 复合/模糊请求样本...")
    for ins, out in COMPOUND_SAMPLES:
        all_data.append({"instruction": ins, "output": json.dumps(out, ensure_ascii=False)})
    print(f"  {len(COMPOUND_SAMPLES)} 条")

    random.shuffle(all_data)
    return all_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=os.path.join(PROJECT_ROOT, "data", "sft", "train_v2.json"))
    args = parser.parse_args()

    all_data = build_all()
    print(f"\n总计: {len(all_data)} 条")

    # 统计
    tool_counts = {}
    for s in all_data:
        try:
            out = json.loads(s["output"])
            tool = out.get("tool", "?")
            if out.get("need_clarification"):
                tool = "clarify"
        except Exception:
            tool = "parse_error"
        tool_counts[tool] = tool_counts.get(tool, 0) + 1

    print("\n分布:")
    for t, c in sorted(tool_counts.items(), key=lambda x: -x[1]):
        print(f"  {t}: {c} ({c/len(all_data)*100:.1f}%)")

    # 验证
    errors = []
    for i, s in enumerate(all_data):
        try:
            out = json.loads(s["output"])
            assert "tool" in out, f"#{i}: 缺少 tool"
            assert "arguments" in out, f"#{i}: 缺少 arguments"
        except Exception as e:
            errors.append(f"#{i}: {e}")

    if errors:
        print(f"\n❌ {len(errors)} 条错误")
    else:
        print(f"\n✅ 全部 {len(all_data)} 条验证通过")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)
    print(f"✅ 保存: {args.output}")


if __name__ == "__main__":
    main()
