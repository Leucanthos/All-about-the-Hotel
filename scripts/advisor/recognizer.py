"""场景识别 — LLM few-shot 分类 + 实务查询预检."""

import re
from _shared.llm import classify_json, chat

# ── 实务查询关键词 ──

_PRACTICAL_PATTERNS = {
    "checkin": [
        "凌晨", "半夜", "午夜", "通宵", "几点.*到", "几点.*入住", "晚上.*到",
        "晚上.*入住", "到达.*入住", "入住.*时间", "办理入住", "可以入住",
        "能入住", "check.?in", "前台.*24", "24.*小时.*前台", "最晚.*入住",
        "提前.*入住", "早到",
    ],
    "checkout": [
        "退房.*时间", "延时.*退", "延迟.*退", "late.*checkout",
        "几点.*退房", "最晚.*退", "晚点.*退", "可以.*退房",
    ],
    "luggage": [
        "行李.*寄存", "寄存.*行李", "存放.*行李", "行李.*放", "行李.*存",
    ],
    "shuttle": [
        "接送机", "接机", "送机", "机场.*接", "接送.*车", "shuttle", "班车",
        "机场大巴", "去机场",
    ],
    "payment": [
        "刷卡", "信用卡", "visa", "mastercard", "支付宝", "微信.*付",
        "押金.*多少", "现金", "银联", "闪付",
    ],
    "room_amenity": [
        "枕头", "床垫", "被子", "被褥", "羽绒", "乳胶", "荞麦",
        "马桶", "浴缸", "淋浴", "水压", "花洒", "热水.*多久",
        "窗户", "开窗", "通风", "窗帘", "遮光",
        "电视", "冰箱", "保险箱", "保险柜", "minibar", "迷你吧",
        "吹风机", "热水壶", "电水壶", "熨斗", "衣架", "浴袍", "拖鞋",
        "空调.*遥控", "遥控器", "暖气", "暖氣", "地暖",
        "智能.*马桶", "卫洗丽", "电动.*窗帘", "智能.*客控",
        "wifi.*密码", "宽带", "上网", "网速", "wifi.*快",
        "转换.*插头", "插座", "插头", "电压",
        "避孕套", "condom", "安全套",
        "牙刷", "牙膏", "浴帽", "剃须", "梳子", "针线",
        "一次性.*用品", "洗漱.*用品", "护理.*用品", "提供.*免费",
    ],
    "dietary": [
        "清真", "素食", "吃素", "斋", "过敏", "无麸质", "gluten",
        "糖尿病", "低糖", "低盐", "忌口", "不吃.*肉",
        "回民", "穆斯林", "kosher", "halal", "vegetarian", "vegan",
        "早餐.*素食", "素食.*早餐", "特殊.*饮食",
    ],
    "accessibility": [
        "盲人", "视障", "导盲犬", "轮椅", "无障碍", "扶手", "坡道",
        "残疾人", "残障", "行动不便", "拐杖",
    ],
    "health_safety": [
        "除颤器", "AED", "急救", "医疗", "医生", "护士",
        "消防", "灭火器", "逃生", "安全.*出口",
        "防蚊", "蚊帐", "驱蚊", "蟑螂", "老鼠", "虫子",
        "甲醛", "异味", "味道.*大", "臭", "霉味", "烟味",
    ],
    "tech_service": [
        "HDMI", "USB", "type.?c", "充电器", "充电线", "数据线",
        "网线", "投屏", "投影", "打印机", "传真",
        "收.*验证码", "收.*短信", "手机.*信号", "信号.*好",
    ],
    "service_misc": [
        "room.?service", "送餐", "叫醒", "morning.?call",
        "外卖", "点.*到房间", "送.*到房间",
        "收快递", "代收", "快递.*代", "寄快递", "邮政",
        "洗衣", "干洗", "熨烫",
        "租车", "叫车", "打车", "包车",
    ],
    "pet": [
        "宠物", "带猫", "带狗", "带兔子", "带鹦鹉", "带猴子", "带.*动物",
        "带.*宠物", "可以带.*入住", "能带.*住", "让带", "允许.*宠物",
        "能带.*进去", "禁止.*宠物", "不让带", "pet.*friendly",
    ],
    "special_req": [
        "生日.*布置", "纪念日", "求婚", "蜜月.*布置", "周年.*庆祝",
        "惊喜", "布置.*房间", "鲜花", "蛋糕", "气球", "蜡烛",
        "高楼层", "低楼层", "靠电梯", "远离.*电梯", "角落.*房间",
        "吸烟", "抽烟", "烟房", "无烟房", "可吸烟", "smoking",
        "多加.*床", "加床", "婴儿床", "儿童.*床",
        "烧香", "拜佛", "祷告", "朝拜", "做礼拜", "佛堂",
        "开.*party", "开.*派对", "办.*party", "办.*聚会",
        "很多人.*聚", "聚会.*酒店",
    ],
}


def _detect_practical(query: str) -> dict | None:
    """检测是否属于实务/政策类查询。命中返回 scene info，未命中返回 None。

    在 LLM 场景识别之前执行，避免将"半夜入住"误判为 solo 等场景。
    """
    q = query.lower()
    for subtype, patterns in _PRACTICAL_PATTERNS.items():
        for pat in patterns:
            if re.search(pat, q, re.IGNORECASE):
                return {
                    "scene": "practical",
                    "specifics": {"subtype": subtype},
                    "confidence": 0.85,
                    "reason": f"关键词命中: {subtype}",
                    "router": "keyword",
                }
    return None


SCENE_PROMPT = """你是酒店入住决策顾问。识别以下用户输入属于哪种出行场景。

场景选项：
- business: 商务出差（开会、办公、出差）
- family: 亲子家庭（带小孩、宝宝、亲子）
- romance: 情侣蜜月（度蜜月、约会、浪漫）
- elder: 带长辈出行（带老人、父母、长辈）
- friends: 朋友出游（和朋友、结伴、聚会）
- solo: 独自旅行（一个人、独行、背包）
- practical: 实务咨询（询问酒店具体设施、服务、政策，而非出行场景判断）
  例如：入住退房时间、行李寄存、接送机、支付方式、客房设施（枕头/淋浴/马桶/窗户）、
  餐饮限制（清真/素食）、宠物政策、快递外卖、洗衣租车、加床吸烟、无障碍设施等。

用户输入：{user_input}

返回 JSON：{{"scene": "...", "specifics": {{"subtype": "..."}}, "confidence": 0.xx, "reason": "..."}}
如果 scene=practical，specifics.subtype 填具体问题类型（如 food_delivery, room_amenity, checkin 等）。
只返回 JSON，不要多余文字。"""


def recognize(user_input: str) -> dict:
    """LLM few-shot 场景识别.

    优先通过关键词预检实务查询（入住/退房/行李等），
    再走 LLM，最后回退关键词。

    Returns:
        {scene, specifics, confidence, reason}
        识别失败时 scene="general", confidence=0
    """
    # ── 0. 实务查询预检 ──
    practical = _detect_practical(user_input)
    if practical:
        return practical

    prompt = SCENE_PROMPT.format(user_input=user_input)
    try:
        result = classify_json("你是酒店入住决策顾问。", prompt, prefer="api")
        if result and "scene" in result:
            scene = result["scene"]
            valid = ["business", "family", "romance", "elder", "friends", "solo", "general", "practical"]
            if scene not in valid:
                scene = "general"
            confidence = min(float(result.get("confidence", 0)), 1.0)
            return {
                "scene": scene,
                "specifics": result.get("specifics", {}),
                "confidence": confidence,
                "reason": result.get("reason", ""),
                "router": "llm",
            }
    except Exception:
        pass

    # Fallback: 关键词匹配
    return _keyword_fallback(user_input)


def _keyword_fallback(query: str) -> dict:
    """关键词回退识别."""
    q = query.lower()

    # 商务
    if any(kw in q for kw in ["出差", "办公", "开会", "视频会议", "商务", "工作"]):
        return {"scene": "business", "specifics": {}, "confidence": 0.6, "reason": "关键词:出差/办公", "router": "keyword"}
    # 亲子
    if any(kw in q for kw in ["亲子", "宝宝", "小孩", "儿童", "带孩子", "娃"]):
        return {"scene": "family", "specifics": {}, "confidence": 0.6, "reason": "关键词:亲子/宝宝", "router": "keyword"}
    # 蜜月
    if any(kw in q for kw in ["蜜月", "浪漫", "情侣", "约会", "求婚"]):
        return {"scene": "romance", "specifics": {}, "confidence": 0.6, "reason": "关键词:蜜月/浪漫", "router": "keyword"}
    # 长辈
    if any(kw in q for kw in ["老人", "长辈", "父母", "爸妈", "奶奶", "爷爷", "轮椅"]):
        return {"scene": "elder", "specifics": {}, "confidence": 0.6, "reason": "关键词:老人/长辈", "router": "keyword"}
    # 朋友
    if any(kw in q for kw in ["朋友", "结伴", "聚会", "哥们", "闺蜜"]):
        return {"scene": "friends", "specifics": {}, "confidence": 0.6, "reason": "关键词:朋友/结伴", "router": "keyword"}
    # 独自
    if any(kw in q for kw in ["一个人", "独自", "独行", "背包", "单人"]):
        return {"scene": "solo", "specifics": {}, "confidence": 0.6, "reason": "关键词:一个人/独自", "router": "keyword"}

    return {"scene": "general", "specifics": {}, "confidence": 0.0, "reason": "无法识别", "router": "keyword"}
