"""场景-维度矩阵 + query 模板.

Layer 2 核心资产:
  - 6 种出行场景 × 12 个评估维度
  - 每个维度有场景特定的 query 模板
"""

from typing import Dict, List

# ═══════════════════════════════════════════════════
# 场景标签
# ═══════════════════════════════════════════════════

SCENE_LABELS = {
    "business":     "商务出差",
    "family":       "亲子家庭",
    "romance":      "情侣蜜月",
    "elder":        "带长辈出行",
    "friends":      "朋友出游",
    "solo":         "独自旅行",
    "practical":    "实务咨询",
}

SCENE_ORDER = ["business", "family", "romance", "elder", "friends", "solo", "practical"]

# ═══════════════════════════════════════════════════
# 场景-维度矩阵
#  ● = 核心维度（必查）  ○ = 次要维度（LLM 按 specifics 决定）
# ═══════════════════════════════════════════════════

MATRIX: Dict[str, Dict[str, str]] = {
    "network":      {"business": "●", "family": "—", "romance": "—", "elder": "—", "friends": "○", "solo": "○", "practical": "—"},
    "quiet":        {"business": "●", "family": "●", "romance": "●", "elder": "●", "friends": "—", "solo": "○", "practical": "—"},
    "checkout":     {"business": "●", "family": "—", "romance": "—", "elder": "—", "friends": "—", "solo": "—", "practical": "●"},
    "location":     {"business": "●", "family": "—", "romance": "—", "elder": "—", "friends": "●", "solo": "●", "practical": "—"},
    "safety":       {"business": "—", "family": "●", "romance": "—", "elder": "—", "friends": "—", "solo": "●", "practical": "—"},
    "children_fac": {"business": "—", "family": "●", "romance": "—", "elder": "—", "friends": "—", "solo": "—", "practical": "—"},
    "view_privacy": {"business": "—", "family": "—", "romance": "●", "elder": "—", "friends": "—", "solo": "—", "practical": "—"},
    "dining":       {"business": "—", "family": "—", "romance": "●", "elder": "—", "friends": "○", "solo": "—", "practical": "—"},
    "accessible":   {"business": "—", "family": "—", "romance": "—", "elder": "●", "friends": "—", "solo": "—", "practical": "—"},
    "service":      {"business": "—", "family": "—", "romance": "—", "elder": "●", "friends": "—", "solo": "—", "practical": "○"},
    "value":        {"business": "○", "family": "—", "romance": "—", "elder": "—", "friends": "●", "solo": "●", "practical": "—"},
    "surrounding":  {"business": "—", "family": "○", "romance": "—", "elder": "○", "friends": "●", "solo": "—", "practical": "—"},
    # 实务维度
    "checkin":      {"business": "—", "family": "—", "romance": "—", "elder": "—", "friends": "—", "solo": "—", "practical": "●"},
    "front_desk":   {"business": "—", "family": "—", "romance": "—", "elder": "—", "friends": "—", "solo": "—", "practical": "●"},
    "luggage":      {"business": "—", "family": "—", "romance": "—", "elder": "—", "friends": "—", "solo": "—", "practical": "○"},
    "shuttle":      {"business": "—", "family": "—", "romance": "—", "elder": "—", "friends": "—", "solo": "—", "practical": "○"},
    "payment":      {"business": "—", "family": "—", "romance": "—", "elder": "—", "friends": "—", "solo": "—", "practical": "○"},
    "pet":          {"business": "—", "family": "—", "romance": "—", "elder": "—", "friends": "—", "solo": "—", "practical": "○"},
    "room_amenity": {"business": "—", "family": "—", "romance": "—", "elder": "—", "friends": "—", "solo": "—", "practical": "○"},
    "dietary":      {"business": "—", "family": "—", "romance": "—", "elder": "—", "friends": "—", "solo": "—", "practical": "○"},
    "health_safety":{"business": "—", "family": "—", "romance": "—", "elder": "—", "friends": "—", "solo": "—", "practical": "○"},
    "tech_service": {"business": "—", "family": "—", "romance": "—", "elder": "—", "friends": "—", "solo": "—", "practical": "○"},
    "service_misc": {"business": "—", "family": "—", "romance": "—", "elder": "—", "friends": "—", "solo": "—", "practical": "○"},
    "special_req":  {"business": "—", "family": "—", "romance": "—", "elder": "—", "friends": "—", "solo": "—", "practical": "○"},
}

# ═══════════════════════════════════════════════════
# 维度中文名 + emoji
# ═══════════════════════════════════════════════════

DIMENSION_INFO = {
    "network":      {"label": "网络质量",    "emoji": "📶"},
    "quiet":        {"label": "安静程度",    "emoji": "🔇"},
    "checkout":     {"label": "退房效率",    "emoji": "⚡"},
    "location":     {"label": "位置交通",    "emoji": "📍"},
    "safety":       {"label": "安全",        "emoji": "🛡️"},
    "children_fac": {"label": "儿童设施",    "emoji": "🧸"},
    "view_privacy": {"label": "景观/私密",   "emoji": "🌿"},
    "dining":       {"label": "餐饮体验",    "emoji": "🍽️"},
    "accessible":   {"label": "无障碍设施",  "emoji": "♿"},
    "service":      {"label": "服务响应",    "emoji": "🤝"},
    "value":        {"label": "性价比",      "emoji": "💰"},
    "surrounding":  {"label": "周边配套",    "emoji": "🏪"},
    # 实务维度
    "checkin":      {"label": "入住时间",    "emoji": "🕐"},
    "front_desk":   {"label": "前台服务",    "emoji": "🛎️"},
    "luggage":      {"label": "行李寄存",    "emoji": "🧳"},
    "shuttle":      {"label": "接送机",      "emoji": "🚗"},
    "payment":      {"label": "支付方式",    "emoji": "💳"},
    "pet":          {"label": "宠物政策",    "emoji": "🐾"},
    "room_amenity": {"label": "客房设施",    "emoji": "🏠"},
    "dietary":      {"label": "饮食限制",    "emoji": "🥗"},
    "health_safety":{"label": "健康安全",    "emoji": "🏥"},
    "tech_service": {"label": "技术服务",    "emoji": "🔌"},
    "service_misc": {"label": "生活服务",    "emoji": "🧺"},
    "special_req":  {"label": "特殊需求",    "emoji": "🎁"},
}

# ═══════════════════════════════════════════════════
# 场景特定的 query 模板
# ═══════════════════════════════════════════════════

QUERY_TEMPLATES: Dict[str, Dict[str, str]] = {
    "business": {
        "network": "出差需要网络办公，WiFi稳定吗？视频会议会不会卡顿？",
        "quiet":   "出差需要安静休息和办公，房间隔音怎么样？有没有噪音投诉？",
        "checkout":"商务出行需要快速退房，退房效率高吗？支持延时退房吗？",
        "location":"酒店位置交通方便吗？离商圈/机场/火车站多远？",
    },
    "family": {
        "quiet":      "带宝宝出行需要安静环境，房间隔音好不好？电梯声音会不会吵？",
        "safety":     "酒店安全措施怎么样？房间有没有安全隐患？适不适合带小孩？",
        "children_fac":"有儿童乐园、亲子活动吗？提供婴儿床/儿童餐具吗？",
        "surrounding":"周边有适合带孩子去的地方吗？公园、游乐场？",
    },
    "romance": {
        "quiet":       "度蜜月想要私密安静，隔音效果好吗？走廊或隔壁噪音大吗？",
        "view_privacy":"房间景观怎么样？私密性好不好的？有没有浪漫布置？",
        "dining":      "酒店餐饮怎么样？有情侣套餐或浪漫晚餐吗？",
    },
    "elder": {
        "quiet":      "老人需要安静休息环境，隔音好吗？远离电梯的安静房间有没有？",
        "accessible": "有电梯吗？房间有没有无障碍设施？淋浴有没有扶手？",
        "service":    "服务响应及时吗？老人需要帮助时能不能第一时间赶到？",
        "surrounding":"周边有公园或散步的地方吗？附近有医院吗？",
    },
    "friends": {
        "location":   "位置方便吗？离热门景点/商圈远不远？",
        "value":      "性价比怎么样？多人入住划算吗？",
        "surrounding":"周边有什么好吃好玩的推荐吗？夜市、酒吧？",
    },
    "solo": {
        "location":   "位置安全方便吗？离地铁站/市中心多远？",
        "safety":     "一个人住安全吗？酒店安保怎么样？",
        "value":      "性价比高吗？独自入住有没有单人间优惠？",
    },
    "practical": {
        "checkin":    "凌晨到达 半夜入住 办理入住时间 前台24小时",
        "checkout":   "延时退房 最晚退房时间 半天房费 延迟退房",
        "front_desk": "前台24小时 半夜服务 夜间值班 前台态度",
        "luggage":    "行李寄存 寄存行李 免费寄存",
        "shuttle":    "接送机 机场大巴 班车 去机场",
        "payment":    "刷卡 信用卡 押金 支付宝 微信支付",
        "pet":        "宠物 带宠物入住 可以带狗 宠物友好 让带宠物 禁止宠物",
        "room_amenity": "枕头 床垫 淋浴 水压 马桶 窗户 空调 冰箱 电视 隔音 设施",
        "dietary":     "清真 素食 过敏 特殊饮食 回民 穆斯林",
        "health_safety":"除颤器 急救 医疗 防蚊 异味 甲醛 消防 安全",
        "tech_service":"HDMI USB 充电 网线 投屏 信号 打印机 传真",
        "service_misc":"送餐 room service 叫醒 洗衣 干洗 快递 租车",
        "special_req": "生日 布置 纪念日 求婚 高楼层 无烟房 加床 婴儿床",
    },
}


def get_dimensions_for_scene(scene: str) -> List[dict]:
    """返回场景的所有维度 (含等级)."""
    dims = []
    for dim_key, scene_map in MATRIX.items():
        level = scene_map.get(scene, "—")
        if level == "—":
            continue
        info = DIMENSION_INFO.get(dim_key, {"label": dim_key, "emoji": "❓"})
        template = QUERY_TEMPLATES.get(scene, {}).get(dim_key, "")
        dims.append({
            "key": dim_key,
            "label": info["label"],
            "emoji": info["emoji"],
            "level": level,        # ● 或 ○
            "query_template": template,
        })
    return dims


def get_scene_label(scene: str) -> str:
    return SCENE_LABELS.get(scene, scene)
