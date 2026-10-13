import base64
from zhipuai import ZhipuAI

ZHIPU_API_KEY = "1f77175df0e24d349ec31af46e861bb8.6L5XNjlZFaP2oJLL"

client = ZhipuAI(api_key=ZHIPU_API_KEY)

PROMPT = """这是中国专利 CN100342976C《静电除尘器》说明书附图第 1 页的图 1（横截面示意图）。

请详细描述这张图，务必包含以下信息：
1. 图中所有标注的部件编号和名称：7（气流方向）、10（圆锥形部分）、6、6'、6"（配气带孔盘）、4（圆柱形部分）、5（台阶）、2（外壳）、3（中心轴线）、D（外壳直径）、X1、X2、X3（尺寸）、h1、h2（尺寸）、1（长度）
2. 气流方向 7 从左侧进入，依次经过哪些部件
3. 部件 4 和部件 5 的相对位置关系
4. 尺寸 X1、X2、X3 分别代表哪两个配气带孔盘之间的间隔距离
5. h1 和 h2 分别代表什么高度，如何确定配气带孔盘的位置
6. 圆锥形部分 10 的开口方向和收缩方向

请用一段完整的中文描述，保留所有部件编号。"""

with open(r"D:\专高六 工单\工单14\images\page7.png", "rb") as f:
    img_b64 = base64.b64encode(f.read()).decode()

resp = client.chat.completions.create(
    model="glm-4v-flash",
    messages=[{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": img_b64}},
            {"type": "text", "text": PROMPT}
        ]
    }]
)

caption = resp.choices[0].message.content
print(caption)

with open(r"D:\专高六 工单\工单14\images\page7_caption.txt", "w", encoding="utf-8") as f:
    f.write(caption)
print("\n已保存 page7_caption.txt")