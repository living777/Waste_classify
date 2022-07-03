import pickle

from pyecharts.charts import Bar
from pyecharts import options as opts
from pyecharts.commons.utils import JsCode
from pyecharts.globals import ThemeType

with open('label_count.pkl', 'rb') as f:
    data = pickle.load(f)

with open('train_label_count.pkl', 'rb') as f:
    train_data = pickle.load(f)

with open('val_label_count.pkl', 'rb') as f:
    val_data = pickle.load(f)

class2name = {
    0: "其他垃圾/一次性快餐盒",
    1: "其他垃圾/污损塑料",
    2: "其他垃圾/烟蒂",
    3: "其他垃圾/牙签",
    4: "其他垃圾/破碎花盆及碟碗",
    5: "其他垃圾/竹筷",
    6: "厨余垃圾/剩饭剩菜",
    7: "厨余垃圾/大骨头",
    8: "厨余垃圾/水果果皮",
    9: "厨余垃圾/水果果肉",
    10: "厨余垃圾叶渣",
    11: "厨余垃圾/菜叶菜根",
    12: "厨余垃圾壳",
    13: "厨余垃圾/鱼骨",
    14: "可回收物/充电宝",
    15: "可回收物/包",
    16: "可回收物/化妆品瓶",
    17: "可回收物/塑料玩具",
    18: "可回收物/塑料碗盆",
    19: "可回收物/塑料衣架",
    20: "可回收物/快递纸袋",
    21: "可回收物/插头电线",
    22: "可回收物/旧衣服",
    23: "可回收物/易拉罐",
    24: "可回收物/枕头",
    25: "可回收物/毛绒玩具",
    26: "可回收物/洗发水瓶",
    27: "可回收物/玻璃杯",
    28: "可回收物/皮鞋",
    29: "可回收物/砧板",
    30: "可回收物/纸板箱",
    31: "可回收物/调料瓶",
    32: "可回收物/酒瓶",
    33: "可回收物/金属食品罐",
    34: "可回收物/锅",
    35: "可回收物/食用油桶",
    36: "可回收物/饮料瓶",
    37: "有害垃圾/干电池",
    38: "有害垃圾/软膏",
    39: "有害垃圾/过期药物"
}

class2name = {key: value.split('/')[-1] for key, value in class2name.items()}

x_name = []
train_num = []
val_num = []

for label, name in class2name.items():
    x_name.append(name)
    all_num = data[label]
    train_num.append({'value': train_data[label], 'percent': train_data[label] / all_num})
    val_num.append({'value': val_data[label], 'percent': val_data[label] / all_num})


c = (
    Bar(init_opts=opts.InitOpts(theme=ThemeType.LIGHT))
    .add_xaxis(x_name)
    .add_yaxis("train", train_num, stack="stack1", category_gap="50%")
    .add_yaxis("val", val_num, stack="stack1", category_gap="50%")
    .set_series_opts(
        label_opts=opts.LabelOpts(
            position="right",
            formatter=JsCode(
                "function(x){return Number(x.data.percent * 100).toFixed() + '%';}"
            ),
        )
    )
    .set_global_opts(
        title_opts=opts.TitleOpts(title="训练集-测试集数据划分"),
        legend_opts=opts.LegendOpts(type_="scroll", pos_left="80%", orient="vertical"),
    )
    .render("train_val_percent.html")
)