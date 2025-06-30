import torch
from d2l import torch as d2l

d2l.set_figsize()
img = d2l.plt.imread(r'D:\learn\img\catdog.png')
d2l.plt.imshow(img)

def box_corner_to_center(boxes):
    """从（左上，右下）转换到（中间，宽度，高度）"""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    boxes = torch.stack((cx, cy, w, h), axis=1)
    return boxes

def box_center_to_corner(boxes):
    """从（中间，宽度，高度）转换到（左上，右下）"""
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    boxes = torch.stack((x1, y1, x2, y2), axis=1)
    return boxes

# 补全猫的边界框坐标（假设最后一个值为493）
dog_bbox = [50.0, 40.0, 280.0, 220.0]  # 狗的边界框
cat_bbox = [260.0, 60.0, 320.0, 180.0]

# 转换为PyTorch张量
boxes = torch.tensor((dog_bbox, cat_bbox))

# 测试转换函数的正确性
boxes_center = box_corner_to_center(boxes)
boxes_corner = box_center_to_corner(boxes_center)

# 验证转换的可逆性
print(f"坐标转换是否可逆: {torch.allclose(boxes, boxes_corner)}")

# 定义辅助函数来绘制边界框
def bbox_to_rect(bbox, color):
    """Convert bounding box to matplotlib format."""
    # 将边界框(左上x,左上y,右下x,右下y)格式转换为matplotlib格式：
    # ((左上x,左上y),宽,高)
    return d2l.plt.Rectangle(
        xy=(bbox[0], bbox[1]), width=bbox[2]-bbox[0], height=bbox[3]-bbox[1],
        fill=False, edgecolor=color, linewidth=2)

# 在图像上绘制边界框
fig = d2l.plt.gca()
fig.add_patch(bbox_to_rect(dog_bbox, 'blue'))
fig.add_patch(bbox_to_rect(cat_bbox, 'red'))

# 添加标签
d2l.plt.text(dog_bbox[0], dog_bbox[1], 'dog',
             color='white', fontsize=12, backgroundcolor='blue')
d2l.plt.text(cat_bbox[0], cat_bbox[1], 'cat',
             color='white', fontsize=12, backgroundcolor='red')

d2l.plt.show()  # 显示图像和边界框