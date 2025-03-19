from matplotlib import pyplot as plt

# def draw_bbox(ax, box, text, color):
#     """
#     - ax: matplotlib Axes 객체
#     - box: 바운딩 박스 좌표 (x_min, y_min, x_max, y_max)
#     - text: 바운딩 박스 위에 표시할 텍스트
#     - color: 바운딩 박스와 텍스트의 색상
#     """
#     ax.add_patch(
#         plt.Rectangle(
#             xy=(box[0], box[1]),
#             width=box[2] - box[0],
#             height=box[3] - box[1],
#             fill=False,
#             edgecolor=color,
#             linewidth=2,
#         )
#     )
#     ax.annotate(
#         text=text,
#         xy=(box[0] - 5, box[1] - 5),
#         color=color,
#         weight="bold",
#         fontsize=13,
#     )