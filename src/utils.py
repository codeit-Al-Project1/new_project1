######## 넌 바뀌면 안된다잉

import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

def get_optimizer(name, model, lr=1e-3, weight_decay=0, momentum=0.9):
    """
    주어진 이름(name)에 해당하는 옵티마이저를 생성하여 반환합니다.
    :param name: 사용할 옵티마이저 이름 (예: 'adam', 'sgd', 'adamw', 'rmsprop')
    :param model: 학습할 모델의 파라미터
    :param lr: 학습률
    :param weight_decay: 가중치 감쇠 (L2 정규화)
    :param momentum: SGD 및 RMSprop에서 사용할 momentum 값 (기본값: 0.9)
    :return: 선택된 옵티마이저 인스턴스
    """
    optimizers = {
        "sgd": optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay),
        "adam": optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay),
        "adamw": optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay),
        "rmsprop": optim.RMSprop(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    }

    if name.lower() not in optimizers:
        raise ValueError(f"지정되지 않은 옵티마이저: {name}. 사용 가능한 옵션: {list(optimizers.keys())}")

    return optimizers[name.lower()]

def get_scheduler(name, optimizer, step_size=10, gamma=0.1, T_max=50, patience=5):
    """
    주어진 이름(name)에 해당하는 스케줄러를 생성하여 반환합니다.
    :param name: 사용할 스케줄러 이름 (예: 'step', 'cosine', 'plateau', 'exponential')
    :param optimizer: 옵티마이저 인스턴스
    :param step_size: StepLR에서 사용하는 스텝 크기
    :param gamma: StepLR, ExponentialLR에서 사용되는 감쇠 계수
    :param T_max: CosineAnnealingLR에서 사용하는 주기 길이
    :param patience: ReduceLROnPlateau에서 사용하는 patience 값
    :return: 선택된 스케줄러 인스턴스
    """
    schedulers = {
        "step": lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma),
        "cosine": lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max),
        "plateau": lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=gamma, patience=patience),
        "exponential": lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    }

    if name.lower() not in schedulers:
        raise ValueError(f"지정되지 않은 스케줄러: {name}. 사용 가능한 옵션: {list(schedulers.keys())}")

    return schedulers[name.lower()]

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

if __name__ == "__main__":