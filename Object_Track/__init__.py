from Object_Track.deep_sort import DeepSort

__all__ = ['DeepSort', 'build_tracker']

def build_tracker(use_cuda):
    return DeepSort('F:\\Animal_behavior_analysis\\Object_Track\\deep\\checkpoint\\ckpt.t7',# namesfile=cfg.DEEPSORT.CLASS_NAMES,
                max_dist=0.2, min_confidence=0.1, 
                nms_max_overlap=0.5, max_iou_distance=0.7,
                max_age=1800, n_init=3, nn_budget=100, use_cuda=True)
