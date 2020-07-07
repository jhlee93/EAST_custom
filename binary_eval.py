def bbox_iou(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0

    # abs area
    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    # compute the intersection over union by taking the intersection
    iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou

def polygon_to_rect(coordinate):
    x1 = max(coordinate[0], coordinate[6])
    y1 = max(coordinate[1], coordinate[3])
    x2 = max(coordinate[2], coordinate[4])
    y2 = max(coordinate[5], coordinate[7])

    return [x1, y1, x2, y2]


def confusion_matrix(boxesA, boxesB, iou_thresh):
    """
    boxesA : prediction boxes list
    boxesB : grount-truth boxes list
    """

    return_bboxes = []
    TP, FP, FN = 0,0,0
    # num of bboxes
    num_boxesA = len(boxesA)
    num_boxesB = len(boxesB)

    if num_boxesA >= num_boxesB: # FP detection
        FP = num_boxesA - num_boxesB

    else: # Detection false
        FN = num_boxesB - num_boxesA

    for a in boxesA: # prediction
        # LT, RT, RB, LB
        # coordA = [int(x) for x in a.split(',')[:8]]
        coordA = a[:8]
        boxA = polygon_to_rect(coordA)

        ious = []
        for b in boxesB: # gt
            # coordB = [int(x) for x in b.split(',')[:8]]
            coordB = b[:8]
            boxB = polygon_to_rect(coordB)
            iou = bbox_iou(boxA, boxB)
            ious.append(iou)

        fiou = max(ious)
        
        # IoU base
        if fiou >= iou_thresh:
            TP += 1
            return_bboxes.append(coordA.tolist())

        elif fiou == 0.:
            FP += 1

        else: # detection false
            FN += 1
    
    # Confusion Matrix, Recall, Precision
    precision = 0
    recall = 0
    # if (TP > 0) and (FP > 0):

    if TP != 0:
        recall = TP / num_boxesB
        precision = TP / (TP + FP)

    return TP, FP, FN, recall, precision, return_bboxes


        








