
def iou(b1, b2):
    (x1,y1,w1,h1) = b1
    (x2,y2,w2,h2) = b2
    xa = max(x1, x2)
    ya = max(y1, y2)
    xb = min(x1+w1, x2+w2)
    yb = min(y1+h1, y2+h2)
    inter = max(0, xb-xa) * max(0, yb-ya)
    union = w1*h1 + w2*h2 - inter
    if union == 0:
        return 0.0
    return inter/union
