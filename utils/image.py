import cv2


def resize2(img, dest_sz, square=False, border=cv2.BORDER_CONSTANT, value=0):
    h, w = img.shape[:2]
    scale = dest_sz / max(img.shape[:2])
    w = int(w * scale + 0.5)
    h = int(h * scale + 0.5)
    img = cv2.resize(img, (w,h))
    pt = pb = pl = pr = 0
    if square:
        h, w = img.shape[:2]
        if w > h:
            d = w - h
            pt = d >> 1
            pb = d - pl
        else:
            d = h - w
            pl = d >> 1
            pr = d - pl
        img = cv2.copyMakeBorder(img, pt, pb, pl, pr, border, value=value)
    return img, (scale, pt, pb, pl, pr)