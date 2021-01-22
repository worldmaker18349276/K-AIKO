import wcwidth


def cover(*rans):
    start = min(ran.start for ran in rans)
    stop = max(ran.stop for ran in rans)
    return range(start, stop)

def clamp(ran, ran_):
    start = min(max(ran.start, ran_.start), ran.stop)
    stop = max(min(ran.stop, ran_.stop), ran.start)
    return range(start, stop)

def addtext(view, y, x, text, ymask=slice(None,None), xmask=slice(None,None)):
    yran = range(len(view))
    xran = range(len(view[0]) if view else 0)

    for ch in text:
        width = wcwidth.wcwidth(ch)

        if ch == "\t":
            x += 1

        elif ch == "\b":
            x -= 1

        elif ch == "\v":
            y += 1

        elif ch == "\f":
            y -= 1

        elif ch == "\x00":
            pass

        elif width == 0:
            x_ = x - 1
            if y in yran and x_ in xran and view[y][x_] == "":
                x_ -= 1
            if y in yran[ymask] and x_ in xran[xmask]:
                view[y][x_] += ch

        elif width == 1:
            if y in yran[ymask] and x in xran[xmask]:
                if x-1 in xran and view[y][x] == "":
                    view[y][x-1] = " "
                if x+1 in xran and view[y][x+1] == "":
                    view[y][x+1] = " "
                view[y][x] = ch
            x += 1

        elif width == 2:
            x_ = x + 1
            if y in yran[ymask] and x in xran[xmask] and x_ in xran[xmask]:
                if x-1 in xran and view[y][x] == "":
                    view[y][x-1] = " "
                if x_+1 in xran and view[y][x_+1] == "":
                    view[y][x_+1] = " "
                view[y][x] = ch
                view[y][x_] = ""
            x += 2

        else:
            raise ValueError

    return view, y, x

def textrange(y, x, text):
    ystart = ystop = y
    xstart = xstop = x

    for ch in text:
        width = wcwidth.wcwidth(ch)

        if ch == "\t":
            x += 1

        elif ch == "\b":
            x -= 1

        elif ch == "\v":
            y += 1

        elif ch == "\f":
            y -= 1

        elif ch == "\x00":
            pass

        elif width == 0:
            ystart = min(ystart, y)
            ystop = max(ystop, y+1)
            xstart = min(xstart, x-1)
            xstop = max(xstop, x)

        elif width == 1:
            ystart = min(ystart, y)
            ystop = max(ystop, y+1)
            xstart = min(xstart, x)
            xstop = max(xstop, x+1)
            x += 1

        elif width == 2:
            ystart = min(ystart, y)
            ystop = max(ystop, y+1)
            xstart = min(xstart, x)
            xstop = max(xstop, x+2)
            x += 2

        else:
            raise ValueError

    return range(ystart, ystop), range(xstart, xstop), y, x

def newpad(view, fill=" ", ymask=slice(None,None), xmask=slice(None,None)):
    ys = range(len(view))[ymask]
    xs = range(len(view[0]) if view else 0)[xmask]
    pad = [[fill for _ in xs] for _ in ys]
    return pad, ys.start, xs.start

def addpad(view, y, x, pad, ymask=slice(None,None), xmask=slice(None,None)):
    yran = range(len(view))
    xran = range(len(view[0]) if view else 0)
    ys = clamp(range(y, y+len(pad)), yran[ymask])
    xs = clamp(range(x, x+len(pad[0]) if pad else x), xran[xmask])

    if ys and xs:
        for y_ in ys:
            if xs.start-1 in xran and view[y_][xs.start] == "":
                view[y_][xs.start-1] = " "
            if xs.stop in xran and view[y_][xs.stop] == "":
                view[y_][xs.stop] = " "
            for x_ in xs:
                view[y_][x_] = pad[y_-y][x_-x]

    return view, ys, xs

def clear(view, ymask=slice(None,None), xmask=slice(None,None)):
    yran = range(len(view))
    xran = range(len(view[0]) if view else 0)
    ys = yran[ymask]
    xs = xran[xmask]

    for y in ys:
        if xs.start-1 in xran and view[y][xs.start] == "":
            view[y][xs.start-1] = " "
        if xs.stop in xran and view[y][xs.stop] == "":
            view[y][xs.stop] = " "
        for x in xs:
            view[y][x] = " "

    return view
