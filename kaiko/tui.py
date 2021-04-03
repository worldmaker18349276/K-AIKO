import wcwidth


def parse(text):
    text = iter(text)
    attr_str = ""
    attr_width = 0

    try:
        while True:
            ch = next(text)

            if ch == "\x1b":
                attr_str += ch
                ch = next(text)

                if ch != "[":
                    raise ValueError
                attr_str += ch
                ch = next(text)

                if ch == "m":
                    attr_str += ch
                    yield (attr_str, attr_width)
                    attr_str = ""
                    attr_width = 0
                    continue

                while True:
                    if ch in "0123456789;":
                        attr_str += ch
                        ch = next(text)
                    elif ch == "m":
                        attr_str += ch
                        break
                    else:
                        raise ValueError

            elif ch in "\t\b\r\x00":
                if attr_str:
                    raise ValueError
                yield (ch, -1)

            else:
                width = wcwidth.wcwidth(ch)
                if width == -1:
                    raise ValueError

                elif attr_str:
                    attr_str += ch
                    attr_width += width

                else:
                    yield (ch, width)

    except StopIteration:
        if attr_str:
            raise ValueError
        return


def cover(*rans):
    start = min(ran.start for ran in rans)
    stop = max(ran.stop for ran in rans)
    return range(start, stop)

def clamp(ran, ran_):
    start = min(max(ran.start, ran_.start), ran.stop)
    stop = max(min(ran.stop, ran_.stop), ran.start)
    return range(start, stop)


def addtext1(view, width, x, text, xmask=slice(None,None)):
    xran = range(width)
    x0 = x

    for ch, width in parse(text):
        if ch == "\t":
            x += 1

        elif ch == "\b":
            x -= 1

        elif ch == "\r":
            x = x0

        elif ch == "\x00":
            pass

        elif width == 0:
            x_ = x - 1
            while x_ in xran and view[x_] == "":
                x_ -= 1
            if x_ in xran[xmask]:
                view[x_] += ch

        else:
            x_ = x + width - 1
            if x in xran[xmask] and x_ in xran[xmask]:
                x1 = x
                if view[x1] == "":
                    while view[x1] == "":
                        view[x1] = " "
                        x1 -= 1
                    view[x1] = " "

                x2 = x_+1
                while x2 in xran and view[x2] == "":
                    view[x2] = " "
                    x2 += 1

                view[x] = ch
                for dx in range(1, width):
                    view[x+dx] = ""
            x += width

    return view, x

def textrange1(x, text):
    xstart = xstop = x0 = x

    for ch, width in parse(text):
        if ch == "\t":
            x += 1

        elif ch == "\b":
            x -= 1

        elif ch == "\r":
            x = x0

        elif ch == "\x00":
            pass

        elif width == 0:
            xstart = min(xstart, x-1)
            xstop = max(xstop, x)

        else:
            xstart = min(xstart, x)
            xstop = max(xstop, x+width)
            x += width

    return range(xstart, xstop), x

def newwin1(width):
    return [" "]*width

def newpad1(view, width, fill=" ", xmask=slice(None,None)):
    xs = range(width)[xmask]
    pad_width = len(xs)
    pad = newwin1(pad_width)
    return pad, xs.start, pad_width

def addpad1(view, width, x, pad, pad_width, xmask=slice(None,None)):
    xran = range(width)
    xs = clamp(range(x, x+pad_width), xran[xmask])

    if xs:
        x1 = xs.start
        if view[x1] == "":
            while view[x1] == "":
                view[x1] = " "
                x1 -= 1
            view[x1] = " "

        x2 = xs.stop
        while x2 in xran and view[x2] == "":
            view[x2] = " "
            x2 += 1

        for x_ in xs:
            view[x_] = pad[x_-x]

    return view, xs

def select1(view, width, xmask=slice(None,None)):
    if xmask.start is None and xmask.stop is None:
        return view

    xran = range(width)
    xs = xran[xmask]

    x1 = xs.start
    x2 = xs.stop
    if xs:
        while view[x1] == "":
            x1 -= 1

        while x2 in xran and view[x2] == "":
            x2 += 1

    return range(x1, x2)

def clear1(view, width, xmask=slice(None,None)):
    if xmask.start is None and xmask.stop is None:
        return view

    xran = range(width)
    xs = xran[xmask]

    if xs:
        x1 = xs.start
        if view[x1] == "":
            while view[x1] == "":
                view[x1] = " "
                x1 -= 1
            view[x1] = " "

        x2 = xs.stop
        while x2 in xran and view[x2] == "":
            view[x2] = " "
            x2 += 1

        for x in xs:
            view[x] = " "

    return view


def addtext2(view, height, width, y, x, text, ymask=slice(None,None), xmask=slice(None,None)):
    yran = range(height)
    xran = range(width)
    x0 = x

    for ch in text:
        width = wcwidth.wcwidth(ch)

        if ch == "\t":
            x += 1

        elif ch == "\b":
            x -= 1

        elif ch == "\r":
            x = x0

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

def textrange2(y, x, text):
    ystart = ystop = y
    xstart = xstop = x0 = x

    for ch in text:
        width = wcwidth.wcwidth(ch)

        if ch == "\t":
            x += 1

        elif ch == "\b":
            x -= 1

        elif ch == "\r":
            x = x0

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

def newwin2(height, width):
    return [[" "]*width for _ in range(height)]

def newpad2(view, height, width, fill=" ", ymask=slice(None,None), xmask=slice(None,None)):
    ys = range(height)[ymask]
    xs = range(width)[xmask]
    pad_height = len(ys)
    pad_width = len(xs)
    pad = newwin2(pad_height, pad_width)
    return pad, ys.start, xs.start, pad_height, pad_width

def addpad2(view, height, width, y, x, pad, pad_height, pad_width, ymask=slice(None,None), xmask=slice(None,None)):
    yran = range(height)
    xran = range(width)
    ys = clamp(range(y, y+pad_height), yran[ymask])
    xs = clamp(range(x, x+pad_width), xran[xmask])

    if ys and xs:
        for y_ in ys:
            if xs.start-1 in xran and view[y_][xs.start] == "":
                view[y_][xs.start-1] = " "
            if xs.stop in xran and view[y_][xs.stop] == "":
                view[y_][xs.stop] = " "
            for x_ in xs:
                view[y_][x_] = pad[y_-y][x_-x]

    return view, ys, xs

def clear2(view, height, width, ymask=slice(None,None), xmask=slice(None,None)):
    yran = range(height)
    xran = range(width)
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
