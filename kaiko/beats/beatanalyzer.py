import shutil
import numpy

def minsec(sec):
    sec = round(sec)
    sgn = +1 if sec >= 0 else -1
    min, sec = divmod(abs(sec), 60)
    min *= sgn
    return f"{min}:{sec:02d}"

def braille_scatter(width, height, xy, xlim, ylim):
    dx = (xlim[1] - xlim[0])/(width*2-1)
    dy = (ylim[1] - ylim[0])/(height*4-1)

    graph = numpy.zeros((height*4, width*2), dtype=bool)
    for x, y in xy:
        i = round((y-ylim[0])/dy)
        j = round((x-xlim[0])/dx)
        if i in range(height*4) and j in range(width*2):
            graph[i,j] = True

    graph = graph.reshape(height, 4, width, 2)
    block = 2**numpy.array([0, 3, 1, 4, 2, 5, 6, 7]).reshape(1, 4, 1, 2)
    code = 0x2800 + (graph * block).sum(axis=(1, 3))
    strs = numpy.concatenate((code, [[ord("\n")]]*height), axis=1).astype('i2').tostring().decode('utf-16')

    return strs

def show_analyze(tol, perfs):
    width = shutil.get_terminal_size().columns
    emax = tol*7
    start = min((perf.time for perf in perfs), default=0.0)
    end   = max((perf.time for perf in perfs), default=0.0)

    grad_minwidth = 15
    stat_minwidth = 15
    scat_height = 7
    acc_height = 2

    # grades infos
    grades = [perf.grade for perf in perfs if not perf.is_miss]
    miss_count = sum(perf.is_miss for perf in perfs)
    failed_count    = sum(not grade.is_wrong and abs(grade.shift) == 3 for grade in grades)
    bad_count       = sum(not grade.is_wrong and abs(grade.shift) == 2 for grade in grades)
    good_count      = sum(not grade.is_wrong and abs(grade.shift) == 1 for grade in grades)
    perfect_count   = sum(not grade.is_wrong and abs(grade.shift) == 0 for grade in grades)
    failed_wrong_count  = sum(grade.is_wrong and abs(grade.shift) == 3 for grade in grades)
    bad_wrong_count     = sum(grade.is_wrong and abs(grade.shift) == 2 for grade in grades)
    good_wrong_count    = sum(grade.is_wrong and abs(grade.shift) == 1 for grade in grades)
    perfect_wrong_count = sum(grade.is_wrong and abs(grade.shift) == 0 for grade in grades)
    accuracy = sum(2.0**(-abs(grade.shift)) for grade in grades) / len(perfs) if perfs else 0.0
    mistakes = sum(grade.is_wrong for grade in grades) / len(grades) if grades else 0.0

    grad_infos = [
        f"   miss: {   miss_count}",
        f" failed: { failed_count}+{ failed_wrong_count}",
        f"    bad: {    bad_count}+{    bad_wrong_count}",
        f"   good: {   good_count}+{   good_wrong_count}",
        f"perfect: {perfect_count}+{perfect_wrong_count}",
        "",
        "",
        f"accuracy: {accuracy:.1%}",
        f"mistakes: {mistakes:.2%}",
        "",
        ]

    # statistics infos
    errors = [(perf.time, perf.err) for perf in perfs if not perf.is_miss]
    misses = [perf.time for perf in perfs if perf.is_miss]
    err = sum(abs(err) for _, err in errors) / len(errors) if errors else 0.0
    ofs = sum(err for _, err in errors) / len(errors) if errors else 0.0
    dev = (sum((err-ofs)**2 for _, err in errors) / len(errors))**0.5 if errors else 0.0

    stat_infos = [
        f"err={err*1000:.3f} ms",
        f"ofs={ofs*1000:+.3f} ms",
        f"dev={dev*1000:.3f} ms",
        ]

    # timespan
    timespan = f"╡{minsec(start)} ~ {minsec(end)}╞"

    # layout
    grad_width = max(grad_minwidth, len(timespan), max(len(info_str) for info_str in grad_infos))
    stat_width = max(stat_minwidth, max(len(info_str) for info_str in stat_infos))
    scat_width = width - grad_width - stat_width - 4

    grad_top = "═"*grad_width
    grad_bot = timespan.center(grad_width, "═")
    scat_top = scat_bot = "═"*scat_width
    stat_top = stat_bot = "═"*stat_width
    grad_infos = [info_str.ljust(grad_width) for info_str in grad_infos]
    stat_infos = [info_str.ljust(stat_width) for info_str in stat_infos]

    # discretize data
    dx = (end - start)/(scat_width*2-1)
    dy = 2*emax/(scat_height*4-1)
    data = numpy.zeros((scat_height*4+1, scat_width*2), dtype=int)
    for time, err in errors:
        i = round((err+emax)/dy)
        j = round((time-start)/dx)
        if i in range(scat_height*4) and j in range(scat_width*2):
            data[i,j] += 1
    for time in misses:
        j = round((time-start)/dx)
        if j in range(scat_width*2):
            data[-1,j] += 1

    braille_block = 2**numpy.array([0, 3, 1, 4, 2, 5, 6, 7]).reshape(1, 4, 1, 2)

    # plot scatter
    scat_data = (data[:-1,:] > 0).reshape(scat_height, 4, scat_width, 2)
    scat_code = 0x2800 + (scat_data * braille_block).sum(axis=(1, 3)).astype('i2')
    scat_graph = [line.tostring().decode('utf-16') for line in scat_code]
    miss_data = (data[-1,:] > 0).reshape(scat_width, 2)
    miss_code = (miss_data * [1, 2]).sum(axis=-1)
    miss_graph = "".join("─╾╼━"[code] for code in miss_code)

    # plot statistics
    stat_data = data[:-1,:].sum(axis=1)
    stat_level = numpy.linspace(0, numpy.max(stat_data), stat_width*2, endpoint=False)
    stat_data = (stat_level[None,:] < stat_data[:,None]).reshape(scat_height, 4, stat_width, 2)
    stat_code = 0x2800 + (stat_data * braille_block).sum(axis=(1, 3)).astype('i2')
    stat_graph = [line.tostring().decode('utf-16') for line in stat_code]

    # plot accuracies
    acc_weight = 2.0**numpy.array([-3, -2, -1, 0, -1, -2, -3])
    acc_data = (data[:-1,:].reshape(scat_height, 4, scat_width, 2).sum(axis=(1,3)) * acc_weight[:,None]).sum(axis=0)
    acc_data /= numpy.maximum(1, data.sum(axis=0).reshape(scat_width, 2).sum(axis=1))
    acc_level = numpy.arange(acc_height)*8
    acc_code = 0x2580 + numpy.clip(acc_data[None,:]*acc_height*8 - acc_level[::-1,None], 0, 8).astype('i2')
    acc_code[acc_code==0x2580] = ord(" ")
    acc_graph = [line.tostring().decode('utf-16') for line in acc_code]

    # print
    print("╒" + grad_top      + "╤" + scat_top      + "╤" + stat_top      + "╕")
    print("│" + grad_infos[0] + "│" + scat_graph[0] + "│" + stat_graph[0] + "│")
    print("│" + grad_infos[1] + "│" + scat_graph[1] + "│" + stat_graph[1] + "│")
    print("│" + grad_infos[2] + "│" + scat_graph[2] + "│" + stat_graph[2] + "│")
    print("│" + grad_infos[3] + "│" + scat_graph[3] + "│" + stat_graph[3] + "│")
    print("│" + grad_infos[4] + "│" + scat_graph[4] + "│" + stat_graph[4] + "│")
    print("│" + grad_infos[5] + "│" + scat_graph[5] + "│" + stat_graph[5] + "│")
    print("│" + grad_infos[6] + "│" + scat_graph[6] + "│" + stat_graph[6] + "│")
    print("│" + grad_infos[7] + "├" + miss_graph    + "┤" + stat_infos[0] + "│")
    print("│" + grad_infos[8] + "│" + acc_graph[0]  + "│" + stat_infos[1] + "│")
    print("│" + grad_infos[9] + "│" + acc_graph[1]  + "│" + stat_infos[2] + "│")
    print("╘" + grad_bot      + "╧" + scat_bot      + "╧" + stat_bot      + "╛")
