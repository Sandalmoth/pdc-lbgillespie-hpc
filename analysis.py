"""
analysis scripts
"""

import csv
import copy
import re

import click
import numpy
import toml

from matplotlib import pyplot as plt


def split_output_file(filename):
    """
    split output file by its segments and transfer to appropriate parsers
    """

    re_splitter = re.compile(r' --- ([A-Z]+) \[([a-z]+)\] ---.*')

    segment = None
    splitmode = None
    filetype = None

    data = {}

    with open(filename, 'r') as infile:
        for line in infile:
            splitter = re_splitter.match(line)
            if splitter is not None:

                if segment is not None:
                    if filetype == 'toml':
                        data[splitmode] = toml.loads(copy.deepcopy(segment))
                    if filetype == 'tsv':
                        header, table = segment.split('\n', 1)
                        header = header.split('\t')
                        data[splitmode] = csv.DictReader(table.split('\n'), fieldnames=header, dialect=csv.excel_tab)

                splitmode = splitter.group(1)
                filetype = splitter.group(2)
                segment = ''

            else:
                segment += line

        if filetype == 'toml':
            data[splitmode] = toml.loads(copy.deepcopy(segment))
        if filetype == 'tsv':
            header, table = segment.split('\n', 1)
            header = header.split('\t')
            data[splitmode] = csv.DictReader(table.split('\n'), fieldnames=header, dialect=csv.excel_tab)

    return data


@click.group()
def main():
    """
    click construct
    """
    pass


@main.command()
@click.option('-i', '--inputs', type=click.Path(), multiple=True)
@click.option('-c', '--colours', type=str, multiple=True)
@click.option('-t', '--titles', type=str, multiple=True)
def timelines(inputs, colours, titles):
    """
    plot all timelines in the input files
    """

    fig, axs = plt.subplots()

    if not titles:
        titles = inputs[:]

    if not colours:
        colours = ['blue', 'orange', 'green', 'red', 'cyan', 'magenta', 'yellow', 'purple']
        while len(colours) < len(inputs):
            print('Possibly too many datasets. Provide colours with -c to prevent overlap.')
            colours.append('black')

    for inp, col, tit in zip(inputs, colours, titles):
        data = split_output_file(inp)
        timelines = [x for x in data['TIMELINES']]
        for i in range(data['FOOTER']['num_simulations']):
            xaxis = []
            yaxis = []
            for point in timelines:
                if int(point['index']) == i:
                    xaxis.append(float(point['time']))
                    yaxis.append(int(point['size']))
            if i == 0:
                axs.plot(xaxis, yaxis, color=col, alpha=0.5, label=tit)
            else:
                axs.plot(xaxis, yaxis, color=col, alpha=0.5)

    axs.set_xlabel('Time')
    axs.set_ylabel('Cell count')
    axs.legend()

    plt.show()


@main.command()
@click.option('-i', '--inputs', type=click.Path(), multiple=True)
@click.option('-t', '--titles', type=str, multiple=True)
def scaling(inputs, titles):
    """
    Check population size scaling of input data
    """

    # scaling plot
    fig, axs = plt.subplots(ncols=len(inputs))

    for i, inp in enumerate(inputs):
        data = split_output_file(inp)
        timeline_data = [x for x in data['TIMELINES']]
        timelines = {k: [float(x[k]) for x in timeline_data] for k in data['TIMELINES'].fieldnames}

        for s in range(data['FOOTER']['num_simulations']):
            axs[i].plot([z for z, w in zip(timelines['size'], timelines['index']) if w == float(s)],
                        [z**0.5 for z, w in zip(timelines['steptime'], timelines['index']) if w == float(s)])
        axs[i].set_title(titles[i])
        axs[i].set_xlabel('Population size')
        axs[i].set_ylabel('sqrt( steptime )')
        axs[i].ticklabel_format(style='sci', scilimits=(-128, 128))

    fig.set_size_inches(3*len(inputs), 3, forward=True)
    plt.tight_layout()

    plt.show()

    # speedup

    for i, inp1 in enumerate(inputs):
        for j, inp2 in enumerate(inputs):
            if i == j:
                continue

            fig, axs = plt.subplots(nrows=2)

            data1 = split_output_file(inp1)
            data2 = split_output_file(inp2)
            timeline_data1 = [x for x in data1['TIMELINES']]
            timeline_data2 = [x for x in data2['TIMELINES']]
            timelines1 = {k: [float(x[k]) for x in timeline_data1] for k in data1['TIMELINES'].fieldnames}
            timelines2 = {k: [float(x[k]) for x in timeline_data2] for k in data2['TIMELINES'].fieldnames}

            for s in range(min(data1['FOOTER']['num_simulations'], data2['FOOTER']['num_simulations'])):
                s = float(s)

                diff = [x - y for x, y in zip([z for z, w in zip(timelines1['size'], timelines1['index']) if w == s],
                                              [z for z, w in zip(timelines2['size'], timelines2['index']) if w == s])]
                avg = [(x + y) / 2 for x, y in zip([z for z, w in zip(timelines1['size'], timelines1['index']) if w == s],
                                                   [z for z, w in zip(timelines2['size'], timelines2['index']) if w == s])]
                speedup = [x/y for x, y in zip([z for z, w in zip(timelines1['steptime'], timelines1['index']) if w == s],
                                               [z for z, w in zip(timelines2['steptime'], timelines2['index']) if w == s])]

                axs[0].plot(range(len(diff)), [x / y for x, y in zip(diff, avg)])
                axs[1].plot(avg, speedup)

            axs[0].set_xlabel('Record point')
            axs[0].set_ylabel('Relative\n divergence')
            axs[1].set_xlabel('Population\n size')
            axs[1].set_ylabel('Speedup')

            fig.set_size_inches(4, 5, forward=True)
            plt.tight_layout()

            plt.show()


    # divergence

    for i, inp1 in enumerate(inputs):
        for j, inp2 in enumerate(inputs):
            if i == j:
                continue

            fig, axs = plt.subplots()

            data1 = split_output_file(inp1)
            data2 = split_output_file(inp2)
            timeline_data1 = [x for x in data1['TIMELINES']]
            timeline_data2 = [x for x in data2['TIMELINES']]
            timelines1 = {k: [float(x[k]) for x in timeline_data1] for k in data1['TIMELINES'].fieldnames}
            timelines2 = {k: [float(x[k]) for x in timeline_data2] for k in data2['TIMELINES'].fieldnames}

            for s in range(min(data1['FOOTER']['num_simulations'], data2['FOOTER']['num_simulations'])):
                s = float(s)

                diff = [x - y for x, y in zip([z for z, w in zip(timelines1['size'], timelines1['index']) if w == s],
                                              [z for z, w in zip(timelines2['size'], timelines2['index']) if w == s])]
                avg = [(x + y) / 2 for x, y in zip([z for z, w in zip(timelines1['size'], timelines1['index']) if w == s],
                                                   [z for z, w in zip(timelines2['size'], timelines2['index']) if w == s])]
                speedup = [x/y for x, y in zip([z for z, w in zip(timelines1['steptime'], timelines1['index']) if w == s],
                                               [z for z, w in zip(timelines2['steptime'], timelines2['index']) if w == s])]

                axs.plot(range(len(diff)), [x / y for x, y in zip(diff, avg)])

            axs.set_xlabel('Record point')
            axs.set_ylabel('Relative\n divergence')

            fig.set_size_inches(4, 3, forward=True)
            plt.tight_layout()

            plt.show()



@main.command()
def compare_hard():
    """
    generate benchmark curves (hardcoded)
    """


    cores = [1, 2, 4, 8, 16]
    s_times = [[], [], []]
    g_times = [[], [], []]

    for simulator in ['g', 's']:
        for n_cores in cores:
            for i, size in enumerate([1, 10, 100]):
                ss = str(size) + 'k'

                print('analyzing', '-'.join(['data/bench', str(simulator), ss, str(n_cores)]) + '.out')
                data = split_output_file('-'.join(['data/bench', str(simulator), ss, str(n_cores)]) + '.out')
                if simulator == 'g':
                    g_times[i].append(data['FOOTER']['total_time'])
                if simulator == 's':
                    s_times[i].append(data['FOOTER']['total_time'])

    # absolute time plot
    fig, axs = plt.subplots(3)

    for i in range(3):
        axs[i].plot(cores, s_times[i], label='sequential')
        axs[i].plot(cores, g_times[i], label='GPU')
        # axs[i].set_yticks([])
        axs[i].set_xticks(cores)
        axs[i].set_title(str([1, 10, 100][i]) + 'k population size')
        axs[i].ticklabel_format(style='sci', scilimits=(-128, 128))

    axs[0].legend()
    axs[1].set_ylabel('Total simulation time [ms]')
    axs[2].set_xlabel('Number of cores')

    fig.set_size_inches(4, 6, forward=True)
    plt.tight_layout()

    plt.show()

    # speedup plot
    fig, axs = plt.subplots(3)

    for i in range(3):
        axs[i].plot(cores, [s_times[i][0] / x / k for x, k in zip(s_times[i], cores)], label='sequential')
        axs[i].plot(cores, [g_times[i][0] / x / k for x, k in zip(g_times[i], cores)], label='GPU')
        # axs[i].plot([0, 16], [0, 16], label='theoretical')
        # axs[i].set_yticks([])
        axs[i].set_xticks(cores)
        axs[i].set_title(str([1.4, 14, 140][i]) + 'k population size')
        axs[i].ticklabel_format(style='sci', scilimits=(-128, 128))

    axs[0].legend()
    axs[1].set_ylabel('Relative speedup')
    axs[2].set_xlabel('Number of cores')

    fig.set_size_inches(4, 6, forward=True)
    plt.tight_layout()

    plt.show()

    # scaling plot
    fig, axs = plt.subplots(1)

    axs.plot([1400, 14000, 140000], [s_times[i][0] for i in range(3)], label='seqential')
    axs.plot([1400, 14000, 140000], [g_times[i][0] for i in range(3)], label='GPU')
    axs.set_xticks(cores)
    axs.set_title(str([1.4, 14, 140][i]) + 'k population size')
    axs.ticklabel_format(style='sci', scilimits=(-128, 128))

    axs.legend()
    axs.set_ylabel('Relative speedup')
    axs.set_xlabel('Number of cores')

    fig.set_size_inches(4, 2, forward=True)
    plt.tight_layout()

    plt.show()


@main.command()
def compare_hard_noprint():
    """
    generate benchmark curves (hardcoded)
    """


    cores = [1, 2, 4, 8, 16]
    s_times = []

    for n_cores in cores:
        print('analyzing', '-'.join(['data/bench', 's', 'np', str(n_cores)]) + '.out')
        data = split_output_file('-'.join(['data/bench', 's', 'np', str(n_cores)]) + '.out')
        s_times.append(data['FOOTER']['total_time'])

    # speedup plot
    fig, axs = plt.subplots()

    axs.plot(cores, [s_times[0] / x / k for x, k in zip(s_times, cores)], label='sequential')
    axs.set_xticks(cores)
    axs.ticklabel_format(style='sci', scilimits=(-128, 128))

    axs.set_ylabel('Relative speedup')
    axs.set_xlabel('Number of cores')
    axs.set_ylim(0, 1.1)

    fig.set_size_inches(4, 2, forward=True)
    plt.tight_layout()

    plt.show()


if __name__ == '__main__':
    main()
