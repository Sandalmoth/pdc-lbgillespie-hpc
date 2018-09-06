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


if __name__ == '__main__':
    main()
