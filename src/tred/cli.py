#!/usr/bin/env python

import click


@click.group()
@click.version_option()
def cli():
    '''
    Command line interface to the tred simulation.
    '''
    pass

@cli.command('plots')
@click.option('-o','--output',default="tred-diagnostics.pdf")
@click.argument("categories", nargs=-1)
def plots(output, categories):
    '''
    Make diagnostic plots.

    Plot categories can be given as arguments.  Default is to use all that are available.
    '''
    import tred.plots
    from tred.plots.util import pages

    if not categories:
        categories = [meth.replace("_plots","") for meth in dir(tred.plots) if meth.endswith('_plots')]

    with pages(output) as out:
        for category in categories:
            meth = getattr(tred.plots, f'{category}_plots')
            meth(out)
        
    


@cli.command('dummy')
def dummy():
    '''
    junk
    '''
    click.echo("derp")
    
