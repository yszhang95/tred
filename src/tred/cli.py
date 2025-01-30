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
@click.option('-o','--output',default=None)
@click.argument("categories", nargs=-1)
def plots(output, categories):
    '''
    Make diagnostic plots.

    Plot categories can be given as arguments.  Default is to use all that are available.
    '''
    import tred.plots
    from tred.plots.util import pages

    all_categories = [m for m in dir(tred.plots) if hasattr(getattr(tred.plots, m), "plots")]
    
    if not categories:
        categories = all_categories
    categories = [c for c in categories if c in all_categories]

    if output is None:
        scat = '-'.join(categories)
        if scat:
            scat = '-' + scat
        output = f"tred-diagnostics{scat}.pdf"

    with pages(output) as out:
        for category in categories:
            mod = getattr(tred.plots, category)
            mod.plots(out)
        
    print(output)


@cli.command('dummy')
@click.option('-o','--output',default=None)
@click.option('-r','--response',default=None)
def dummy(output, response):
    '''
    This command does not exist.

    Temporary harness to drive S*R step 
    '''
    import tred.ndlar
    r = tred.ndlar.response(response)
    
    
