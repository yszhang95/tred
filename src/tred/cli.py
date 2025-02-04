#!/usr/bin/env python

import click

from .util import setup_logging, debug, info

cmddef = dict(context_settings = dict(auto_envvar_prefix='TRED',
                                      help_option_names=['-h', '--help']))

@click.group()
@click.option("-c", "--config", default=None, type=str,
              help="Specify a config file")
@click.option("-l","--log-output", multiple=True,
              help="log to a file [default:stdout]")
@click.option("-L","--log-level", default="info",
              help="set logging level [default:info]")
@click.version_option()
@click.pass_context
def cli(ctx, config, log_output, log_level):
    '''
    Command line interface to the tred simulation.
    '''
    setup_logging(log_output, log_level)
    # ... make context object


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
        
    info(output)


@cli.command('dummy')
@click.option('-o','--output',default=None)
@click.option('-r','--response',default=None)
def dummy(output, response):
    '''
    This command does not exist.
    '''
    from .web import download
    from .response import ndlarsim

    # fixme: eventually need to abstract away different response formats/locations.
    if not response:
        response = 'https://www.phy.bnl.gov/~bviren/tmp/tred/response_38_v2b_50ns_ndlar.npy'
    fname = download(response)
    debug(f'Loading {fname} from {response}')
    r = ndlarsim(fname)
    print(f'response of shape {r.shape}')
    
