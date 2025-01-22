try:
    from urllib import request
except ImportError:
    from urllib import urlopen
else:
    urlopen = request.urlopen
    from urllib.parse import urlparse
from pathlib import Path
import ssl
# www.phy.bnl.gov cert is in a format that is legit but not correctly parsed by python
ssl_context = ssl._create_unverified_context()

def download(url, target=None, force=False):
    '''
    Download content at URL to target file name.

    If no target is given, guess a file name from the URL.

    If named file exists, do not download unless force is True.

    Return the target filename as Path.
    '''
    web = urlopen(url, context=ssl_context)
    if web.getcode() != 200:
        raise IOError(f'failed to download {url} got error {web}')
    if target is None:
        p = Path(urlparse(url).path)
        target = p.name
    target = Path(target)
    if target.exists() and not force:
        return target
    target.write_bytes(web.read())
    return target

