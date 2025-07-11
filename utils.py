import sys
from contextlib import contextmanager
from os import devnull
from tkinter import Tk
from tkinter.filedialog import askdirectory, asksaveasfilename, askopenfilenames, askopenfilename


def tk_popover(save: bool = False, open_dir: bool = False, open_many: bool = False, **kwargs):
    """Tk helper to ensure window appears on top. Default behavior returns a single file location."""
    assert not (save and open_dir)
    assert not (open_dir and open_many)
    assert not (open_many and save)
    root = Tk()
    root.iconify()
    root.attributes('-topmost', True)
    root.update()
    try:
        if not save:
            if open_dir:
                loc = askdirectory(parent=root, **kwargs)
            elif open_many:
                loc = askopenfilenames(parent=root, **kwargs)
            else:
                loc = askopenfilename(parent=root, **kwargs)
        else:
            loc = asksaveasfilename(parent=root, **kwargs)
    finally:
        root.attributes('-topmost', False)
        root.destroy()
        if "loc" not in locals():  # Default return if all above fail
            loc = None
    return loc


@contextmanager
def suppress_stdout():
    """Context manager to suppress printing to stdout by piping into devnull; by Dave Smith:
    https://thesmithfam.org/blog/2012/10/25/temporarily-suppress-console-output-in-python/"""
    with open(devnull, "w") as dn:
        old_stdout = sys.stdout
        sys.stdout = dn
        try:
            yield
        finally:
            sys.stdout = old_stdout


def minmax_norm(x):
    return (x - x.min()) / (x.max() - x.min())
