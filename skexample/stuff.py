#
# A sample file
#

__all__ = ['do_something']

def do_something(x):
    """
    Routine that does something

    Parameters
    ----------
    x : object
        Some input parameter
    
    Returns
    -------
    y : int
        Some relevant integer

    Notes
    -----
    This routine actually doesn't do much.

    """
    print("something: %r" % x)
    return 123

