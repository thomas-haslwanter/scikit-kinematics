'''
Miscellaneous user interface utilities for

    - getting the screen size
    - selecting files or directories.
      If nothing or a non-existing file/direcoty is selected, the return is "0". 
      Otherwise the file/directory is returned.
    - Selection from a list.
    - waitbar
    - listbox

'''

'''
ThH, April 2016
Ver 1.0
'''

import matplotlib.pyplot as plt
import os
import sys

if sys.version_info.major == 3:
    # Python 3.x
    import tkinter as tk
    import tkinter.filedialog as tkf
else:
    # Python 2.x
    import Tkinter as tk
    import tkFileDialog as tkf
    

def get_screensize():
    '''
    Get the height and width of the screen. 
    
    Parameters
    ----------
        None
    
    Returns
    -------
    width :  int
        width of the current screen
    height:  int
        height of the current screen
    
    Examples
    --------
    >>> (width, height) = thLib.ui.get_screensize()
    '''
    
    
    try:
        # Use the methods form PyQt first, since tk gave me some strange error messages sometimes
        from PyQt4 import QtGui
        import sys
        
        MyApp = QtGui.QApplication(sys.argv)
        V = MyApp.desktop().screenGeometry()
        screen_h = V.height()
        screen_w = V.width()    
    
    except ImportError:
        # If PyQt4 is not available
        root = tk.Tk()
        (screen_w, screen_h) = (root.winfo_screenwidth(), root.winfo_screenheight())
        root.destroy()
    
    return (screen_w, screen_h)
    
def progressbar(it, prefix = "", size = 60):
    '''
    Shows a progress-bar on the commandline.
    This has the advantage that you don't need to bother with windows
    managers. Nifty coding!
    
    Parameters
    ----------
    it : integer array
        index variable
    prefix : string
        Text preceding the progress-bar
    size : integer
        Length of progress-bar

    Examples
    --------
    >>> import time
    >>> for ii in progressbar(range(50), 'Computing ', 25):
    >>>    #print(ii)
    >>>    time.sleep(0.05)
    
    '''

    count = len(it)
    def _show(_i):
        # Helper function to print the desired information line.

        x = int(size*_i/count)
        sys.stdout.write("%s[%s%s] %i/%i\r" % (prefix, "#"*x, "."*(size-x), _i, count))
#        sys.stdout.flush()
    
    _show(0)
    for i, item in enumerate(it):
        yield item
        _show(i+1)
    sys.stdout.write("\n")
    sys.stdout.flush()
    
if __name__ == "__main__":   
    # Test functions
    
    width, height = get_screensize()
    print('Your screen is {0} x {1} pixels.'.format(width, height))
    
    '''
    import time
    for ii in progressbar(range(50), 'Computing ', 25):
        #print(ii)
        time.sleep(0.05)
        

    (myFile, myPath) = getfile('*.eps', 'Testing file-selection', r'c:\temp\test.eps')
    if myFile == 0:          
        print(0)
    else:
        print('File: %s, Path: %s' % (myFile, myPath))
    (myFile, myPath) = savefile('*.txt', 'Testing saving-selection', r'c:\temp\test.txt')
        
    myDir = getdir()
    print(myDir)

    
    items = ['Peter', 'Paul', 'Mary']    
    selected = listbox(items*4)
    if selected == '':
        print('No selection made.')
    else:
        print('You have selected {0}'.format(selected))
    
    import numpy as np
    import pandas as pd
    x = np.arange(5)
    y = np.random.randn(5,3)
    s = pd.Series(x)
    df = pd.DataFrame(y)
    z = 'abc'
    
    selVal, selName = selectPlotVar(sys._getframe())
    #selected = selectPlotVar()
    curFrame = sys._getframe()
    varList = curFrame.f_locals.keys()
    ndList = [var for var in varList if type(curFrame.f_locals[var])==np.ndarray]
    dfList = [var for var in varList if type(curFrame.f_locals[var])==pd.core.frame.DataFrame]
    seriesList = [var for var in varList if type(curFrame.f_locals[var])==pd.core.series.Series]
    fullList = ndList+dfList+seriesList
    
    selected = listbox(fullList)
    
    print(selName)
    print(selVal)
    
    root = tk.Tk()
    app = Demo1(root, sys._getframe())
    root.mainloop()

    '''
