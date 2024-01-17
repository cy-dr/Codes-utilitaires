import numpy as np
import matplotlib.pyplot as plt

def binner(Map, Map_err, binning):
    """
    Function that applies a binning to a 2D numpy.array with a weigted mean, it ignores np.nan values if present.

    Map : 2D numpy.array to be binned
    Map_err : 2D numpy.array of uncertainties on Map, also to be binned by a non-weighted mean, can also contain nans
    binning : int value for the wanted binning

    ex : a (100,100) array with binning = 5 will produce a (20,20) output

    output : binnedMap, binnedMap_err
    """
    if binning < 2:
        print('no binning')
        return Map, Map_err
    else:
        a1 = np.arange(Map.shape[0]//binning)*binning
        a2 = np.arange(Map.shape[1]//binning)*binning
        binnedMap = np.zeros((Map.shape[0]//binning,Map.shape[1]//binning))
        binnedMap_err = np.zeros((Map_err.shape[0]//binning,Map_err.shape[1]//binning))
        for i in a1: 
            for j in a2:                                 #moyenne pondérée
                binnedMap[i//binning,j//binning] = (np.nansum(Map[i:i+binning,j:j+binning]/
                                                             (Map_err[i:i+binning,j:j+binning])**2)/
                                                    np.nansum(1/(Map_err[i:i+binning,j:j+binning])**2)) 
                binnedMap_err[i//binning,j//binning] = np.sqrt(np.nansum(((1/Map_err[i:i+binning,j:j+binning])/
                                                                          np.nansum(1/Map_err[i:i+binning,j:j+binning]**2))**2)) 
        
        return binnedMap, binnedMap_err

def reshaper(array, new_shape, binning):
    """
    Function that reshapes a 2D_array to a new 2D_array of bigger size (usually that has been binned)

    array : 2D numpy.array
    new_shape : Tuple corresponding to the wanted array.shape() output
    binning : The binning value that was applied to produce "array"
    """
    if array.shape == new_shape:
        return array
    else:
        b = binning
        new_array = np.zeros(new_shape)
        a1 = np.arange(new_shape[0]//b)*b
        a2 = np.arange(new_shape[1]//b)*b
        for i in a1:
            for j in a2:#Moyenne pondérée
                new_array[i:i+b,j:j+b] = array[i//b,j//b]
        return new_array

def corr(map, map_err, line, RCHaHb, RCHaHbP, RCHaHbM):
    """Function that applies a redenning correction to an emission map of a specific line
    map, map_err: 2D_array
    line : float
    RCHaHb, RCHaHbP, RCHaHbM : RedCorr() objects from PyNeb"""
    mapCorr = map*RCHaHb.getCorr(line)
    mapCorr_err = np.sqrt((map_err*RCHaHb.getCorr(line))**2 + (map*np.abs(RCHaHbP.getCorr(line)-RCHaHbM.getCorr(line))/2)**2)
    return mapCorr, mapCorr_err
