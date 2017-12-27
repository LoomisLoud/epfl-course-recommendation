from enrolment_matrix import UNITS, store_enrolment_matrix, get_last_year_registrations, DATA_FOLDER
from co_enrolment_matrix import store_co_enrolment_matrix
from train import train_all_individual_models
import os.path

def store_enrolment_matrices(verbose=False):
    """
    This function will store all of the enrolment matrices,
    and subject mappings, one for each unit name.
    """
    if not verbose:
        print("Storing enrolment matrices: 0.00%", end="\r")
    for i, unit in enumerate(UNITS.keys()):
      store_enrolment_matrix(unit, verbose=verbose)
      if not verbose:
          print("Storing enrolment matrices: {:.2f}%".format(100*(i+1)/len(UNITS.keys())), end="\r")
    print()

def store_co_enrolment_matrices(verbose=False):
    """
    Stores the co enrolment matrices, stores the enrolment matrices
    if they're not already on disk.
    """
    if not verbose:
        print("Storing co enrolment matrices: 0.00%", end="\r")
    for i, unit in enumerate(UNITS.keys()):
        if not os.path.isfile(DATA_FOLDER + '{}_enrolment_matrix.pkl'.format(UNITS[unit])):
            print("\nMissing the {} enrolment_matrix pickle, loading it\n".format(unit))
            store_enrolment_matrix(unit, verbose=verbose)
        store_co_enrolment_matrix(unit, verbose=verbose)
        if not verbose:
            print("Storing co enrolment matrices: {:.2f}%".format(100*(i+1)/len(UNITS.keys())), end="\r")
    print()

def store_last_year_mapping(verbose=False):
    """
    Stores the last year mapping of courses
    """
    if not verbose:
        print("Storing the last year's mapping: 0.00%", end="\r")
    for i, unit in enumerate(UNITS.keys()):
        get_last_year_registrations(unit, verbose=verbose)
        if not verbose:
            print("Storing the last year's mapping: {:.2f}%".format(100*(i+1)/len(UNITS.keys())), end="\r")
    print()
