from enrolment_matrix import load_enrolment_matrix, UNITS, DATA_FOLDER
import pandas as pd

def load_co_enrolment_matrix(unit_name="Informatique", from_pickle=False, verbose=False):
    """
    Loads the co-enrolment matrix from disk or the database
    """
    if verbose:
        print("Loading the {} co enrolment matrix".format(unit_name))
    if from_pickle:
        return pd.read_pickle(DATA_FOLDER + '{}_co_enrolment_matrix.pkl'.format(UNITS[unit_name]))

    courses_matrix = load_enrolment_matrix(unit_name, from_pickle=True, verbose=verbose)
    co_enrolments = pd.DataFrame(data=0, columns=courses_matrix.columns, index=courses_matrix.columns)
    for row in courses_matrix.iterrows():
        taken_courses = row[1][row[1] == 1].index.tolist()
        for i,course in enumerate(taken_courses):
            co_enrolments.loc[course, taken_courses[i+1:]] += 1

    # Copy the upper triangle matrix to lower triangle one
    co_enrolments = co_enrolments + co_enrolments.T

    # Transforming to probabilities and removing the rows summing to nan
    co_enrolments = co_enrolments / co_enrolments.sum(axis=0)
    return co_enrolments

def store_co_enrolment_matrix(unit_name="Informatique", verbose=False):
    """
    Stores the co-enrolment to disk
    """
    load_co_enrolment_matrix(unit_name, verbose=verbose).to_pickle(DATA_FOLDER + '{}_co_enrolment_matrix.pkl'.format(UNITS[unit_name]))
    if verbose:
        print("Stored the {} co enrolment matrix".format(unit_name))

def get_coenrolment(course, other_enrolments, unit_name="Informatique"):
    """
    Returns the co-enrolment from a course
    """
    co_enrolments = load_co_enrolment_matrix(unit_name, from_pickle=True)
    return co_enrolments.loc[course, other_enrolments].mean()

def training_weight_coenrolments(user_index, unit_name="Informatique"):
    """
    Returns the training weights of co-enrolment
    """
    courses_matrix = load_enrolment_matrix(unit_name, from_pickle=True)
    courses_taken = courses_matrix.iloc[user_index][courses_matrix.iloc[user_index] == 1].index.tolist()
    return [ get_coenrolment(c, courses_taken, unit_name) for c in courses_matrix.columns.tolist() ]
