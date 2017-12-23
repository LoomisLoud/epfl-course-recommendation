from enrolment_matrix import load_enrolment_matrix, load_db_data
import numpy as np
import pandas as pd

def course_id_mapper(sub_id):
    """
    Maps a subjectID to the corresponding course
    """
    subject_mapping = load_db_data(sub_map_from_pickle=True)
    mapped = subject_mapping[subject_mapping.SubjectID == sub_id].SubjectName
    return mapped.values[0] if not mapped.empty else np.nan

def correlation_series_mean(f_corr, s_corr):
    """
    Returns the mean correlation inbetween the two,
    if one of the correlations is non-existent, return the other
    """
    if f_corr == -5 and s_corr == -5: raise Exception("both correlations non-existent")
    if f_corr == -5 or s_corr == -5: return max(f_corr, s_corr)
    return np.mean([f_corr, s_corr])

def load_grade_corr_matrix(from_pickle=False):
    """
    Returns the matrix of grade correlations inbetween courses
    """
    if from_pickle:
        return pd.read_pickle('../data/grade_correlation_matrix.pkl')
    # Retrieve courses correlations
    grade_corr = pd.read_csv('../data/correlation-subject-pair.csv')
    grade_corr = grade_corr[['sub1', 'sub2', "cor1", "cor2"]]
    grade_corr['cor_mean'] = grade_corr[['cor1', 'cor2']].apply(lambda x: correlation_series_mean(x[0],x[1]), axis=1)
    grade_corr = grade_corr[['sub1', 'sub2', 'cor_mean']]

    # Use SubjectName instead of SubjectID
    grade_corr['sub1_name'] = grade_corr.sub1.map(course_id_mapper)
    grade_corr['sub2_name'] = grade_corr.sub2.map(course_id_mapper)
    grade_corr = grade_corr.dropna()[['sub1_name', 'sub2_name', 'cor_mean']]

    # In case there are no correlations, we set to the mean of all of them
    mean_correlations = grade_corr.mean()

    # Let's make it a matrix
    grade_corr_matrix = grade_corr.set_index(["sub1_name", "sub2_name"]).unstack(level=0).fillna(mean_correlations)
    # normalize correlations by adding 1 and dividing by the max
    grade_corr_matrix = (grade_corr_matrix + 1)/2

    # Set not found courses correlations to the mean of all correlations
    no_corr_courses = [ c for c in load_enrolment_matrix(from_pickle=True).columns.tolist() if c not in grade_corr_matrix.index.tolist() ]
    missing_correlations = pd.DataFrame(np.full(fill_value=mean_correlations,
                                                shape=(grade_corr_matrix.shape[0], len(no_corr_courses))),
                                        columns=no_corr_courses,
                                        index=grade_corr_matrix.index.tolist())
    grade_corr_matrix.columns = grade_corr_matrix.columns.droplevel()
    grade_corr_matrix = pd.concat([grade_corr_matrix, missing_correlations], axis=1)

# Let's transform it into probabilistic
    grade_corr_matrix = grade_corr_matrix / grade_corr_matrix.sum(axis=0)

    grade_corr_matrix.to_pickle('../data/grade_correlation_matrix.pkl')
    return grade_corr_matrix

def get_grades_corr(course, other_enrolments):
    """
    Returns the grade correlations inbetween the
    given course and the other enrolments
    """
    grade_corr_matrix = load_grade_corr_matrix(from_pickle=True)
    if course not in grade_corr_matrix.index.tolist():
        return 1/grade_corr_matrix.shape[1]
    return grade_corr_matrix.loc[course, other_enrolments].mean()

def training_weight_grade_corr(user_index, unit_name="Informatique"):
    """
    Returns the grade correlation weights for a certain user
    """
    courses_matrix = load_enrolment_matrix(unit_name, from_pickle=True)
    courses_taken = courses_matrix.iloc[user_index][courses_matrix.iloc[user_index] == 1].index.tolist()
    return [ get_grades_corr(c, courses_taken) for c in courses_matrix.columns.tolist() ]
