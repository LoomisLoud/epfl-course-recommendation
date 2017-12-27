"""
Populates the disk with all the data necessary
to recommend courses
"""
from grade_correlations import load_grade_corr_matrix
import populate

VERBOSITY = False

print("Let's load all the data needed to recommend courses, the enrolment "
+ "matrices, the co-enrolment matrices, last year's mapping, the grade "
+ "correlations matrices, and training the models data.")

populate.store_enrolment_matrices(verbose=VERBOSITY)
populate.store_co_enrolment_matrices(verbose=VERBOSITY)
populate.store_last_year_mapping(verbose=VERBOSITY)
load_grade_corr_matrix(from_pickle=False)
populate.train_all_individual_models(verbosity=0)
