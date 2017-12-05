from enrolment_matrix import load_enrolment_matrix
import pandas as pd

def load_co_enrolment_matrix(from_pickle=False):
  if from_pickle:
    return pd.read_pickle('../data/co_enrolment_matrix.pkl')

  courses_matrix = load_enrolment_matrix(from_pickle=True)
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

def store_co_enrolment_matrix():
  load_co_enrolment_matrix().to_pickle('../data/co_enrolment_matrix.pkl')

def get_coenrolment(course, other_enrolments):
  co_enrolments = load_co_enrolment_matrix(from_pickle=True)
  return co_enrolments.loc[course, other_enrolments].mean()

def training_weight_coenrolments(user_index):
  courses_matrix = load_enrolment_matrix(from_pickle=True)
  courses_taken = courses_matrix.iloc[user_index][courses_matrix.iloc[user_index] == 1].index.tolist()
  return [ get_coenrolment(c, courses_taken) for c in courses_matrix.columns.tolist() ]
