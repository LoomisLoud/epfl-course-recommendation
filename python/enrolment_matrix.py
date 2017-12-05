import configparser
import mysql.connector as sql
import numpy as np
import pandas as pd

# Number of minimum courses a student has to have
# taken to be considered in the recommender system
MIN_COURSES_BY_STUDENT = 10

# Found courses that should be removed:
COURSES_TO_REMOVE = [
    "Admission année sup.",
    "Projet de master en systèmes de communication",
    "SHS : Introduction au projet",
    "Cycle master",
    "Projet de Master",
    "Groupe Core courses & options",
    "Bloc Projets et SHS",
    "Groupe 2 : Options",
    "Master SC",
    "Mineur",
    "Groupe 1",
    "Projet en systèmes de communication II",
    "Projet en informatique II",
    "Projet de master en informatique",
    "Cours réservés spécifiquement aux étudiants s'inscrivant pour le mineur Area and Cultural Studies",
    "SHS : Projet",
    "Optional project in communication systems",
    "Optional project in computer science",
    "Mineur : Neurosciences computationnelles",
    "Stage d'ingénieur crédité avec le PDM (master en Systèmes de communication)",
    "Stage d'ingénieur crédité avec le PDM (master en Informatique)",
    "Cours UNIL - Faculté des hautes études commerciales HEC I (printemps)",
]

DOMAINS_TO_REMOVE = [
    "Humanities and social sciences",
    "Programme Sciences humaines et sociales",
]

ALL_INFO = """
            select distinct
                PersonID,
                PedagogicalCode,
                StudyDomain,
                UnitName,
                UnitID,
                UnitCode,
                SubjectName,
                Course_Enrolments.SubjectID,
                SectionName,
                CourseCode,
                YearName
            from
                Course_Enrolments
                inner join
                Course_Codes
                    on Course_Codes.planid = course_enrolments.planid
                    and Course_Codes.subjectid = course_enrolments.subjectid
            where
                {}
                and LevelName = "Master"
            """

def init_connection():
  """
  Read the confidential token.
  """
  credentials = configparser.ConfigParser()
  credentials.read('../config/credentials.ini')
  db_connection = sql.connect(host=credentials.get('mysql', 'url'),
                              database='semester_project_romain',
                              user=credentials.get('mysql', 'username'),
                              password=credentials.get('mysql', 'password'))
  return db_connection

def load_db_data(unit_name):
  db_connection = init_connection()
  all_df = pd.read_sql(ALL_INFO.format(unit_name), con=db_connection)
  # Removing the useless courses, and SHS ones
  all_df = all_df[~all_df.SubjectName.isin(COURSES_TO_REMOVE)]
  all_df = all_df[~(all_df.StudyDomain.isin(DOMAINS_TO_REMOVE))]
  # Mapping of subject ids to subject names
  subject_mapping = all_df[['SubjectID', 'SubjectName']].drop_duplicates()
  return all_df, subject_mapping

def load_enrolment_matrix(unit_name='(UnitName like "%nform%" or UnitName like "%omm%")', from_pickle=False):
  """
  Loading the enrolment matrix
  """
  if from_pickle:
    return pd.read_pickle('../data/enrolment_matrix.pkl')

  all_df, _ = load_db_data(unit_name)
  courses_matrix = all_df[['PersonID', 'SubjectName']]
  courses_matrix = courses_matrix.drop_duplicates()
  courses_matrix = courses_matrix.set_index(['PersonID', 'SubjectName'])

  # If the course was taken, set it to 1
  courses_matrix['joined'] = 1
  courses_matrix = courses_matrix.reset_index().pivot(index='PersonID', columns='SubjectName', values='joined')
  courses_matrix = courses_matrix.fillna(0)
  courses_matrix = courses_matrix.apply(series_to_integers)

  # Removing all students that took less than MIN_COURSES_BY_STUDENT courses
  courses_matrix =courses_matrix[np.sum(courses_matrix == 1, axis=1) > MIN_COURSES_BY_STUDENT]
  return courses_matrix

def series_to_integers(series):
  "Converts a whole series to integers"
  return pd.to_numeric(series, downcast='integer')

def store_enrolment_matrix(unit_name='(UnitName like "%nform%" or UnitName like "%omm%")'):
  load_enrolment_matrix(unit_name).to_pickle("../data/enrolment_matrix.pkl")

def get_last_year_registrations(unit_name='(UnitName like "%nform%" or UnitName like "%omm%")', from_pickle=False):
  if from_pickle:
    return pd.read_pickle('../data/last_year_registrations.pkl')
  all_df, _ = load_db_data(unit_name)
  registrations_df = all_df.set_index(['SubjectName', 'YearName'])
  all_df_registrations = registrations_df.groupby(['SubjectName', 'YearName']).size()

  registrations_df['Registration'] = all_df_registrations
  registrations_df = registrations_df.reset_index()
# Pick only courses that have a study domain (removes bullshit)
# such as Projects and groups, minors etc
  registrations_df = registrations_df[~registrations_df.StudyDomain.isnull()]
# Removes non important information
  registrations_df = registrations_df.drop([
      'PersonID', "StudyDomain", "SectionName", "PedagogicalCode",
      "CourseCode"], axis=1)
  registrations_df = registrations_df.drop_duplicates()
  registrations_df = registrations_df.set_index(['SubjectName', 'YearName']).sort_index()
  registrations = registrations_df.sort_values(ascending=False, by='Registration')

# Latest data registrations
  registrations_final = registrations.xs('2015-2016', level='YearName')
  registrations_final.to_pickle('../data/last_year_registrations.pkl')
  return registrations_final
