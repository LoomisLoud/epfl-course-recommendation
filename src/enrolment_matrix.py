import configparser
import mysql.connector as sql
import numpy as np
import pandas as pd

DATA_FOLDER = "data/"
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
    "Chemical engineering of heterogenous reactions",
    "Process development I",
    "Chemical engineering lab & project",
    "Stage d'ingénieur (master en Génie chimique et Biotechnologie)",
    "Projet de master en génie chimique et biotechnologie",
    "Interdisciplinary project",
    "Projet de master en chimie moléculaire et biologique",
    "Project in molecular sciences",
    "Superstudio",
    "Enoncé théorique de master",
    "De la structure à l'ornement",
    "Projet de master en architecture",
    "Pré-étude projet de master",
    "Projet SIE/ENAC",
    "Projet de master en sciences et ingénierie de l'environnement",
    "Stage d'ingénieur crédité avec le PDM (master en Sciences et ingénierie de l'environnement)",
    "Projet de master en génie électrique et électronique",
    "Projet Génie mécanique II",
    "Projet Génie mécanique I",
    "Stage d'ingénieur crédité avec le PDM (master en Génie mécanique)",
    "Projet de master en génie mécanique",
    "Research project in materials I",
    "Projet de master en science et génie des matériaux",
    "Stage d'ingénieur crédité avec le PDM (master en Science et génie des matériaux)",
    "Projet microtechnique I",
    "Projet de master en microtechnique",
    "Stage d'ingénieur crédité avec le PDM (master en Microtechnique)",
    "Projet de master en mathématiques",
    "Projet de Mathématiques (master)",
    "Stage d'ingénieur (master en Ingénierie mathématique)",
    "Projet de master en mathématiques",
    "Projet de Mathématiques (master)",
    "Stage d'ingénieur crédité avec le PDM (master en Ingénierie mathématique)",
    "Stage d'ingénieur (master en Bioingénierie)",
    "Projet de master en bioingénierie et biotechnologie",
    "Stage d'ingénieur (master en Sciences et technologie du vivant)",
    "Projet de master en sciences et technologies du vivant",
    "Stage d'ingénieur (master en Génie nucléaire)",
    "Projet de master en génie nucléaire",
    "Stage d'ingénieur (master en Ingénierie physique)",
    "Projet de master en physique",
    "Stage d'ingénieur (master en Sciences et ingénierie computationnelles)",
    "Projet de master en science et ingénierie computationelles",
    "Projet CSE I",
    "Projet CSE II",
    "Project in energy management and sustainability I",
    "Stage d'ingénieur crédité avec le PDM (master en Gestion de l'énergie et construction durable)",
    "Stage d'ingénieur (master en Génie électrique et électronique)",
]

DOMAINS_TO_REMOVE = [
    "Humanities and social sciences",
    "Programme Sciences humaines et sociales",
]

UNITS = {
    'Architecture': 'AR',
    'Bioingénierie': 'SV_B',
    'Chimie moléculaire et biologique': 'CGC_CHIM',
    "Gestion de l'énergie et construction durable": 'EME_MES',
    'Génie chimique et biotechnologie': 'CGC_ING',
    'Génie civil': 'GC',
    'Génie mécanique': 'GM',
    'Génie nucléaire': 'PH_NE',
    'Génie électrique et électronique': 'EL',
    'Informatique': 'IN',
    'Ingénierie financière': 'IF',
    'Ingénierie mathématique': 'ING_MATH',
    'Ingénierie physique': 'ING_PHYS',
    'Management, technologie et entrepreneuriat': 'MTEE',
    'Mathématiques - master': 'MATH',
    'Micro and Nanotechnologies for Integrated Systems': 'MNIS',
    'Microtechnique': 'MT',
    'Physique - master': 'PHYS',
    'Science et génie des matériaux': 'MX',
    'Science et ingénierie computationnelles': 'MA_CO',
    "Sciences et ingénierie de l'environnement": 'SIE',
    'Sciences et technologies du vivant - master': 'SV_STV',
    'Systèmes de communication - master': 'SC_EPFL'
}

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
                UnitName = "{}"
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


def load_db_data(unit_name='Informatique', sub_map_from_pickle=False):
    """
    Loads data from the database, returns only the subject mappings if specified
    """
    if sub_map_from_pickle:
        return pd.read_pickle(DATA_FOLDER + '{}_subject_mapping.pkl'.format(UNITS[unit_name]))
    db_connection = init_connection()
    all_df = pd.read_sql(ALL_INFO.format(unit_name), con=db_connection)
    # Removing the useless courses, and SHS ones
    all_df = all_df[~all_df.SubjectName.isin(COURSES_TO_REMOVE)]
    all_df = all_df[~(all_df.StudyDomain.isin(DOMAINS_TO_REMOVE))]
    # Mapping of subject ids to subject names
    subject_mapping = all_df[['SubjectID', 'SubjectName']].drop_duplicates()
    # Saving the subject mappings to a pkl
    subject_mapping.to_pickle(DATA_FOLDER + '{}_subject_mapping.pkl'.format(UNITS[unit_name]))
    return all_df, subject_mapping


def load_enrolment_matrix(unit_name='Informatique', from_pickle=False, verbose=False):
    """
    Loading the enrolment matrix from the unit
    """
    if verbose:
        print("Loading the {} enrolment matrix".format(unit_name))
    if from_pickle:
        return pd.read_pickle(DATA_FOLDER + '{}_enrolment_matrix.pkl'.format(UNITS[unit_name]))

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
    courses_matrix = courses_matrix[np.sum(courses_matrix == 1, axis=1) > MIN_COURSES_BY_STUDENT]
    return courses_matrix


def series_to_integers(series):
    """
    Converts a whole series to integers
    """
    return pd.to_numeric(series, downcast='integer')


def store_enrolment_matrix(unit_name='Informatique', verbose=False):
    """
    Stores the enrolment matrix
    """
    load_enrolment_matrix(unit_name, verbose=verbose).to_pickle(DATA_FOLDER + "{}_enrolment_matrix.pkl".format(UNITS[unit_name]))
    if verbose:
        print("Stored {} enrolment matrix".format(unit_name))


def get_last_year_registrations(unit_name='Informatique', from_pickle=False, verbose=False):
    """
    Returns the last year registrations of the given course
    """
    if verbose:
        print("Loading the {} last year registrations".format(unit_name))
    if from_pickle:
        return pd.read_pickle(DATA_FOLDER + '{}_last_year_registrations.pkl'.format(UNITS[unit_name]))
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
    registrations_final.to_pickle(DATA_FOLDER + '{}_last_year_registrations.pkl'.format(UNITS[unit_name]))
    return registrations_final
