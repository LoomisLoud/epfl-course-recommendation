# Obligatory to import files from the src folder
import sys
import html
sys.path.insert(0, "src")
from flask import Flask, render_template, request
from recommend_course import predict
from enrolment_matrix import UNITS, load_enrolment_matrix
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def start():
    if request.method == "GET":
        # Initial website
        return render_template("recommender.html", units=UNITS, courses=None, selected_section=None, recommendation=None)
    else:
        if request.get_json():
            # The user wants a recommendation
            data = request.get_json()
            # unescape solves html badly formatted characters
            courses_found = [ html.unescape(course) for course in data['courses'] ]
            section_found = html.unescape(data['section'])
            recommendation = predict(section_found, courses_found)
            return render_template("recommender.html", units=UNITS, courses=None, selected_section=section_found, recommendation=recommendation)
        else:
            # The user selected a section
            section_found = request.form['section']
            found_courses = load_enrolment_matrix(unit_name=section_found, from_pickle=True)
            found_courses = found_courses.columns.tolist()
            found_courses.sort()
            return render_template("recommender.html", units=UNITS, courses=found_courses, selected_section=section_found, recommendation=None)
