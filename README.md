# EPFL Courses recommendation system

## Install
  To install the dependencies, download the data and train the models, set up the EPFL VPN, install python3.6 and run:
  ```shell
  pip3.6 install -r requirements.txt
  python3.6 src/init.py
  ```

## Use
  To recommend a list of courses for a user, run `recommend_course.predict(unit, courses)`, unit being the unit the student is in, and courses being the list of courses he already took in masters.
  
  For a graphical usage, run flask on the application.py file: `FLASK_APP=application.py flask run`.

## Research
  You can find the research in the Jupyter Notebooks provided, the models tried in the Exploration.ipynb notebook and the best model found in the Recommender.ipynb notebook. Feel free to read the technical report to learn more about the research !

## Model
  There is one trained model per unit of the school, each trained on the subset of courses in the unit. It's composed of three parts, all the same for each unit, so we won't distinguish after this point.

  The first part is the grade correlations inbetween courses, we create a matrix of such correlations and use it to create a vector of interest of grade correlations between all the courses of the user using the recommender system. The second part is pretty much the same, but for courses co-enrolment. The idea is that if multiple people have chosen to take a similar list of courses, then the courses might have a link to one another.
  The remaining part is a neural network, in fact a Collaborative Denoising Auto-Encoders network from this [paper](http://alicezheng.org/papers/wsdm16-cdae.pdf) which returns a confidence by course by user.
  We multiply the results of all three parts which gives us a confidence score for each course, the higher the better.

  Using these, we achieve on the worst case a top-5 success-rate of 70% and best case of 90%

## Credits
Huge thanks to the scientists who worked on the [Collaborative Denoising Auto-Encoders
for Top-N Recommender Systems](http://alicezheng.org/papers/wsdm16-cdae.pdf), as well as the [henry0312](https://github.com/henry0312/CDAE) for his quick implementation of the paper.

Thanks also to Francisco Pinto and Kshitij Sharma at the CEDE lab at EPFL for the great help when supervising this project.
