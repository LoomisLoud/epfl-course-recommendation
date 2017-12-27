from enrolment_matrix import UNITS, load_enrolment_matrix, DATA_FOLDER
from keras.layers import Input, Dense, Embedding, Flatten, Dropout, Activation
from keras.layers.merge import Add
from keras.models import Model, load_model
from keras.regularizers import l2
import numpy as np

def create_model(I, U, K, hidden_activation, output_activation, q=0.5, l=0.01):
    '''
    Reference:
      Yao Wu, Christopher DuBois, Alice X. Zheng, Martin Ester.
        Collaborative Denoising Auto-Encoders for Top-N Recommender Systems.
          The 9th ACM International Conference on Web Search and Data Mining (WSDM'16), p153--162, 2016.

    :param I: number of items
    :param U: number of users
    :param K: number of units in hidden layer
    :param hidden_activation: activation function of hidden layer
    :param output_activation: activation function of output layer
    :param q: drop probability
    :param l: regularization parameter of L2 regularization
    :return: CDAE
    :rtype: keras.models.Model
    '''
    x_item = Input((I,), name='x_item')
    h_item = Dropout(q)(x_item)
    h_item = Dense(K, kernel_regularizer=l2(l), bias_regularizer=l2(l))(h_item)

    # dtype should be int to connect to Embedding layer
    x_user = Input((1,), dtype='int32', name='x_user')
    h_user = Embedding(input_dim=U, output_dim=K, input_length=1, embeddings_regularizer=l2(l))(x_user)
    h_user = Flatten()(h_user)

#    h = merge([h_item, h_user], mode='sum')
    h = Add()([h_item, h_user])
    if hidden_activation:
        h = Activation(hidden_activation)(h)
    y = Dense(I, activation=output_activation)(h)

    return Model(inputs=[x_item, x_user], outputs=y)

def train_model(data, dropout=0.998, hidden_layers=27, verbosity=2, save=None):
    """
    Trains the model on the specified data, using dropout, the number of
    hidden layers, and fill save with the unit name
    """
    training_set, testing_set, users = split_data(data)
    print("train: {}, users: {}".format(training_set.shape, users.shape))
    model = create_model(I=training_set.shape[1], U=len(users)+1, K=hidden_layers,
                         hidden_activation='relu', output_activation='sigmoid',
                         q=dropout, l=0.01)
    model.compile(loss='mean_absolute_error', optimizer='adam')
    model.fit(x=[training_set, users], y=training_set,
              batch_size=128, epochs=2000, verbose=verbosity,
              validation_split=0.20)
    if save:
        model.save(DATA_FOLDER + '{}_cdae_model.hd5'.format(UNITS[save]))
    return model

def split_data(data):
    """
    Splits the given data (pandas) in a train set, test set and the
    labels for users
    """
    testing_set = data.applymap(lambda x: 0)

    taken_courses_flat = data.stack().to_frame()
    taken_courses_flat = taken_courses_flat[taken_courses_flat[0] == 1]

    for student in taken_courses_flat.index.get_level_values('PersonID').unique():
        courses = taken_courses_flat.loc[student]
        for course in courses.sample(frac=0.2, replace=False).index:
            testing_set.loc[student, course] = 1
    training_set = data - testing_set

    # Numpifies the data
    train_np = training_set.apply(axis=1, func=lambda x: x.astype(int)).as_matrix()
    test_np = testing_set.apply(axis=1, func=lambda x: x.astype(int)).as_matrix()

    # the indices of each user
    users = np.array(np.arange(data.shape[0])[np.newaxis].T, dtype=np.int32)

    return train_np, test_np, users

def train_all_individual_models(dropout=0.998, hidden_layers=27, verbosity=2):
    """
    The aim of this method is to simply train all the models in order
    to store them on disk afterwards for dynamic loading.
    """
    for i, unit in enumerate(UNITS):
        print("Training the model for {} ({}/{})".format(unit, i+1, len(UNITS)))
        train_model(load_enrolment_matrix(unit, from_pickle=True), dropout, hidden_layers, verbosity, save=unit)

def load_trained_model(unit):
    """
    Loads the trained model for the given unit name
    """
    return load_model(DATA_FOLDER + "{}_cdae_model.hd5".format(UNITS[unit]))
