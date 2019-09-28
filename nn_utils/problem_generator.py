"""
problem_generator.py

Initial version by R. Mukai 29 April 2018

This module is for generating logic problems to
be solved by a neural network.
"""

import re
import copy
import logging
import numpy as np
from nn_utils.config import CONFIG_SYMBOL_LIST

# We begin by defing a function to generate one-hot
# encodings of key logical symbols.

def gen_one_hot_encoding(num_vars=10):
    """
    gen_one_hot_encoding: Generate a dictionary of allowed
    characters with corresponding one-hot encodings of
    characters used in logic sentences.

    :param num_vars: This is the number of variables, number
    of constants, and number of predictates allowed.

    :return A dictionary whose keys are logical symbols
    and whose values are one-hot vectors encoding these.
    """

    # Create a list of symbols.

    symbol_list_logic = copy.deepcopy(CONFIG_SYMBOL_LIST)

    symbol_list_predicates = []
    symbol_list_variables = []
    symbol_list_constants = []
    symbol_list_names = []

    for index in range(num_vars):

        symbol_list_predicates.append('p'+str(index))
        symbol_list_variables.append('x'+str(index))
        symbol_list_constants.append('a'+str(index))
        symbol_list_names.append('n'+str(index))

    # End symbol generator loop

    symbol_list = symbol_list_logic + symbol_list_predicates + \
    symbol_list_variables + symbol_list_constants + symbol_list_names

    num_symbols = len(symbol_list)

    symbol_dictionary = {}

    for symbol_index in range(num_symbols):

        # Initialize vector to all zeros.

        current_vector = np.zeros((1, num_symbols))

        current_vector[0,symbol_index] = 1.0

        # Use one-hot encoding and write to dictionary

        symbol_dictionary[symbol_list[symbol_index]] = \
        current_vector
    # End one-hot loop

    #symbol_dictionary['symbol_list'] = symbol_list

    return symbol_dictionary

# End function gen_one_hot_encoding

def encode_string(input_string, symbol_dictionary):
    """
    encode_string: Given a string and a symbol dictionary,
    create an encoding as an array of one-hot row vectors.

    :param input_string: String to be encoded, which all symbols
    separated by SPACES
    :param symbol_dictionary: Dictionary mapping symbols to
    one-hot row vectors.

    :return encoded_array: An array with one one-hot row vector
    for each symbol.

    """

    # Split the string using spaces.

    if input_string is not None:

        token_list = input_string.split()

    else:

        token_list = [None]

    # Get number of tokens.

    n_tokens = len(token_list)

    # Obtain length of a dictionary vector.

    vector_len = symbol_dictionary['and'].size

    # Generate an array of zeros to hold the encoded data.

    encoded_array = np.zeros((n_tokens, vector_len))

    # Write to the encoded array.

    for row_index in range(n_tokens):

        try:

            encoded_array[row_index, :] = \
            symbol_dictionary[token_list[row_index]]

        except:

            encoded_array[row_index, :] = \
            symbol_dictionary[None]

    # End the loop to write encoded vectors

    # Return encoded array.

    return encoded_array

# End function encode string.

def permutation_generator(n_vars):
    """
    permutation_generator: Create permutation dictionary to be
    used to instantiate a given logical sentence pattern.

    # Suppose you have the following very
    simple pattern:

    if x1 then x2
    x1 is true

    Then x2 is true

    Now we have specific variables x(n) above.  But
    we wish to have our network learn basic generalization
    by knowing how to do basic substitutions.  A pattern
    should contain placeholders, like this:

    if px1 then px2
    px1 is true
    Therefore px2 is true

    We can do this by creating a permutation that maps
    for example px1 => x7, px2=> x3, resulting an in
    instance:

    if x7 then x3
    x7 is true
    Therefore x3 is true

    This next function is meant to do just that.  Here,
    pp would mean pre-predicate so pp1 might, for a given
    instance, get mapped into p8 while pp2 might be mapped
    into p5.  Likewise, px means pre-x, and a specific x
    gets chosen.

    What is the reason for this?  Given the pattern

    if px1 then px2
    px1 is true
    Therefore px2 is true

    We want to create many instances using different x variables
    to teach our network to recognize simple modus ponens.

    :param n_vars: Number of possible variables, names, predicates,
    and constants.
    :return: Dictionaries to map from placeholders to actual names
    when instantiating a problem.
    """

    # Generate permutations for variables "x".

    permutation_x = np.random.permutation(n_vars)
    permutation_p = np.random.permutation(n_vars)
    permutation_a = np.random.permutation(n_vars)
    permutation_n = np.random.permutation(n_vars)

    var_x_dict = {}
    inv_var_x_dict = {}
    var_p_dict = {}
    inv_var_p_dict = {}
    var_a_dict = {}
    inv_var_a_dict = {}
    var_n_dict = {}
    inv_var_n_dict = {}

    for n in range(n_vars):

        var_x_dict['px'+str(n)] = 'x'+str(permutation_x[n])
        inv_var_x_dict['x'+str(permutation_x[n])] = 'px'+str(n)

        var_p_dict['pp' + str(n)] = 'p'+str(permutation_p[n])
        inv_var_p_dict['p' + str(permutation_p[n])] = 'pp' + str(n)

        var_a_dict['pa' + str(n)] = 'a'+str(permutation_a[n])
        inv_var_a_dict['a' + str(permutation_a[n])] = 'pa' + str(n)

        var_n_dict['pn' + str(n)] = 'n'+str(permutation_n[n])
        inv_var_n_dict['n' + str(permutation_n[n])] = 'pn' + str(n)

    # End permutation loop

    permutation_instance = {
        'var_x_dict'        :       var_x_dict,
        'inv_var_x_dict'    :       inv_var_x_dict,
        'var_p_dict'        :       var_p_dict,
        'inv_var_p_dict'    :       inv_var_p_dict,
        'var_a_dict'        :       var_a_dict,
        'inv_var_a_dict'    :       inv_var_a_dict,
        'var_n_dict'        :       var_n_dict,
        'inv_var_n_dict'    :       inv_var_n_dict,
    }

    return permutation_instance

# End function permutation_generator

def string_encoder(target_string,
                   permutation_instance, symbol_dictionary):

    """
    problem_encoder: Takes a string template full of placeholder
    names for variables, names, predicates, and constants and
    generates a problem instance both as strings and as
    encoded vectors.

    :param target_string:
    :param permutation_instance:
    :param symbol_dictionary:
    :return: dictionary containing problem instance strings and
    arrays of corresponding vectors, but
    """

    # Lower case and tokenize each string, splitting using spaces

    target_list = target_string.lower().split()

    m_size = len(target_list)

    n_size = encode_string('and', symbol_dictionary).size

    initial_array = np.zeros((0, n_size))

    # Loop through the list

    for index in range(m_size):

        current_target = target_list[index]

        for translation in ['var_x_dict', 'var_p_dict',
                            'var_n_dict', 'var_a_dict']:

            test_value = \
            permutation_instance[translation].get(current_target)

            if test_value is not None:
                target_list[index] = test_value
                current_target = test_value
        # End loop over the permutation translation process

        if current_target not in symbol_dictionary:
            current_target = None
        # End logic to change unknown symbols to None

        # Assign processed current target back to the list.

        target_list[index] = current_target

        # Add encoding vector

        initial_array = np.vstack((initial_array,
                                   encode_string(current_target,
                                                 symbol_dictionary)))

    # End element loop

    return (target_list, initial_array)

# End function string_encoder

def gen_problem(question_template, answer_template,
                permutation_instance, symbol_dictionary):
    """
    gen_problem: Generates a problem instance.  Accepts generic
    question and answer templates, a permutation instance, and
    a dictionary of symbols to be used.
    :param question_template: Template with placeholders for question
    :param answer_template: Template with placeholders for answer
    :param permutation_instance: Permutation mapping placeholders to values
    :param symbol_dictionary: One-hot dictionary of allowed symbols
    :return: tuple of form (question_list, answer_list,
    question_array, answer_array).  The lists show the substitutions and
    final symbols used in encoding.  The question and answer arrays are
    one hot encoded with each symbol mapping into a row.
    """

    (question_list, question_array) = string_encoder(question_template,
                                                     permutation_instance,
                                                     symbol_dictionary)

    (answer_list, answer_array) = string_encoder(answer_template,
                                                 permutation_instance,
                                                 symbol_dictionary)

    return (question_list, answer_list, question_array, answer_array)

# End function gen_problem

def gen_problem_extended(problem_template,
                         permutation_instance, symbol_dictionary):
    """
    gen_problem_extended: This is designed to handled extended templates
    in which you have a multi-part question and and multi-part answer,
    sort of like a Q and A.  Here is an example:

    if a then b . b is false .  what is a ?

    a is false .

    if c then a . what is c ?

    c is false .

    In the above, notice how the state of the world from the first part
    of the problem, where a must be false, impacts how the system must
    answer the second question in the problem.  This is important because
    we wish to train systems to handle highly interactive sessions involving
    the logical state of the world.

    A question template consists of at least one part but may have up to
    N parts.  There are then (N-1) newline "\n" characters in it.  An answer
    must also have exactly N parts with (N-1) newlines.  So a simple case of
    N=1 would not have newlines.  We will split with newlines and verify number
    of parts is correct.

    :param problem_template['question_template']: Newline separated multi-part question.
    :param problem_template['answer_template']: Newline separated multi-part answer.
    :param permutation_instance: A dictionary of substitutions to replace
    the generic parts with real constants, variables, predictates, names,
    etc.
    :param symbol_dictionary: One-hot dictionary of allowable symbols.
    :return: problem_dictionary: A dictionary with keys and values:
        key "substitutions" : permutation_instance
        key "question_one_hot" : List of one-hot arrays corresponding to
        problem lines
        key "answer_one_hot" : List of one-hot arrays corresponding to
        answer lines
    """

    # Read the input problem template dictionary to get string templates out.

    question_template = problem_template.get('q')
    answer_template = problem_template.get('a')

    # Split on "\n" to get problem lines and answer lines.

    question_lines = question_template.lower().split("\n")
    answer_lines = answer_template.lower().split("\n")

    if len(question_lines) != len(answer_lines):

        logging.warning("Number of lines of problem must equal lines in answer!")
        return None

    # End error check on number of lines

    # Generate lists to hold the arrays for the question and
    # answer parts encoded in one-hot format.

    question_one_hot = []
    answer_one_hot = []



    for index in range(len(question_lines)):

        question_template = question_lines[index]
        answer_template = answer_lines[index]

        (question_list, answer_list,
         question_array, answer_array) = \
            gen_problem(question_template, answer_template,
                        permutation_instance, symbol_dictionary)

        question_one_hot.append(question_array)
        answer_one_hot.append(answer_array)

    # End loop over lines

    # Generate the return dictionary.

    return_dict = {}

    return_dict["substitutions"] = permutation_instance
    return_dict["question_one_hot"] = question_one_hot
    return_dict["answer_one_hot"] = answer_one_hot
    return_dict["question_template"] = question_template
    return_dict["answer_template"] = answer_template

    (full_q_array, full_a_array) = create_full_problem_arrays(question_one_hot,
                               answer_one_hot,
                               symbol_dictionary)
    return_dict["full_q_array"] = full_q_array
    return_dict["full_a_array"] = full_a_array

    return return_dict

# End function gen_problem_extended

def gen_problem_set(problem_template_list,
                    num_vars, symbol_dictionary,
                    input_permutation_list=None):

    """
    gen_problem_set: Generates a set of problems based on
    a list of problem dictionary templates.
    :param problem_template_list: List of dictionaries, each of
    which defines a generic problem pattern.
    :param num_vars: Number of variables in permutation
    :param symbol_dictionary: One hot symbol dictionary
    :param input_permutation_list: List of permutations to use.
    Default None.
    :return: A tuple with a list of question symbols, a
    list of answer symbols, a list of question encoded arrays,
    and a corresponding list of answer encoded arrays.
    """

    num_problems = len(problem_template_list)

    problem_dictionaries = []

    # Create variables to track minimum question and minimum answer length.
    # Necessary in order to pad with None as appropriate.
    
    max_q_len = 0
    max_a_len = 0

    output_permutation_list = []

    for index in range(num_problems):

        # Create a permutation to map from placeholders to actual values.
        # Yes, each problem needs its own permutation!

        if input_permutation_list is None:

            current_permutation = permutation_generator(num_vars)

        else:

            current_permutation = copy.deepcopy(input_permutation_list[index])

        output_permutation_list.append(current_permutation)

        problem_dict = gen_problem_extended\
            (problem_template_list[index],
             current_permutation, symbol_dictionary)

        (m_q, n_q) = problem_dict['full_q_array'].shape

        (m_a, n_a) = problem_dict['full_a_array'].shape

        if n_q != n_a or m_q != m_a:
            logging.error("ERROR: Full array padding malfunction!")

        if m_q > max_q_len:
            max_q_len = m_q
            max_a_len = m_a

        problem_dictionaries.append(problem_dict)

    # End loop over number of questions
    return (problem_dictionaries, max_q_len, n_q, output_permutation_list)

# End function gen_problem_set

def pad_question_answer_arrays(question_array, answer_array,
                               symbol_dictionary):

    """
    pad_question_answer_arrays: Given a question array and an answer
    array, both of which could be part of a multi-segment problem, we
    want to pad it so that the question has rows of encoded None while waiting
    for the answer and the answer has a rows of None while listening to
    the question.
    :param question_array: One-hot encoded array with characters in rows
    that contains the question.
    :param answer_array: One-hot encoded array with characters in rows
    that contains the answer.
    :param symbol_dictionary: Dictionary to allow us to encode None
    using one-hot.
    :return: Tuple containing padded question/answer arrays.
    """

    (m_q, n_q) = question_array.shape
    (m_a, n_a) = answer_array.shape


    filler_q = np.array(np.matmul(np.ones((m_a, 1)),
                                  encode_string(None, symbol_dictionary)))

    filler_a = np.array(np.matmul(np.ones((m_q, 1)),
                                  encode_string(None, symbol_dictionary)))

    new_question_array = np.vstack((question_array, filler_q))

    new_answer_array = np.vstack((filler_a, answer_array))

    return (new_question_array, new_answer_array)

# End function pad_question_answer_arrays

def create_full_problem_arrays(question_array_list,
                              answer_array_list,
                              symbol_dictionary):
    """
    create_full_problem_array: Given a list of question arrays and
    answer arrays, create two equal size arrays, one with one-hot
    encoded question segments and one with one-hot encoded answers.
    :param question_array_list: List of question arrays.
    :param answer_array_list: List of answer arrays
    :param symbol_dictionary: Encoding dictionary (used to encode None)
    :return: Two arrays: a full question array and a full answer array.
    """

    if len(question_array_list) != len(answer_array_list):
        logging.warning("WARNING: Number of question/answer parts must be equal!")
        return None

    m_q, n_q = question_array_list[0].shape
    m_a, n_a = answer_array_list[0].shape

    q = np.zeros((0, n_q))
    a = np.zeros((0, n_a))

    for index in range(len(question_array_list)):

        (new_q, new_a) = pad_question_answer_arrays(
            question_array_list[index], answer_array_list[index],
            symbol_dictionary)

        q = np.vstack((q, new_q))

        a = np.vstack((a, new_a))

    # End loop

    return (q, a)

# End function create_full_problem_array

def template_list_to_problem_set(num_vars, template_list, pad_output=True,
                                 one_hot_dictionary=None):
    """
    template_list_to_problem_set: Take a list of dictionaries with
    question and answer templates along with number of variables and
    create a problem set to be used in training, validation, or testing.
    :param num_vars: Maximum number of variables.
    :param template_list: List of templates.  Each template is a dictionary.
    Key 'q' leads to a question string value and key 'a' leads to an answer
    string value.  Such a list of templates maybe generated using some of
    the functions below this one.
    :param pad_output: Pad with None to achieve uniform problem length and
    uniform question length.  Default is True due to batch processing
    possibility.
    :return: A tuple with the same return format as gen_problem_set.
     A tuple with a list of question symbols, a
    list of answer symbols, a list of question encoded arrays,
    and a corresponding list of answer encoded arrays.
    """

    # Create a one-hot encoding dictionary.

    if one_hot_dictionary is None:
        one_hot_dictionary = gen_one_hot_encoding(num_vars=num_vars)

    # Create array of return dictionaries

    (problem_set_list, m_rows, n_cols, permutations) = gen_problem_set(template_list,
                    num_vars, one_hot_dictionary)

    # Create X and Y.

    X = np.zeros((len(problem_set_list), m_rows, n_cols))

    Y = np.zeros((len(problem_set_list), m_rows, n_cols))

    for prob_index in range(len(problem_set_list)):

        (m, n) = problem_set_list[prob_index]['full_q_array'].shape

        X[prob_index, :m, :] = \
        problem_set_list[prob_index]['full_q_array']

        Y[prob_index, :m, :] = \
        problem_set_list[prob_index]['full_a_array']

        (target_list, output_array) = string_encoder("INVALID " * (m_rows - m),
                       problem_set_list[prob_index]['substitutions'], one_hot_dictionary)

        X[prob_index, m:, :n] = output_array

        Y[prob_index, m:, :n] = output_array

    # Return it

    return problem_set_list, X, Y, one_hot_dictionary

# End template_list_to_problem_set

def ext_template_list_to_problem_set(num_vars, template_list, pad_output=True,
                                 one_hot_dictionary=None):
    """
    template_list_to_problem_set: Take a list of dictionaries with
    question and answer templates along with number of variables and
    create a problem set to be used in training, validation, or testing.
    :param num_vars: Maximum number of variables.
    :param template_list: List of templates.  First member is a list
    of question templates with repetition removed.  Second member is a list
    of answer templates.
    :param pad_output: Pad with None to achieve uniform problem length and
    uniform question length.  Default is True due to batch processing
    possibility.
    :return: A tuple with the same return format as gen_problem_set.
     A tuple with a list of question symbols, a
    list of answer symbols, a list of question encoded arrays,
    and a corresponding list of answer encoded arrays.
    """

    # Create a one-hot encoding dictionary.

    if one_hot_dictionary is None:
        one_hot_dictionary = gen_one_hot_encoding(num_vars=num_vars)

    # Create array of return dictionaries

    (problem_set_list_q, m_rows_q, n_cols, permutation_list) = gen_problem_set(template_list[0],
                    num_vars, one_hot_dictionary)

    (problem_set_list_a, m_rows_a, n_cols, permutation_list) = gen_problem_set(template_list[1],
                    num_vars, one_hot_dictionary, input_permutation_list=permutation_list)

    # Create X and Y.

    m_rows = np.max(np.array([m_rows_q, m_rows_a]))

    X = np.zeros((len(problem_set_list_q), m_rows, n_cols))

    Y_q = np.zeros((len(problem_set_list_q), m_rows, n_cols))

    Y_a = np.zeros((len(problem_set_list_a), m_rows, n_cols))

    for prob_index in range(len(problem_set_list_q)):

        (m, n) = problem_set_list_q[prob_index]['full_q_array'].shape

        (m2, n) = problem_set_list_a[prob_index]['full_a_array'].shape

        X[prob_index, :m, :] = \
        problem_set_list_q[prob_index]['full_q_array']

        Y_q[prob_index, :m, :] = \
        problem_set_list_q[prob_index]['full_a_array']

        Y_a[prob_index, :m2, :] = \
        problem_set_list_a[prob_index]['full_a_array']

        (target_list, output_array) = string_encoder("INVALID " * (m_rows - m),
                       problem_set_list_q[prob_index]['substitutions'], one_hot_dictionary)

        X[prob_index, m:, :n] = output_array

        Y_q[prob_index, m:, :n] = copy.deepcopy(output_array)

        (target_list, output_array) = string_encoder("INVALID " * (m_rows - m2),
                       problem_set_list_a[prob_index]['substitutions'], one_hot_dictionary)

        Y_a[prob_index, m2:, :n] = copy.deepcopy(output_array)

    # Return it

    return (problem_set_list_q, problem_set_list_a,
            X, Y_q, Y_a, one_hot_dictionary)

# End ext_template_list_to_problem_set

def equalize_time_samples(list_of_3d_tensors, symbol_dictionary):
    """
    equalize_time_samples: Given a set of 3D batch tensors
    with the same number of batches and the same dimensionality
    for time samples (size = num_batches x num_time_samples x time_sample_dimensionality),
    so that the first and last dimensions are equal but the second might not be
    equal, create new tensors that have equal second dimensions for time
    series compatbility for LSTM training and processing.
    :param list_of_3d_tensors: A list whose members are 3 D tensors
    meeting the above requirements (first/third dimensions are equal)
    :param symbol_dictionary: A dictionary of symbols to vectors used to
    include the null or blank character.
    :return: New tensor list with equal number of time samples.
    """

    (b, m, n) = list_of_3d_tensors[0].shape

    m_max = m

    for current_tensor in list_of_3d_tensors:

        (bc, mc, nc) = current_tensor.shape

        if bc != b or nc != n:
            print ("ERROR in equalize_time_samples: incompatible tensor sizes \n")
            return None

        if mc > m:
            m = mc

    # End search loop over tensor sizes

    # Next, build new tensor list and new tensors

    new_tensor_list = []

    for current_tensor in list_of_3d_tensors:

        new_tensor = np.zeros((b, m, n))

        (bc, mc, nc) = current_tensor.shape

        new_tensor[:, :mc, :] = current_tensor

        filler = np.matmul(np.ones((m-mc,1)), encode_string("INVALID", symbol_dictionary))

        new_tensor[:, mc:, :] = filler

        new_tensor_list.append(new_tensor)

    return new_tensor_list

# End equalize_time_samples

def equal_length_q_a(q_array, a_array, symbol_dictionary):
    """
    equal_length_q_a: Take arrays from q_array and a_array and
    pad them to have equal numbers of rows.
    :param q_array:
    :param a_array:
    :return:
    """

    n_arrays = len(q_array)

    if n_arrays != len(a_array):
        logging.error("Error: Q and A arrays must be of equal length!")
        return None

    (m_q, n_q) = q_array[0].shape
    (m_a, n_a) = a_array[0].shape

    if n_q != n_a:
        logging.error("Error: n_q and n_a are not equal - one hot coding issue!")
        return None

    new_q_array = []
    new_a_array = []

    filler_q = np.array(np.matmul(np.ones((m_a, 1)),
                                  encode_string(None, symbol_dictionary)))

    filler_a = np.array(np.matmul(np.ones((m_q, 1)),
                                  encode_string(None, symbol_dictionary)))

    for index in range(n_arrays):

        new_q = np.vstack((q_array[index], filler_q))
        new_a = np.vstack((filler_a, a_array[index]))

        new_q_array.append(new_q)
        new_a_array.append(new_a)

    # End loop

    return (new_q_array, new_a_array)

# End function equal_length_q_a

# TODO 5 DECEMBER 2018: THINK OF A WAY TO AUTOMATE DISTRACTING VARIABLES
# AND PREVIOUS UNKNOWNS!  Also, include way to ask for help and respond to
# entity requests for help so entities can have conversations.
# Create templates simulating conversations.

def return_simple_propositional_templates():
    """
    return_simple_propositional_templates
    Generates a list of dictionaries.  Each one has
    a problem or question along with an answer.
    These are in string template format with the
    values to be instantiated by random substitution.

    :return: List of dictionaries, each of which has
    a question and an answer, both in string format.
    """

    initial_template_list = []

    # 15 April 2018 - Templates below are Block 1 templates
    # for very basic sanity checking

    initial_template_list.append(
        {
            'statement_list' : [],
            'question' : 'what is pa1 ?',
            'answer' : 'pa1 is unknown',
            'max_instances' : 3
        }
    )

    initial_template_list.append(
        {
            'statement_list' : ['pa1 is unknown'],
            'question' : 'what is pa1 ?',
            'answer' : 'pa1 is unknown',
            'max_instances' : 3
        }
    )

    initial_template_list.append(
        {
            'statement_list' : ['if pa1 then pa2'],
            'question' : 'what is pa2 ?',
            'answer' : 'pa2 is unknown',
            'max_instances' : 3
        }
    )

    initial_template_list.append(
        {
            'statement_list' : ['if pa1 then pa2'],
            'question' : 'what is pa1 ?',
            'answer' : 'pa1 is unknown',
            'max_instances' : 35
        }
    )

    initial_template_list.append(
        {
            'statement_list' : ['pa1 is true'],
            'question' : 'what is pa1 ?',
            'answer' : 'pa1 is true',
            'max_instances' : 3
        }
    )

    initial_template_list.append(
        {
            'statement_list' : ['pa1 is true'],
            'question' : 'what is pa2 ?',
            'answer' : 'pa2 is unknown',
            'max_instances' : 3
        }
    )

    initial_template_list.append(
        {
            'statement_list' : ['pa1 is unknown'],
            'question' : 'what is pa2 ?',
            'answer' : 'pa2 is unknown',
            'max_instances' : 3
        }
    )

    initial_template_list.append(
        {
            'statement_list' : ['pa1 is false'],
            'question' : 'what is pa2 ?',
            'answer' : 'pa2 is unknown',
            'max_instances' : 3
        }
    )

    initial_template_list.append(
        {
            'statement_list' : ['pa1 is false'],
            'question' : 'what is pa1 ?',
            'answer' : 'pa1 is false',
            'max_instances' : 3
        }
    )

    initial_template_list.append(
        {
            'statement_list' : ['pa1 or pa2'],
            'question' : 'what is pa2 ?',
            'answer' : 'pa2 is unknown',
            'max_instances' : 3
        }
    )

    initial_template_list.append(
        {
            'statement_list' : ['pa1 or pa2'],
            'question' : 'what is pa1 ?',
            'answer' : 'pa1 is unknown',
            'max_instances' : 3
        }
    )

    initial_template_list.append(
        {
            'statement_list' : ['pa1 xor pa2'],
            'question' : 'what is pa2 ?',
            'answer' : 'pa2 is unknown',
            'max_instances' : 3
        }
    )

    initial_template_list.append(
        {
            'statement_list' : ['pa1 xor pa2'],
            'question' : 'what is pa1 ?',
            'answer' : 'pa1 is unknown',
            'max_instances' : 3
        }
    )

    # Combined cases (syllogism)

    initial_template_list.append(
        {
            'statement_list' : ['pa1 is true', 'if pa1 then pa2'],
            'question' : 'what is pa1 ?',
            'answer' : 'pa1 is true',
            'max_instances' : 15
        }
    )

    initial_template_list.append(
        {
            'statement_list' : ['pa1 is true', 'if pa1 then pa2'],
            'question' : 'what is pa2 ?',
            'answer' : 'pa2 is true',
            'max_instances' : 15
        }
    )

    initial_template_list.append(
        {
            'statement_list' : ['pa1 is false', 'if pa1 then pa2'],
            'question' : 'what is pa1 ?',
            'answer' : 'pa1 is false',
            'max_instances' : 15
        }
    )

    initial_template_list.append(
        {
            'statement_list' : ['pa1 is false', 'if pa1 then pa2'],
            'question' : 'what is pa2 ?',
            'answer' : 'pa2 is unknown',
            'max_instances' : 15
        }
    )

    initial_template_list.append(
        {
            'statement_list' : ['pa2 is true', 'if pa1 then pa2'],
            'question' : 'what is pa1 ?',
            'answer' : 'pa1 is unknown',
            'max_instances' : 15
        }
    )

    initial_template_list.append(
        {
            'statement_list' : ['pa2 is true', 'if pa1 then pa2'],
            'question' : 'what is pa2 ?',
            'answer' : 'pa2 is true',
            'max_instances' : 15
        }
    )

    initial_template_list.append(
        {
            'statement_list' : ['pa2 is false', 'if pa1 then pa2'],
            'question' : 'what is pa1 ?',
            'answer' : 'pa1 is false',
            'max_instances' : 15
        }
    )

    initial_template_list.append(
        {
            'statement_list' : ['pa2 is false', 'if pa1 then pa2'],
            'question' : 'what is pa2 ?',
            'answer' : 'pa2 is false',
            'max_instances' : 15
        }
    )

    # Combined cases (OR)

    initial_template_list.append(
        {
            'statement_list' : ['pa1 is true', 'pa1 or pa2'],
            'question' : 'what is pa1 ?',
            'answer' : 'pa1 is true',
            'max_instances' : 15
        }
    )

    initial_template_list.append(
        {
            'statement_list' : ['pa1 is true', 'pa1 or pa2'],
            'question' : 'what is pa2 ?',
            'answer' : 'pa2 is unknown',
            'max_instances' : 15
        }
    )

    initial_template_list.append(
        {
            'statement_list' : ['pa1 is false', 'pa1 or pa2'],
            'question' : 'what is pa1 ?',
            'answer' : 'pa1 is false',
            'max_instances' : 15
        }
    )

    initial_template_list.append(
        {
            'statement_list' : ['pa1 is false', 'pa1 or pa2'],
            'question' : 'what is pa2 ?',
            'answer' : 'pa2 is true',
            'max_instances' : 15
        }
    )

    initial_template_list.append(
        {
            'statement_list' : ['pa2 is true', 'pa1 or pa2'],
            'question' : 'what is pa1 ?',
            'answer' : 'pa1 is unknown',
            'max_instances' : 15
        }
    )

    initial_template_list.append(
        {
            'statement_list' : ['pa2 is true', 'pa1 or pa2'],
            'question' : 'what is pa2 ?',
            'answer' : 'pa2 is true',
            'max_instances' : 15
        }
    )

    initial_template_list.append(
        {
            'statement_list' : ['pa2 is false', 'pa1 or pa2'],
            'question' : 'what is pa1 ?',
            'answer' : 'pa1 is true',
            'max_instances' : 15
        }
    )

    initial_template_list.append(
        {
            'statement_list' : ['pa2 is false', 'pa1 or pa2'],
            'question' : 'what is pa2 ?',
            'answer' : 'pa2 is false',
            'max_instances' : 15
        }
    )

    # Combined cases (XOR)

    initial_template_list.append(
        {
            'statement_list' : ['pa1 is true', 'pa1 xor pa2'],
            'question' : 'what is pa1 ?',
            'answer' : 'pa1 is true',
            'max_instances' : 15
        }
    )

    initial_template_list.append(
        {
            'statement_list' : ['pa1 is true', 'pa1 xor pa2'],
            'question' : 'what is pa2 ?',
            'answer' : 'pa2 is false',
            'max_instances' : 15
        }
    )

    initial_template_list.append(
        {
            'statement_list' : ['pa1 is false', 'pa1 xor pa2'],
            'question' : 'what is pa1 ?',
            'answer' : 'pa1 is false',
            'max_instances' : 15
        }
    )

    initial_template_list.append(
        {
            'statement_list' : ['pa1 is false', 'pa1 xor pa2'],
            'question' : 'what is pa2 ?',
            'answer' : 'pa2 is true',
            'max_instances' : 15
        }
    )

    initial_template_list.append(
        {
            'statement_list' : ['pa2 is true', 'pa1 xor pa2'],
            'question' : 'what is pa1 ?',
            'answer' : 'pa1 is false',
            'max_instances' : 15
        }
    )

    initial_template_list.append(
        {
            'statement_list' : ['pa2 is true', 'pa1 xor pa2'],
            'question' : 'what is pa2 ?',
            'answer' : 'pa2 is true',
            'max_instances' : 15
        }
    )

    initial_template_list.append(
        {
            'statement_list' : ['pa2 is false', 'pa1 xor pa2'],
            'question' : 'what is pa1 ?',
            'answer' : 'pa1 is true',
            'max_instances' : 15
        }
    )

    initial_template_list.append(
        {
            'statement_list' : ['pa2 is false', 'pa1 xor pa2'],
            'question' : 'what is pa2 ?',
            'answer' : 'pa2 is false',
            'max_instances' : 15
        }
    )

    # Combined cases (AND)

    initial_template_list.append(
        {
            'statement_list' : ['pa1 and pa2'],
            'question' : 'what is pa1 ?',
            'answer' : 'pa1 is true',
            'max_instances' : 15
        }
    )

    initial_template_list.append(
        {
            'statement_list' : ['pa1 and pa2'],
            'question' : 'what is pa2 ?',
            'answer' : 'pa2 is true',
            'max_instances' : 15
        }
    )

    # Add the "pa4" distractors where we ask about a totally off-the-wall variable.

    pa4_distractors = []

    for a_template in initial_template_list:

        new_template = copy.deepcopy(a_template)

        new_template['question'] = "what is pa4 ?"
        new_template['answer'] = "pa4 is unknown"

        pa4_distractors.append(new_template)

    # End loop

    initial_template_list = initial_template_list + pa4_distractors

    # Use the above very simple templates for
    # initial LIMITED demo!

    template_list = copy.deepcopy(initial_template_list)

    distractor_statements = ['pa1 is unknown', 'pa2 is unknown',
                             'pa3 is unknown', 'pa3 is true', 'pa3 is false',
                             'if pa1 then pa3', 'if pa2 then pa3',
                             'if pa3 then pa1', 'if pa3 then pa2']
    distractor_statements = ['pa1 is unknown', 'pa2 is unknown',
                             'pa3 is unknown', 'pa3 is true', 'pa3 is false']

    distractor_statements_2 = ['pa1 is unknown', 'pa2 is unknown',
                             'pa3 is unknown']

    n_distractors = len(distractor_statements)

    n_distractors_2 = len(distractor_statements_2)

    for current_template in initial_template_list:



        new_templates_list_a = []
        new_templates_list_b = []

        for index in range(n_distractors):

            template_copy_a = copy.deepcopy(current_template)

            template_copy_b = copy.deepcopy(current_template)

            template_copy_a['statement_list'].append(distractor_statements[index])

            template_copy_b['statement_list'].append(distractor_statements[index])

            template_copy_b['question'] = 'help'

            template_list.append(template_copy_a)

            template_list.append(template_copy_b)

            # When the question changes to "help", the answer
            # string depends on the ORDER in which statements are made.
            # This will be finalized in the function create_problem_with_repetion,
            # and in THAT function the template answer will be changed to be
            # a cleaned up version of the statement list.

    intermediate_template_list = copy.deepcopy(template_list)

    for current_template in intermediate_template_list:

        for index in range(n_distractors_2):

            template_copy_a = copy.deepcopy(current_template)

            template_copy_b = copy.deepcopy(current_template)

            template_copy_a['statement_list'].append(distractor_statements_2[index])

            template_copy_b['statement_list'].append(distractor_statements_2[index])

            template_copy_b['question'] = 'help'

            template_list.append(template_copy_a)

            template_list.append(template_copy_b)

            # When the question changes to "help", the answer
            # string depends on the ORDER in which statements are made.
            # This will be finalized in the function create_problem_with_repetion,
            # and in THAT function the template answer will be changed to be
            # a cleaned up version of the statement list.



    # End loop for adding distracting variables

    # Return it

    return template_list

# End return_simple_propositional_templates

def clean_statement_list(statement_list):

    # Search for variable is true, variable is false,
    # and variable is unknown.  If a variable is assigned
    # either true or false, purge away all unknown statements for
    # that variable.  If it is assigned contradictory true or
    # false, return an error.

    regex_true = re.compile(" is true")
    regex_false = re.compile(" is false")
    regex_unknown = re.compile(" is unknown")

    variable_scoreboard = dict()

    # The variable scoreboard uses variables as keys and
    # true, false, and unknown as values.

    # Additionally, we track the ORDER in which the variables
    # are mentioned!

    mention_index = 0

    for statement in statement_list:

        search_true = regex_true.search(statement)
        search_false = regex_false.search(statement)
        search_unknown = regex_unknown.search(statement)

        if search_true is not None:

            var_template = statement[:search_true.start()].strip()

            if var_template in variable_scoreboard and variable_scoreboard[var_template][0] == "false":
                # This is a contradiction.  Must be flagged.
                variable_scoreboard[var_template] = ("ERROR", mention_index)
            elif var_template not in variable_scoreboard or variable_scoreboard[var_template][0] == "unknown":
                # Update to true and update mention_index
                variable_scoreboard[var_template] = ("true", mention_index)
            elif variable_scoreboard[var_template][0] == "true":
                # DO NOTHING AND AVOID UPDATING MENTION INDEX TO KEEP ORDER INTEGRITY!
                pass

        elif search_false is not None:

            var_template = statement[:search_false.start()].strip()

            if var_template in variable_scoreboard and variable_scoreboard[var_template][0] == "true":

                # This is a contradiction - flag as an error
                variable_scoreboard[var_template] = ("ERROR", mention_index)
            elif var_template not in variable_scoreboard or variable_scoreboard[var_template][0] == "unknown":
                variable_scoreboard[var_template] = ("false", mention_index)
            elif variable_scoreboard[var_template][0] == "false":
                # AGAIN DO NOT UPDATE MENTION INDEX TO KEEP INTEGRITY OF ORDERING CORRECT!
                pass
        elif search_unknown is not None:

            var_template = statement[:search_unknown.start()].strip()

            if var_template not in variable_scoreboard:
                variable_scoreboard[var_template] = ("unknown", mention_index)
            else:
                # Again do not change mention index
                pass

        else:
            if statement not in variable_scoreboard:
                variable_scoreboard[statement] = ("other", mention_index)
            else:
                # Again avoid corrupting mention index.
                pass

        # End the conditional for assigning true, false, and unknown.

        mention_index = mention_index + 1

    # End loop over the statements in the input list

    # Run a loop over scorecard and generate new list of statements.

    initial_statement_list = []

    if len(variable_scoreboard) >= 1:

        for variable in variable_scoreboard:

            initial_statement_list.append([variable_scoreboard[variable][1],
                                           variable,
                                           variable_scoreboard[variable][0]])

        # End loop to create initial statement list

        # Create a numpy array

        initial_statement_array = np.array(initial_statement_list)

        # Obtain the sort indices on the first column in order to sort
        # by order of mention.

        sort_indices = np.argsort(initial_statement_array[:, 0])

        # Sort it

        sorted_statement_array = initial_statement_array[sort_indices, :]

        new_statement_list = []

        for index in range(len(sort_indices)):

            if sorted_statement_array[index][2] == "other":
                statement = sorted_statement_array[index][1]
            else:

                statement = "%s is %s" % (sorted_statement_array[index][1],
                                          sorted_statement_array[index][2])

            new_statement_list.append(statement)

        # End loop

        return (new_statement_list, variable_scoreboard)

    else:

        return ([], {})

# End clean_statement_list

# TODO 15 April 2018
# 1. Must write decoder to allow human-level comparison (DONE)
# 2. Must write comparator that removes the Nones to allow true
# accuracy computation (DONE)
# 3. Add code to check for invalid Nones when coding problems
# to catch errors in the templates.
# 3. Then add quad propositional variables, examine, and test.
# 4. Then add quantifiers, which are extremely important in logic! (STARTED)

# IMPORTANT: We believe a fundamental problem is that by demanding question
# and answer have equal length we are wasting the gradient descent algorithm as
# most of the 60+ output symbols are None anyway!  Need to get this to focus
# on shorter answers.  Must come up with a far better architecture for the
# network itself, which seems to be our most fundamental problem.
# Leads to very weak gradient signal, which prevents learning.

def inverse_one_hot_dictionary(one_hot_dictionary, array_to_invert):
    """
    inverse_one_hot_dictionary: Code for running a one-hot dictionary backwards
    using nearest neighbors to obtain original characters.

    If the input array has dimensions (m, p) and each one-hot vector is 1 x p, then
    output will be an m length list, and each element will be a list of p strings that
    are results from inverting the relationship.  We invert by always picking the
    closest one-hot vector.

    :param one_hot_dictionary: A dictionary mapping symbols to one-hot row vectors.
    :param tensor_to_invert: An input tensor to be inverted.
    :return: An array of strings from inversion.
    """

    # Generate an array with the one-hot dictionaries row vectors stack atop each
    # other in key order.  Also generate a corresponding key list.

    key_list = []

    n = one_hot_dictionary[None].size

    vector_array = np.zeros((0, n))

    for key in one_hot_dictionary:

        key_list.append(key)

        vector_array = np.vstack((vector_array, one_hot_dictionary[key]))

    # End generator loop

    (n_vec, n) = vector_array.shape

    # Next, dimension check the input array.

    (m, p) = array_to_invert.shape

    if p != n:
        logging.error("ERROR: Unable to invert one-hot dictionary due to incompatible size!")
        return None

    # Next, generate inversion.

    list_of_results = []

    for index in range(m):

        # Take current vector.

        current_vector = array_to_invert[index, :]

        # Next, check its distance to reference vectors.

        diff_array = np.array(np.matmul(np.ones((n_vec, 1)),
                                        np.reshape(current_vector, (1, n)))) - vector_array

        # Obtain sum of squares and sum over axis 1 to get a column.

        sum_of_squares = np.sum(diff_array**2, axis=1)

        index_min = np.argmin(sum_of_squares)



        closest_key = key_list[index_min]

        list_of_results.append(closest_key)

    # End loop

    return list_of_results

# End function inverse_one_hot_dictionary

def create_problem_with_repetition(problem_def, max_rep=3):
    """
    create_problem_with_repetition:
    If we have a set of statements with a question that
    leads to a given answer, we have to be robust not only
    to the order of the statements but also to repetitions of
    the same true statement(s).  We define a problem in terms
    of a set of true statements and a question and answer.
    Then we create an instance by stating each of the statements
    at lest once and maybe many times in random order followed
    by the question.  The network ought to provide
    two answers:
    1. A single repeat of each statement in any order.
    2. The correct answer to the question.
    Note that we want the network to be capable of
    stating the reasons for its answer, and reciting the
    logical statements it used is a key step.  Note that
    this training would involve regurgitating irrelevant
    statements as well.
    Warning: Present training scheme expects statements
    to be recited in an order.  Future versions must accept
    any order.  This is an ADVANCED topic for LATER!
    :param problem_def: Problem definition.  Has
    list of statements, question, and answer.
    :param max_rep: Max number of repetitions of any one
    statement.
    :return: An instance consisting of a problem string,
    a first answer string consisting of the statements in
    order, and then the answer to the question.
    """

    statement_list = problem_def['statement_list']

    # Now obtain second answer, which is the answer
    # to the question.  Also obtain question itself.

    question = problem_def['question']
    second_answer = problem_def['answer']

    # Create list to hold problem statement copies, including
    # the repetitions.

    num_statements = len(problem_def['statement_list'])
    repetition_count = np.zeros((num_statements,))
    rep_choices = np.random.randint(1, max_rep + 1, size=(num_statements,))
    total_reps = np.sum(rep_choices)

    statement_choices = np.zeros((total_reps,))

    final_statement_list = [None] * total_reps

    working_statement_set = set()

    order_of_first_mention = []

    # Loop through the repetitions and put in statement choices.

    for index in range(total_reps):

        # Keep trying to choose a statement to insert until you
        # get one that still needs to be repeated.

        statement_choice = np.random.randint(num_statements)

        while repetition_count[statement_choice] >= rep_choices[statement_choice]:
            statement_choice = np.random.randint(num_statements)

        # Now increment its counter.

        repetition_count[statement_choice] = repetition_count[statement_choice] + 1

        statement_choices[index] = statement_choice

        final_statement_list[index] = statement_list[statement_choice]

        if statement_choice not in working_statement_set:
            order_of_first_mention.append(statement_choice)
            working_statement_set = working_statement_set | {statement_choice}

    # End loop

    ordered_statement_list = []

    for statement_choice in order_of_first_mention:
        ordered_statement_list.append(statement_list[statement_choice])


    # Clean statement list up.

    (cleaned_statement_list, variable_dictionary) = clean_statement_list(final_statement_list)

    # Generate period separated string for first answer

    first_answer = " . ".join(cleaned_statement_list) + " . "


    # Create problem string.

    problem_string_1 = " . ".join(final_statement_list + [question])

    first_problem_dictionary = {
        'q' : problem_string_1,
        'a' : first_answer
    }

    regex_unknown = re.compile("unknown")

    if question == "help":
        second_answer = copy.deepcopy(first_answer)

    elif regex_unknown.search(second_answer):
        second_answer = " . ".join([second_answer, "help"])

    second_problem_dictionary = {
        'q': problem_string_1,
        'a': second_answer
    }

    # Return response list

    return [first_problem_dictionary, second_problem_dictionary]

# End function create_problem_with_repetition

# MAJOR NOTES:
#
# 1. Introduce two part problem.  Self knows fact a.
# Self asks other entity about fact b.  If other entity
# does not know or knows b is false, conclude c is false or unknown.
# Else conclude c is true.
#
# 2. Maybe introduce new type in the problem generator function.
#
# 3. IMPORTANT: Also introduce answer that if for all entities
# something is known then self knows something as well!
# Put that in the AUGMENTATION.
#
# 4. IMPORTANT: Also introduce answer if that if for all entities
# something is known then random entity x knows it as well!
#
# 5. Also introduce that if for all entities something is
# not known then for entity x it may not be known.  Note
# that we have to be careful how to handle self here.

# See if we can train on something like modus ponens for
# the a, b, c problem above and see if it then can start
# generalizing.  May or may not be able to.
#
# See if we can create general template for a and b jointly
# imply c.  Self knows a.  Ask entity for b.  If b is false
# or if b is unknown then c is not known.  If b is true then
# c.  Note that special case of b being false implying not c.
# For example, fact a is "p implies q".  If statement q is answered
# as false, then statement p is false too, so b is "q is false"
# and c is "p is false" in this context.  But b could be p is true.
# Then c would be "q is true".  However, if b is "p is false",
# then c would be "q is unknown".  Let's attempt the creation
# of a generalized version of this so we can get the system
# to handle both cases where it has facts and cases where it
# knows c is unknown.  Ask if c is unknown.  If c is unknown
# then ask entity x for b.  Then branch accordingly.  This is
# very crude self awareness because when we know we do not
# know we ask to update our knowledge.  If we know we conclude
# what we can.  If we do not know then we ask something that
# is appropriate.

# CRITICAL TODO: Add new function to this and to problem_generator_2
# that will run templates through encode/decode and verify that
# what comes out is the same.  Otherwise, we have very serious
# problems.  That was the bug corrected on 18 November 2018
# that may have been a contributing factor to recent experiment
# failures.
