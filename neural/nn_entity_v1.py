"""
nn_entity_v1.py

Initial version by R. Mukai 4 March 2018

Updated version 24 December 2018.  This code is used for our
initial paper on two cooperating agents with knowledge states
solving a simple propositional logic problem.
"""

import pickle
import numpy as np
import json
from keras.models import Model
from keras.layers import Input
from keras.layers import Bidirectional, TimeDistributed
from keras.layers.core import Activation, Dense, Dropout
from keras.layers.recurrent import LSTM
from keras.layers.merge import concatenate
from keras.models import load_model

from nn_utils.problem_generator import *
from nn_utils.string_and_array import *
#from neural.basic_nn_1 import *

DATA_PATH = "../data"

class NN_Entity_1:

    def __init__(self, id_number,
                 initial_sentences = [],
                 nn_file = None, data_set_file = None,
                 gen_data_sets=True, start_training_epoch=0, max_training_epoch=1024, max_data_sets=64,
                 num_instances=10000
                 ):

        # Create network if nn_file is None.  Otherwise
        # just load existing network.

        if nn_file is None:

            self.generate_network(gen_data_sets, start_training_epoch, max_training_epoch, max_data_sets,
                 num_instances)
            self.network_knowledge_test()
        else:

            self.model = load_model(nn_file)

            # Now decide whether to use existing data
            # if supplied or make new data set.

            if data_set_file is not None:

                fp = open('%s_X1.npy' % (data_set_file,), 'rb')
                X = np.load(fp)
                fp.close()

                fp = open('%s_Y1.npy' % (data_set_file,), 'rb')
                Y1 = np.load(fp)
                fp.close()

                fp = open('%s_Y2.npy' % (data_set_file,), 'rb')
                Y2 = np.load(fp)
                fp.close()

                fp = open('%s_one_hot_dictionary.pck' % (data_set_file,), 'rb')
                one_hot_dictionary = pickle.load(fp)
                fp.close()

                fp = open('%s_template_choices.pck' % (data_set_file,), 'rb')
                template_choices = pickle.load(fp)
                fp.close()

                fp = open('%s_question_template_list.pck' % (data_set_file,), 'rb')
                question_template_list = pickle.load(fp)
                fp.close()

                fp = open('%s_answer_template_list.pck' % (data_set_file,), 'rb')
                answer_template_list = pickle.load(fp)
                fp.close()

                self.data_set={}

                self.data_set['X'] = X
                self.data_set['Y1'] = Y1
                self.data_set['Y2'] = Y2
                self.data_set['one_hot_dictionary'] = one_hot_dictionary
                self.data_set['template_choices'] = template_choices
                self.data_set['question_template_list'] = question_template_list
                self.data_set['answer_template_list'] = answer_template_list

                (b, self.m, self.n) = self.data_set['X'].shape

            # End if-then logic for data set

        # End if-then logic for making new network

        # Set up an entity number for this entity.
        # When referring to itself, it should use "me".
        # When other entities refer to it, it should be
        # "n<entity_number>", for example "n7" if entity
        # number is 7.  Also, set up initial sentences this
        # entity knows, if any.

        self.entity_id_and_knowledge = {
            'id_number' : id_number,
            'id_string' : 'n'+str(id_number),
            'sentence_list' : copy.deepcopy(initial_sentences)
        }



    # End initializer

    def generate_network(self, gen_data_sets=True, start_training_epoch=0, max_training_epoch=1024, max_data_sets=64,
                         num_instances = 10000):
        template_list = return_simple_propositional_templates()

        # Next, create the list of actual question and answer templates.

        new_templates_distilled_question = []

        new_templates_answer = []

        num_template_repeats = 5

        for template in template_list:

            for index in range(template['max_instances']):
                new_pair = create_problem_with_repetition(template, max_rep=3)

                new_templates_distilled_question.append(new_pair[0])
                new_templates_answer.append(new_pair[1])

                question_string = json.dumps(new_pair[0],indent=4)
                answer_string = json.dumps(new_pair[1],indent=4)
        # End loop

        num_templates = len(new_templates_distilled_question)

        template_error_scorecard = {}
        for index in range(num_templates):
            template_error_scorecard[index] = [0, 0, 0]


        fp = open('%s/blank_template_error_scorecard.npy' % (DATA_PATH,), 'wb')
        pickle.dump(template_error_scorecard, fp)
        fp.close()
        template_choices = []

        if gen_data_sets:
            for data_set_index in range(max_data_sets):

                template_choices = []
                question_template_list = []
                answer_template_list = []

                for index in range(num_instances):
                    random_index = np.random.randint(0, num_templates)
                    template_choices.append(random_index)
                    question_template_list.append(new_templates_distilled_question[random_index])
                    answer_template_list.append(new_templates_answer[random_index])

                one_hot_dictionary = gen_one_hot_encoding(num_vars=10)

                result_list_q, result_list_a, X1, Y1, Y2, one_hot_dictionary = ext_template_list_to_problem_set(num_vars=10,
                                                                                                                template_list=(
                                                                                                                question_template_list,
                                                                                                                answer_template_list),
                                                                                                                one_hot_dictionary=one_hot_dictionary)

                data_set = {}
                data_set['X'] = X1
                data_set['Y1'] = Y1
                data_set['Y2'] = Y2
                data_set['one_hot_dictionary'] = one_hot_dictionary
                data_set['template_choices'] = template_choices
                data_set['question_template_list'] = question_template_list
                data_set['answer_template_list'] = answer_template_list

                fp = open('%s/data_set_%d_X1.npy' % (DATA_PATH, data_set_index,), 'wb')
                np.save(fp, X1)
                fp.close()

                fp = open('%s/data_set_%d_Y1.npy' % (DATA_PATH, data_set_index,), 'wb')
                np.save(fp, Y1)
                fp.close()

                fp = open('%s/data_set_%d_Y2.npy' % (DATA_PATH, data_set_index,), 'wb')
                np.save(fp, Y2)
                fp.close()

                fp = open('%s/data_set_%d_one_hot_dictionary.pck' % (DATA_PATH, data_set_index,), 'wb')
                pickle.dump(one_hot_dictionary, fp)
                fp.close()

                fp = open('%s/data_set_%d_template_choices.pck' % (DATA_PATH, data_set_index,), 'wb')
                pickle.dump(template_choices, fp)
                fp.close()

                fp = open('%s/data_set_%d_question_template_list.pck' % (DATA_PATH, data_set_index,), 'wb')
                pickle.dump(question_template_list, fp)
                fp.close()

                fp = open('%s/data_set_%d_answer_template_list.pck' % (DATA_PATH, data_set_index,), 'wb')
                pickle.dump(answer_template_list, fp)
                fp.close()

            # End data set generator loop

        # End data set generator code

        if gen_data_sets:

            (mx, nx) = X1[0].shape

        else:
            fp = open('%s/data_set_%d_X1.npy' % (DATA_PATH, 0,), 'rb')
            X1 = np.load(fp)
            (mx, nx) = X1[0].shape
            fp.close()

            fp = open('%s/data_set_%d_Y1.npy' % (DATA_PATH, 0,), 'rb')
            Y1 = np.load(fp)
            fp.close()

            fp = open('%s/data_set_%d_Y2.npy' % (DATA_PATH, 0,), 'rb')
            Y2 = np.load(fp)
            fp.close()

            fp = open('%s/data_set_%d_one_hot_dictionary.pck' % (DATA_PATH, 0,), 'rb')
            one_hot_dictionary = pickle.load(fp)
            fp.close()

            fp = open('%s/data_set_%d_template_choices.pck' % (DATA_PATH, 0,), 'rb')
            template_choices = pickle.load(fp)
            fp.close()

            fp = open('%s/data_set_%d_question_template_list.pck' % (DATA_PATH, 0,), 'rb')
            question_template_list = pickle.load(fp)
            fp.close()

            fp = open('%s/data_set_%d_answer_template_list.pck' % (DATA_PATH, 0,), 'rb')
            answer_template_list = pickle.load(fp)
            fp.close()

        if start_training_epoch == 0:

            inputs = Input(shape=(None, nx))

            x1 = Bidirectional(LSTM(256, return_sequences=True))(inputs)

            x2 = Bidirectional(LSTM(256, return_sequences=True))(x1)

            n_y1_batch, n_y1_timesteps, n_y1_size = Y1.shape

            n_y2_batch, n_y2_timesteps, n_y2_size = Y2.shape

            x3 = (Dense(Y1[0, 0].size))(x1)

            y1 = Activation('softmax')(x3)

            x4 = (Dense(Y1[0, 0].size))(x2)

            y2 = Activation('softmax')(x4)

            model = Model(inputs=inputs, outputs=[y1, y2])

            model.compile(optimizer='Adam', loss='categorical_crossentropy',
                          metrics=['accuracy'])

            model.save("%s/untrained_dual_output.h5" % (DATA_PATH, ))

        else:

            model = load_model("%s/trained_model_prop_new_%d.h5" % (DATA_PATH, start_training_epoch-1,))

        model.summary()

        for index in range(start_training_epoch, max_training_epoch):

            # IMPORTANT: EXCLUDE SET 0 AND USE FOR TESTING LATER ON!

            data_set_index = np.random.randint(1, max_data_sets)

            fp = open('%s/data_set_%d_X1.npy' % (DATA_PATH, data_set_index,), 'rb')
            X = np.load(fp)
            fp.close()

            fp = open('%s/data_set_%d_Y1.npy' % (DATA_PATH, data_set_index,), 'rb')
            Y1 = np.load(fp)
            fp.close()

            fp = open('%s/data_set_%d_Y2.npy' % (DATA_PATH, data_set_index,), 'rb')
            Y2 = np.load(fp)
            fp.close()

            fp = open('%s/data_set_%d_one_hot_dictionary.pck' % (DATA_PATH, data_set_index,), 'rb')
            one_hot_dictionary = pickle.load(fp)
            fp.close()

            fp = open('%s/data_set_%d_one_hot_dictionary.pck' % (DATA_PATH, data_set_index,), 'rb')
            one_hot_dictionary = pickle.load(fp)
            fp.close()

            fp = open('%s/data_set_%d_template_choices.pck' % (DATA_PATH, data_set_index,), 'rb')
            template_choices = pickle.load(fp)
            fp.close()

            fp = open('%s/data_set_%d_question_template_list.pck' % (DATA_PATH, data_set_index,), 'rb')
            question_template_list = pickle.load(fp)
            fp.close()

            fp = open('%s/data_set_%d_answer_template_list.pck' % (DATA_PATH, data_set_index,), 'rb')
            answer_template_list = pickle.load(fp)
            fp.close()

            model.fit(X, [Y1, Y2], epochs=1, validation_split=0.2)
            # model.fit(X, Y2, epochs=1, validation_split=0.2)

            if index % 16 == 0:

                model.save("%s/trained_model_prop_new_%d.h5" % (DATA_PATH, index,))

        # End training segment

        # Copy key data to self.

        self.model = model
        data_set = {}
        data_set['X'] = X1
        data_set['Y1'] = Y1
        data_set['Y2'] = Y2
        data_set['one_hot_dictionary'] = one_hot_dictionary
        data_set['template_choices'] = template_choices
        data_set['question_template_list'] = question_template_list
        data_set['answer_template_list'] = answer_template_list
        self.data_set = data_set

        (b, self.m, self.n) = self.data_set['X'].shape

        (b1, m1, n1) = Y1.shape

        (b2, m2, n2) = Y2.shape

        self.m1 = m1

        self.n1 = n1

        self.m2 = m2

        self.n2 = n2

    # End method generate_network

    def network_knowledge_test(self):

        fp = open('%s/blank_template_error_scorecard.npy' % (DATA_PATH, ), 'rb')
        template_error_scorecard = pickle.load(fp)
        fp.close()

        fp = open('%s/data_set_%d_X1.npy' % (DATA_PATH, 0,), 'rb')
        X1 = np.load(fp)
        (mx, nx) = X1[0].shape
        fp.close()

        fp = open('%s/data_set_%d_Y1.npy' % (DATA_PATH, 0,), 'rb')
        Y1 = np.load(fp)
        fp.close()

        fp = open('%s/data_set_%d_Y2.npy' % (DATA_PATH, 0,), 'rb')
        Y2 = np.load(fp)
        fp.close()

        fp = open('%s/data_set_%d_one_hot_dictionary.pck' % (DATA_PATH, 0,), 'rb')
        one_hot_dictionary = pickle.load(fp)
        fp.close()

        fp = open('%s/data_set_%d_template_choices.pck' % (DATA_PATH, 0,), 'rb')
        template_choices = pickle.load(fp)
        fp.close()

        fp = open('%s/data_set_%d_question_template_list.pck' % (DATA_PATH, 0,), 'rb')
        question_template_list = pickle.load(fp)
        fp.close()

        fp = open('%s/data_set_%d_answer_template_list.pck' % (DATA_PATH, 0,), 'rb')
        answer_template_list = pickle.load(fp)
        fp.close()

        [Y1_hat, Y2_hat] = self.model.predict(X1)
        # Y2_hat = model.predict(X1)

        (b1, m1, n1) = Y1.shape

        (b2, m2, n2) = Y2.shape

        error_count1 = 0
        error_count2 = 0


        fp = open("%s/raw_results.txt" % (DATA_PATH, ), "w")


        for batch_index in range(b1):

            original_question = inverse_one_hot_dictionary(one_hot_dictionary,
                                                           X1[batch_index, :, :])

            original_answer1 = inverse_one_hot_dictionary(one_hot_dictionary,
                                                          Y1[batch_index, :, :])

            network_answer1 = inverse_one_hot_dictionary(one_hot_dictionary,
                                                         Y1_hat[batch_index, :, :])

            original_answer2 = inverse_one_hot_dictionary(one_hot_dictionary,
                                                          Y2[batch_index, :, :])

            network_answer2 = inverse_one_hot_dictionary(one_hot_dictionary,
                                                         Y2_hat[batch_index, :, :])

            fp.write("%s\n%s\n%s\n%s\n%s\n" % (
                " ".join([x for x in original_question if x]),
                " ".join([x for x in original_answer1 if x]),
                " ".join([x for x in network_answer1 if x]),
                " ".join([x for x in original_answer2 if x]),
                " ".join([x for x in network_answer2 if x])
            ))

            current_template = template_choices[batch_index]

            template_error_scorecard[current_template][0] = \
                template_error_scorecard[current_template][0] + 1

            n_len = len(original_answer1)

            n_len = len(original_answer2)
            error_flag1 = False

            error_flag2 = False

            for answer_index in range(n_len):
                if original_answer1[answer_index] != network_answer1[answer_index]:
                    error_flag1 = True
                    fp.write("ERROR ANSWER 1\n")

            for answer_index in range(n_len):
                if original_answer2[answer_index] != network_answer2[answer_index]:
                    error_flag2 = True
                    fp.write("ERROR ANSWER 2\n")

            if error_flag1:
                error_count1 = error_count1 + 1

                template_error_scorecard[current_template][1] = \
                    template_error_scorecard[current_template][1] + 1

            if error_flag2:
                error_count2 = error_count2 + 1

                template_error_scorecard[current_template][2] = \
                    template_error_scorecard[current_template][2] + 1
            fp.write("\n")
        # End of error printing loop

        fp.close()

        fp = open("%s/key_results_prop_new.txt" % (DATA_PATH, ), "w")

        for template_id in template_error_scorecard:
            print("Template ID and name %d: Tests %d Errors %d Errors %d\n" % (template_id,
                                                                               template_error_scorecard[template_id][0],
                                                                               template_error_scorecard[template_id][1],
                                                                               template_error_scorecard[template_id][
                                                                                   2]))

            fp.write("Template ID and name %d: Tests %d Errors %d Errors %d\n" % (template_id,
                                                                                  template_error_scorecard[template_id][
                                                                                      0],
                                                                                  template_error_scorecard[template_id][
                                                                                      1],
                                                                                  template_error_scorecard[template_id][
                                                                                      2]))

        fp.close()

    # End method network_knowledge_test

    def query_my_knowledge(self, item_to_query):

        X = encode_string(item_to_query, self.data_set['one_hot_dictionary'])
        (m, n) = X.shape
        encoding_of_none = encode_string(None, self.data_set['one_hot_dictionary'])[0]
        X_pad = np.ones((1, self.m, self.n)) * encoding_of_none
        Y1_pad = np.ones((1, self.m, self.n)) * encoding_of_none
        Y2_pad = np.ones((1, self.m, self.n)) * encoding_of_none
        X_pad[0, :m, :] = X

        x_check = np.sum(X_pad, 2)
        y1_check = np.sum(Y1_pad, 2)
        y2_check = np.sum(Y2_pad, 2)
        [Y1_hat,Y2_hat] = self.model.predict(X_pad)

        original_question = inverse_one_hot_dictionary(self.data_set['one_hot_dictionary'],
                                                       X_pad[0, :, :])

        network_answer_array1 = inverse_one_hot_dictionary(self.data_set['one_hot_dictionary'],
                                                    Y1_hat[0, :, :])

        network_answer_array2 = inverse_one_hot_dictionary(self.data_set['one_hot_dictionary'],
                                                    Y2_hat[0, :, :])

        network_answer_string1 = array_to_string(network_answer_array1)

        network_answer_string2 = array_to_string(network_answer_array2)

        return_dict = {
            'original_question' : original_question,

            'network_answer_array1': network_answer_array1,

            'network_answer_string1': network_answer_string1,

            'network_answer_array2' : network_answer_array2,

            'network_answer_string2' : network_answer_string2,
            'X' : X_pad,
            'Y1' : Y1_hat,
            'Y2' : Y2_hat
        }

        return return_dict

    # End function query_my_knowledge

    def add_knowledge(self, knowledge_sentence):

        knowledge_sentence = knowledge_sentence.rstrip().lstrip()

        self.entity_id_and_knowledge['sentence_list'].append(knowledge_sentence)

    # End method add_knowledge

    # Ask question method combines knowledge base with the question and then asks.

    def ask_question(self, question):

        combined_list = copy.deepcopy(self.entity_id_and_knowledge['sentence_list'])

        combined_list.append(question)

        data_string = " . ".join(combined_list)

        return_dict = self.query_my_knowledge(data_string)

        regex_unknown = re.compile('unknown')

        return return_dict

    # End ask_question

    def ask_question_remember_answer(self, question):

        return_dict = self.ask_question(question)

        regex_unknown = re.compile("unknown")

        if question != "help" and regex_unknown.search(return_dict['network_answer_string2']) is None:

            self.add_knowledge(return_dict['network_answer_string2'])

        return return_dict

    # End method ask_question_remember_answer

# End class declaration NN_Entity_1

def two_agent_demo(agent_1_knowledge, agent_2_knowledge, question_for_agent_1):


    # Create the two agents
    test_entity = NN_Entity_1(id_number=1,
                          nn_file="%s/trained_model_prop_new_2048.h5" % (DATA_PATH, ),
                          data_set_file='%s/data_set_0' % (DATA_PATH, ))
    
    
    test_entity_2 = NN_Entity_1(id_number=2,
                              nn_file="%s/trained_model_prop_new_2048.h5" % (DATA_PATH, ),
                              data_set_file='%s/data_set_0' % (DATA_PATH, ))
    # Give the agents knowledge
    test_entity.add_knowledge(agent_1_knowledge)
    test_entity_2.add_knowledge(agent_2_knowledge)
    
    # Ask first entity for value of a3.  Get its answer
    # and convert answer into format entity 2 can use.
    print("Agent 1 knowledge: %s\n" % (agent_1_knowledge,))
    print("Agent 2 knowledge: %s\n" % (agent_2_knowledge,))
    print("Asking agent 1 the following question: %s\n" % (question_for_agent_1,))
    
    return_dict = test_entity.ask_question_remember_answer(question_for_agent_1)
    return_string = return_dict['network_answer_string2']
    print("Agent 1 response: %s\n" % (return_string,))
    
    return_sentences = return_string.split(".")
    
    return_list = []
    
    regex_help = re.compile("help")
    
    help_flag = False # Only goes true if first entity asks for help.
    for sentence in return_sentences:
        new_sentence = sentence.strip()
        if regex_help.search(new_sentence) is None:
            return_list.append(sentence)
            test_entity_2.add_knowledge(sentence)
        else:
            help_flag = True
    
    # If first entity asked for help, then second entity responds with
    # dump of its own knowledge.
    
    if help_flag:
        print("Requesting help from agent 2\n")
        return_dict_2 = test_entity_2.ask_question_remember_answer("help")
    
        return_string_2 = return_dict_2['network_answer_string2']
    
        print ("Agent 2 knowledge base dump response: %s\n" % (return_string_2,))
    
        return_list_2 = return_string_2.split(".")
    
        for sentence in return_list_2:
    
            new_sentence = sentence.strip()
            test_entity.add_knowledge(new_sentence)
    
        # RE-run
        print("Asking agent 1 the same question again\n")
        new_dictionary = test_entity.ask_question_remember_answer(question_for_agent_1)
    
        new_answer = new_dictionary['network_answer_string2']
        print ("Agent 1's new response given Agent 2's knowledge dump: %s\n" % (new_answer,))
    # End block that runs if agent 1 needs help from agent 2
# End function two-agent demo

