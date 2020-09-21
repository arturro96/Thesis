import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from collections import Counter, defaultdict

from sklearn import metrics
import six
import sys
sys.modules['sklearn.externals.six'] = six

from data_preparation import save_obj, load_obj, print_dict

import json

from sklearn.model_selection import train_test_split

from imblearn.over_sampling import (RandomOverSampler,
                                    SMOTE,
                                    ADASYN)

from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.utils import resample
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

import collections

np.set_printoptions(suppress=True)

from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

from sklearn.metrics import accuracy_score


from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import ComplementNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from imblearn.under_sampling import RandomUnderSampler
from sklearn import tree

import graphviz

def oversample_dataset(X, y):

    under = RandomUnderSampler(sampling_strategy={0.0: 700})
    X, y = under.fit_sample(X, y)
    # print('Under {}'.format(Counter(y)))

    sampler = ADASYN(random_state=42)
    X_rs, y_rs = sampler.fit_sample(X, y)
    # print('ADASYN {}'.format(Counter(y_rs)))

    return X_rs, y_rs


def load_dataset(path):
    df = pd.read_csv(path)
    return df


classifiers = {
    "DecisionTree": DecisionTreeClassifier(),
    "AdaBoost": AdaBoostClassifier(),
    "RandomForest": RandomForestClassifier(),
    "MultilayerPerceptron": MLPClassifier(),
    "ComplementNB": ComplementNB()
}



def combine_data_history_length(df, actual_slot, slot_history, features, goal):

    users = df[df['slot'] == actual_slot]['user']
    d = {}
    for u in users:
        d[u] = []

    features = features[:-1]
    f = []
    for i in range(actual_slot - slot_history + 1, actual_slot + 1):
        new_features = [s + str(i) for s in features]
        f.append(new_features)
        for u in d:
            d_u = df[(df['user'] == u) & (df['slot'] == i)]
            d_u = d_u[features].values.tolist()
            for j in d_u:
                for k in j:
                    d[u].append(k)

    flat = [x for sublist in f for x in sublist]

    l = []
    for u in d:
        l.append(d[u])

    all = np.array(l)
    data = pd.DataFrame(all)
    data.columns = flat

    # print(data.columns)

    return data


def plot_accuracy(d, history, goal):
    print_dict(d)
    # fig, ax = plt.subplots(figsize=(10, 6))
    plt.figure(figsize=(5, 5))
    bars = plt.bar(range(len(d)), list(d.values()), align='center')
    plt.xticks(range(len(d)), list(d.keys()))
    plt.ylabel("accuracy")
    plt.xticks(rotation=45)
    plt.gcf().subplots_adjust(bottom=0.25)
    for rect in bars:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2.0, height, '%.2f' % height, ha='center', va='bottom')
    plt.savefig("charts/{}_{}_accuracy.png".format(goal, history))
    plt.show()


def plot_accuracies(d, goal):
    for history in range(6):
        p = {}
        for c in classifiers:
            p[c] = d[history][c]['accuracy']
        plot_accuracy(p, history, goal)


def add_value_labels(ax, spacing=5):
    # For each bar: Place a label
    for rect in ax.patches:
        # Get X and Y placement of label from rect.
        y_value = rect.get_height()
        x_value = rect.get_x() + rect.get_width() / 2

        # Number of points between bar and label. Change to your liking.
        space = spacing
        # Vertical alignment for positive values
        va = 'bottom'

        # If value of bar is negative: Place label below bar
        if y_value < 0:
            # Invert space to place label below
            space *= -1
            # Vertically align label at top
            va = 'top'

        # Use Y value as label and format number with one decimal place
        label = "{:.2f}".format(y_value)

        # Create annotation
        ax.annotate(
            label,                      # Use `label` as label
            (x_value, y_value - 0.025),         # Place label at end of the bar
            xytext=(0, space),          # Vertically shift label by `space`
            textcoords="offset points", # Interpret `xytext` as offset in points
            ha='center',                # Horizontally center label
            va=va)                      # Vertically align label differently for
                                        # positive and negative values.


def plot_3_stats(d, g, history, goal):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 5))
    # fig.suptitle('Horizontally stacked subplots')
    ax1.bar(range(len(d['f1'])), list(d['f1'].values()), align='center')
    ax1.set_xticks(range(len(d['f1'])))
    ax1.set_xticklabels(list(d['f1'].keys()))
    ax1.xaxis.set_tick_params(rotation=45)
    ax1.set_ylabel("f1 score")
    add_value_labels(ax1)

    ax2.bar(range(len(d['precision'])), list(d['precision'].values()), align='center')
    ax2.set_xticks(range(len(d['precision'])))
    ax2.set_xticklabels(list(d['precision'].keys()))
    ax2.xaxis.set_tick_params(rotation=45)
    ax2.set_ylabel("precision")
    add_value_labels(ax2)

    ax3.bar(range(len(d['recall'])), list(d['recall'].values()), align='center')
    ax3.set_xticks(range(len(d['recall'])))
    ax3.set_xticklabels(list(d['recall'].keys()))
    ax3.xaxis.set_tick_params(rotation=45)
    ax3.set_ylabel("recall")
    add_value_labels(ax3)

    fig.subplots_adjust(bottom=0.25)
    plt.savefig("charts/{}_hist_{}_class_{}.png".format(goal, history, g))


def plot_by_classes(d, goal):
    for history in range(6):
        p = {}
        for g in range(4):
            p[g] = {}
            p[g]['f1'] = {}
            p[g]['recall'] = {}
            p[g]['precision'] = {}
            for c in classifiers:
                p[g]['f1'][c] = d[history][c]['level'][g]['f1']
                p[g]['recall'][c] = d[history][c]['level'][g]['recall']
                p[g]['precision'][c] = d[history][c]['level'][g]['precision']
            # print_dict(p[g])
            plot_3_stats(p[g], g, history, goal)




def plot_accuracy_by_classifier(d, c, goal):
    plt.figure(figsize=(5, 5))
    # print_dict(d)
    bars = plt.bar(range(len(d)), list(d.values()), align='center')
    plt.xticks(range(len(d)), list(d.keys()))
    plt.ylabel("accuracy")
    plt.xlabel("długość historii")
    # plt.xticks(rotation=45)
    # plt.gcf().subplots_adjust(bottom=0.25)
    for rect in bars:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2.0, height, '%.2f' % height, ha='center', va='bottom')
    plt.savefig("charts/{}_{}_accuracy.png".format(goal, c))
    plt.show()


def plot_by_classifier(d, goal):
    for c in classifiers:
        p = {}
        for history in range(6):
            p[history] = d[history][c]['accuracy']
        plot_accuracy_by_classifier(p, c, goal)


def plot_3_stats_by_classifier(d, g, c, goal):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 5))
    # fig.suptitle('Horizontally stacked subplots')
    ax1.bar(range(len(d['f1'])), list(d['f1'].values()), align='center')
    ax1.set_xticks(range(len(d['f1'])))
    ax1.set_xticklabels(list(d['f1'].keys()))
    ax1.set_ylim(top=1.0)
    ax1.set_ylabel("f1 score")
    ax1.set_xlabel("długość historii")
    add_value_labels(ax1)

    ax2.bar(range(len(d['precision'])), list(d['precision'].values()), align='center')
    ax2.set_xticks(range(len(d['precision'])))
    ax2.set_xticklabels(list(d['precision'].keys()))
    ax2.set_ylim(top=1.0)
    ax2.set_ylabel("precision")
    ax2.set_xlabel("długość historii")
    add_value_labels(ax2)

    ax3.bar(range(len(d['recall'])), list(d['recall'].values()), align='center')
    ax3.set_xticks(range(len(d['recall'])))
    ax3.set_xticklabels(list(d['recall'].keys()))
    ax3.set_ylim(top=1.0)
    ax3.set_ylabel("recall")
    ax3.set_xlabel("długość historii")
    add_value_labels(ax3)

    fig.subplots_adjust(bottom=0.25)

    plt.savefig("charts/{}_{}_{}_0.png".format(goal, c, g))


def plot_by_classes_by_classifiers(d, goal):
    for c in classifiers:
        p = {}
        for g in range(4):
            p[g] = {}
            p[g]['f1'] = {}
            p[g]['recall'] = {}
            p[g]['precision'] = {}
            for history in range(6):
                p[g]['f1'][history] = d[history][c]['level'][g]['f1']
                p[g]['recall'][history] = d[history][c]['level'][g]['recall']
                p[g]['precision'][history] = d[history][c]['level'][g]['precision']
            plot_3_stats_by_classifier(p[g], g, c, goal)



class MachineLearning(object):
    def __init__(self):
        with open('config.json', 'r') as f:
            config = json.load(f)
        self.goal = config['goal']
        self.data = load_dataset(config['data_path'])
        self.features = self.get_features()
        self.slot_to_predict = config['slot_to_predict']
        self.oversample = config['oversample']
        self.d = self.prepare_dict()
        # self.d = load_obj("{}_{}_quarter".format(self.goal, self.oversample))

    def get_features(self):
        features = []
        if self.goal == 'popularity':
            features = ['post_written', 'comments_noreply_written', 'comments_reply_written',
                             'post_interaction_count', 'interacted_with', 'relations_count_with_popular', 'popularity']
        elif self.goal == 'influence':
            features = ['post_written', 'comments_noreply_written', 'comments_reply_written',
                             'post_interaction_count', 'interacted_with', 'relations_count_with_popular', 'influence']
        elif self.goal == 'relation':
            features = ['interacted_with', 'user_interaction_received', 'interacted_with_last_slot_percentage', 'user_interaction_received_last_slot_percentage', 'relations_count_with_popular', 'relations_count_with_influential']
        return features

    def prepare_data(self, slot, slot_history):
        df = self.data

        actual_slot = slot
        last_slot = actual_slot - 6

        y = df[df['slot'] == actual_slot]

        df = combine_data_history_length(df, last_slot, slot_history, self.features, self.goal)

        X = df
        y = y[self.goal]

        if self.oversample:
            X, y = oversample_dataset(X, y)
        return X, y

    def prepare_dict(self):
        p = {}
        for history in range(1, 6):
            p[history] = {}
            for c in classifiers:
                p[history][c] = {}
                p[history][c]['accuracy'] = 0
                p[history][c]['level'] = {}
                for l in range(4):
                    p[history][c]['level'][l] = {}
                    p[history][c]['level'][l]['f1'] = 0
                    p[history][c]['level'][l]['precision'] = 0
                    p[history][c]['level'][l]['recall'] = 0

        return p

    def perform_analysis(self):
        for history in range(1, 6):
            slot = self.slot_to_predict
            print("{} - {}".format(history, slot))
            X, y = self.prepare_data(slot, history)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=27)
            for c in classifiers:
                self.classify(classifiers[c], X_train, X_test, y_train, y_test, history)


    def classify(self, classifier, X_train, X_test, y_train, y_test, history):
        pred = classifier.fit(X_train, y_train).predict(X_test)
        cl = list(classifiers.keys())[list(classifiers.values()).index(classifier)]
        self.get_scores(y_test, pred, cl, history)

    def get_scores(self, y_test, pred, c, h):

        acc = round(accuracy_score(y_test, pred), 2)
        self.d[h][c]['accuracy'] = acc

        for l in range(3):
            self.d[h][c]['level'][l]['f1'] = round(f1_score(y_test, pred, average=None)[l], 2)
            self.d[h][c]['level'][l]['recall'] = round(recall_score(y_test, pred, average=None)[l], 2)
            self.d[h][c]['level'][l]['precision'] = round(precision_score(y_test, pred, average=None)[l], 2)


# if __name__ == '__main__':
#     m = MachineLearning()
#     m.perform_analysis()
#     print_dict(m.d)
#     save_obj(m.d, "{}_{}_{}_half_year".format(m.goal, m.oversample, m.slot_to_predict))



def plot_3_stats_by_classifier_fix(d, c, goal):

    labels = [1, 2, 3, 4, 5]
    x = np.arange(len(labels))  # the label locations
    width = 0.16  # the width of the bars

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4))
    # ax1.bar(x - 0.24, list(d[0]['f1'].values()), width, label='0')
    # ax1.bar(x - 0.08, list(d[1]['f1'].values()), width, label='1')
    # ax1.bar(x + 0.08, list(d[2]['f1'].values()), width, label='2')
    # ax1.bar(x + 0.24, list(d[3]['f1'].values()), width, label='3')

    ax1.bar(x - 0.16, list(d[0]['f1'].values()), width, label='0')
    ax1.bar(x, list(d[1]['f1'].values()), width, label='1')
    ax1.bar(x + 0.16, list(d[2]['f1'].values()), width, label='2')
    # ax1.bar(x + 0.24, list(d[3]['f1'].values()), width, label='3')

    # rects2 = ax1.bar(x + width / 2, women_means, width, label='Women')
    # fig.suptitle('Horizontally stacked subplots')
    # ax1.bar(range(len(d[0]['f1'])), list(d[0]['f1'].values()), align='center', label=2008)
    # ax1.bar(range(len(d[1]['f1'])), list(d[1]['f1'].values()), align='center', label=2009)
    # ax1.bar(range(len(d[2]['f1'])), list(d[2]['f1'].values()), align='center', label=2010)
    # ax1.bar(range(len(d[3]['f1'])), list(d[3]['f1'].values()), align='center', label=2011)
    # ax1.bar(range(len(d[4]['f1'])), list(d[4]['f1'].values()), align='center', label=2012)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylim(top=1.0)
    ax1.set_ylabel("f1 score")
    ax1.set_xlabel("długość historii")
    # ax1.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    # add_value_labels(ax1)

    # ax2.bar(range(len(d['precision'])), list(d['precision'].values()), align='center')
    # ax2.bar(x - 0.24, list(d[0]['precision'].values()), width, label='0')
    # ax2.bar(x - 0.08, list(d[1]['precision'].values()), width, label='1')
    # ax2.bar(x + 0.08, list(d[2]['precision'].values()), width, label='2')
    # ax2.bar(x + 0.24, list(d[3]['precision'].values()), width, label='3')

    ax2.bar(x - 0.16, list(d[0]['f1'].values()), width, label='0')
    ax2.bar(x, list(d[1]['f1'].values()), width, label='1')
    ax2.bar(x + 0.16, list(d[2]['f1'].values()), width, label='2')

    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_ylim(top=1.0)
    ax2.set_ylabel("precision")
    ax2.set_xlabel("długość historii")
    # ax2.legend()
    # add_value_labels(ax2)

    # ax3.bar(range(len(d['recall'])), list(d['recall'].values()), align='center')
    # ax3.bar(x - 0.24, list(d[0]['recall'].values()), width, label='brak popularności')
    # ax3.bar(x - 0.08, list(d[1]['recall'].values()), width, label='słaba popularność')
    # ax3.bar(x + 0.08, list(d[2]['recall'].values()), width, label='średnia popularność')
    # ax3.bar(x + 0.24, list(d[3]['recall'].values()), width, label='duża popularność')
    # ax3.bar(x - 0.24, list(d[0]['recall'].values()), width, label='brak wpływowości')
    # ax3.bar(x - 0.08, list(d[1]['recall'].values()), width, label='słaba wpływowość')
    # ax3.bar(x + 0.08, list(d[2]['recall'].values()), width, label='średnia wpływowość')
    # ax3.bar(x + 0.24, list(d[3]['recall'].values()), width, label='duża wpływowość')
    ax3.bar(x - 0.16, list(d[0]['recall'].values()), width, label='brak relacji')
    ax3.bar(x, list(d[1]['recall'].values()), width, label='słaba relacja')
    ax3.bar(x + 0.16, list(d[2]['recall'].values()), width, label='średnia siła relacji')
    # ax3.bar(x + 0.24, list(d[3]['recall'].values()), width, label='silna relacja')


    ax3.set_xticks(x)
    ax3.set_xticklabels(labels)
    ax3.set_ylim(top=1.0)
    ax3.set_ylabel("recall")
    ax3.set_xlabel("długość historii")
    # ax3.legend(bbox_to_anchor=(0.90, 1.1), loc='upper left')
    # add_value_labels(ax3)

    handles, labels = ax3.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=4)

    # fig.subplots_adjust(bottom=0.25)

    plt.savefig("charts/{}_{}_a.png".format(goal, c))


def plot_acc(d, goal):
    p = {}
    for c in classifiers:
        p[c] = {}
        for q in range(1, 6):
            p[c][q] = d[q][c]['accuracy']

    labels = [1, 2, 3, 4, 5]
    x = np.arange(len(labels))  # the label locations
    width = 0.15  # the width of the bars

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - 0.3, list(p['ComplementNB'].values()), width, label='ComplementNaiveBayes')
    ax.bar(x - 0.15, list(p['AdaBoost'].values()), width, label='AdaBoost')
    ax.bar(x, list(p['DecisionTree'].values()), width, label='DecisionTree')
    ax.bar(x + 0.15, list(p['RandomForest'].values()), width, label='RandomForest')
    ax.bar(x + 0.3, list(p['MultilayerPerceptron'].values()), width, label='MultilayerPerceptron')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(top=1.0)
    ax.set_ylabel("accuracy")
    ax.set_xlabel("długość historii")

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=5)

    plt.savefig("charts/{}_accuracy_a.png".format(goal))



def plot_by_classes_by_classifiers_fix(d, goal):
    plot_acc(d, goal)
    for c in classifiers:
        p = {}
        p['accuracy'] = {}
        for y in range(0, 3):      #cele
            p[y] = {}
            p[y]['f1'] = {}
            p[y]['recall'] = {}
            p[y]['precision'] = {}
            for q in range(1, 6):       #długość
                p['accuracy'][q] = d[q][c]['accuracy']
                p[y]['f1'][q] = d[q][c]['level'][y]['f1']
                p[y]['recall'][q] = d[q][c]['level'][y]['recall']
                p[y]['precision'][q] = d[q][c]['level'][y]['precision']
        plot_3_stats_by_classifier_fix(p, c, goal)


# d = load_obj("relation_1_50_half_year")
# plot_by_classes_by_classifiers_fix(d, 'relation')

#
# def plot_treee(slot, slot_history, goal, oversample):
#     if goal == 'popularity':
#         features = ['post_written', 'comments_noreply_written', 'comments_reply_written',
#                     'post_interaction_count', 'interacted_with', 'relations_count_with_popular', 'popularity']
#     elif goal == 'influence':
#         features = ['post_written', 'comments_noreply_written', 'comments_reply_written',
#                     'post_interaction_count', 'interacted_with', 'relations_count_with_popular', 'influence']
#     elif goal == 'relation':
#         features = ['interacted_with', 'user_interaction_received', 'interacted_with_last_slot_percentage',
#                     'user_interaction_received_last_slot_percentage', 'relations_count_with_popular',
#                     'relations_count_with_influential']
#
#     df = pd.read_csv("files/clustering_all_active_all.csv")
#     actual_slot = slot
#     last_slot = actual_slot - 6
#
#     y = df[df['slot'] == actual_slot]
#
#     df = combine_data_history_length(df, last_slot, slot_history, features, goal)
#
#     X = df
#     y = y[goal]
#
#     if oversample:
#         X, y = oversample_dataset(X, y)
#
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=27)
#
#     clf = tree.DecisionTreeClassifier().fit(X_train, y_train)
#
#     features = features[:-1]
#     dot_data = tree.export_graphviz(clf, out_file=None,
#                                     feature_names=features,
#                                     class_names=["0", "1", "2", "3"],
#                                     filled=True)
#
#     # Draw graph
#     graph = graphviz.Source(dot_data, format="png")
#     graph
#     graph.render("decision_tree_graphivz")

# plot_treee(47, 1, "popularity", 1)