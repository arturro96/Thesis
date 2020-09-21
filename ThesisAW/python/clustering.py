import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter, defaultdict

from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
np.set_printoptions(suppress=True)

pd.options.mode.chained_assignment = None
import json

labels = ["post_count", "comment_count", "popular_count", "comments_received", "user_count_interacted_with", "user_count_interaction_received"]#, "popularity_value"]


def elbow(data):

    mms = MinMaxScaler()
    mms.fit(data)
    data_transformed = mms.transform(data)
    Sum_of_squared_distances = []
    K = range(1, 15)
    for k in K:
        print(k)
        km = KMeans(n_clusters=k)
        km = km.fit(data_transformed)
        Sum_of_squared_distances.append(km.inertia_)

    plt.plot(K, Sum_of_squared_distances, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Suma kwadratów odległości')
    plt.title('Metoda Elbow')
    plt.savefig("charts/elbow_all.png")
    plt.show()


def plot_centers(id, centers, n_cluster):
    clusters = [i for i in range(n_cluster)]
    plt.yticks(clusters)
    plt.ylabel("Cluster")
    plt.xlabel(labels[id])
    plt.scatter(centers, clusters, c=clusters)
    plt.show()


def plot_pop_hist(n_cluster, data):
    for i in data:
        i = int(i)
        plt.title("Histogram - popularność w klastrze {}".format(i))
        plt.xlabel("Popularność")
        plt.ylabel("Liczba użytkowników")
        plt.yscale('log', nonposy='clip')
        plt.hist(data[i], 10, facecolor='blue', edgecolor='black', alpha=0.5)
        plt.savefig('charts/hist_popularity_{}_{}'.format(n_cluster, i))
        plt.show()


def calculate_cluster_size_by_month(data, goal, n_clusters):
    months = data.groupby(['slot', 'clusters'])

    d = {}
    for i in range(0, 65):
        for j in range(0, n_clusters):
            d[(i, j)] = {}
            if (i, j) not in months.groups:
                d[(i, j)]['all'] = 0
                d[(i, j)]['users'] = []
            else:
                d[(i, j)]['all'] = len(months.groups[(i, j)])
                d[(i, j)]['users'] = data.loc[(data['slot'] == i) & (data['clusters'] == j)]['u']
            if i == 0:
                d[(i, j)]['remain'] = len(d[(i, j)]['users'])
            else:
                d[(i, j)]['remain'] = len(set(list(d[(i - 1, j)]['users'])) & set(list(d[(i, j)]['users'])))
    # print_dict(d)
    d2 = {}
    for k in range(0, n_clusters):
        d2[k] = {}
        d2[k]['all'] = []
        d2[k]['remain'] = []
        for l in range(0, 65):
            d2[k]['all'].append(d[(l, k)]['all'])
            d2[k]['remain'].append(d[(l, k)]['remain'])
    for row in d2:
        plot_clusters_quantity(row, d2[row], goal)

    # for c in months:
    #     clusters = months.groups[c].groupby()


def plot_clusters_quantity(cluster, data, goal):
    dates = []
    for i in range(2008, 2014):
        for j in range(1, 13):
            if i == 2013 and j == 6:
                break
            date = "0{}-{}".format(j, i)
            dates.append(date)

    fig, ax = plt.subplots(figsize=(10, 6))
    # plt.ylim(ymin=0)

    # plt.title('Liczność klastra {}'.format(cluster))
    plt.ylabel('Liczność klastra')
    plt.plot(dates, data['all'])
    plt.plot(dates, data['remain'])
    plt.legend(['wszyscy użytkownicy', 'użytkownicy, którzy nie zmienili klastra'], loc='upper right')
    # plt.yscale('log', nonposy='clip')
    plt.xticks(rotation=45)


    every_nth = 6
    for n, label in enumerate(ax.xaxis.get_ticklabels()):
        if n % every_nth != 0:
            label.set_visible(False)

    plt.savefig('charts/{}_cluster_quantity_by_month_{}.png'.format(cluster, goal))
    plt.show()





def get_stats_for_each_cluster(data, n_clusters, f, labels):
    groupped = data.groupby('clusters')
    min = groupped.min()
    max = groupped.max()
    avg = groupped.mean()
    std = groupped.std()

    for i in range(n_clusters):
        print('Cluster {}'.format(i))
        for g in f:
            print('{} & '.format(g.replace('_', '\\_')), end='')

            mi = round(min[g][i], 2)
            ma = round(max[g][i], 2)
            av = round(avg[g][i], 2)
            st = round(std[g][i], 2)
            print('{} & {} & {} & {} \\\\'.format(mi, ma, av, st))

            print('\\hline')

        print()

        for g in labels:
            print('{} & '.format(g.replace('_', '\\_')), end='')

            mi = round(min[g][i], 2)
            ma = round(max[g][i], 2)
            av = round(avg[g][i], 2)
            st = round(std[g][i], 2)
            print('{} & {} & {} & {} \\\\'.format(mi, ma, av, st))

            print('\\hline')
        print()


def print_table_with_features(data, labels):
    min = data.min()
    max = data.max()
    avg = data.mean()
    med = data.median()
    if len(labels) == 1:
        print(labels)
        for g in labels:
            print('{} & '.format(g.replace('_', '\\_')), end='')
            counter = 0
            for mi, av, ma, me in zip(min[g], avg[g], max[g], med[g]):
                if counter == 4:
                    a = '\\\\'
                else:
                    a = '&'
                mi = round(mi, 2)
                av = round(av, 2)
                ma = round(ma, 2)
                print('{} {} {} {} {} '.format(mi, av, ma, me, a), end='')
                counter += 1
            print()
            print('\\hline')
    else:
        for g in labels:
            print('{} & '.format(g.replace('_', '\\_')), end='')
            counter = 0
            for mi, av, ma in zip(min[g], avg[g], max[g]):
                if counter == 4:
                    a = '\\\\'
                else:
                    a = '&'
                mi = round(mi, 2)
                av = round(av, 2)
                ma = round(ma, 2)
                print('{} {} {} {} '.format(mi, av, ma, a), end='')
                counter += 1
            print()
            print('\\hline')






def get_ids():
    data = pd.read_csv("files/clustering_popularity.csv")
    pd.set_option('display.max_columns', None)
    d = data.loc[(data['popularity'] == 2) & (data['clusters'] == 1)]
    sorted = d.sort_values(by='user')
    print(sorted)


# get_ids()

def calculate_cluster_stats_by_month(data, labels, goal, n_clusters):

    d = {}
    for l in labels:
        d[l] = {}
        for i in range(0, 65):
            for j in range(0, n_clusters):
                d[l][(i, j)] = data.loc[(data['slot'] == i) & (data['clusters'] == j)][l].mean()
    # print_dict(d)
    d2 = {}
    for l in labels:
        d2[l] = {}
        for k in range(0, n_clusters):
            d2[l][k] = []
            for i in range(0, 65):
                d2[l][k].append(d[l][(i, k)])
    for row in d2:
        plot_clusters_stats(row, d2[row], goal, n_clusters)

    # for c in months:
    #     clusters = months.groups[c].groupby()


def plot_clusters_stats(label, data, goal, n_clusters):
    dates = []
    for i in range(2008, 2014):
        for j in range(1, 13):
            if i == 2013 and j == 6:
                break
            date = "0{}-{}".format(j, i)
            dates.append(date)

    fig, ax = plt.subplots(figsize=(10, 6))
    # plt.ylim(ymin=0)
    # plt.title('Liczność klastra {}'.format(cluster))
    plt.ylabel('Średnia wartość {}'.format(label))
    for i in data:
        plt.plot(dates, data[i])


    legend = ['klaster {}'.format(i) for i in range(n_clusters)]
    plt.legend(legend, loc='upper right')
    # plt.yscale('log', nonposy='clip')
    plt.xticks(rotation=45)


    every_nth = 6
    for n, labell in enumerate(ax.xaxis.get_ticklabels()):
        if n % every_nth != 0:
            labell.set_visible(False)

    plt.savefig('charts/{}_cluster_stats_by_month_{}.png'.format(label, goal))
    plt.show()


def calculate_cluster_goal_levels_by_month(data, l, n_clusters):

    d = {}
    for i in range(0, 65):
        for j in range(0, n_clusters):
            d[(i, j)] = {}
            for p in range(1, 4):
                d[(i, j)][p] = data.loc[(data['slot'] == i) & (data['clusters'] == j) & (data[l] == p)][l].count()
    # print_dict(d)
    d2 = {}
    for k in range(0, n_clusters):
        d2[k] = {}
        for p in range(1, 4):
            d2[k][p] = []
            for i in range(0, 65):
                d2[k][p].append(d[(i, k)][p])
    for row in d2:
        plot_clusters_goal_levels(row, d2[row], l)


def plot_clusters_goal_levels(label, data, l):
    dates = []
    for i in range(2008, 2014):
        for j in range(1, 13):
            if i == 2013 and j == 6:
                break
            date = "0{}-{}".format(j, i)
            dates.append(date)

    fig, ax = plt.subplots(figsize=(10, 6))
    # plt.ylim(ymin=0)

    # plt.title('Liczność klastra {}'.format(cluster))
    plt.ylabel('Liczba użytkowników')
    for i in data:
        plt.plot(dates, data[i])

    if l == 'popularity':
        plt.legend(['słaba popularność', 'średnia popularność', 'duża popularność'], loc='upper right')
    elif l == 'influence':
        plt.legend(['słaba wpływowość', 'średnia wpływowość', 'duża wpływowość'], loc='upper right')
    elif l == 'relation':
        plt.legend(['słaba relacja', 'średni poziom relacji', 'silna relacja'], loc='upper right')
    # plt.yscale('log', nonposy='clip')
    plt.xticks(rotation=45)


    every_nth = 6
    for n, labell in enumerate(ax.xaxis.get_ticklabels()):
        if n % every_nth != 0:
            labell.set_visible(False)

    plt.savefig('charts/{}_cluster_{}_level_by_month.png'.format(label, l))
    plt.show()


def check_outliers():
    df = pd.read_csv("files/result_popularity.csv")
    g_features = ['user', 'slot', 'post_written', 'comments_noreply_written', 'comments_reply_written', 'post_interaction_count',
                  'comment_interaction_count', 'interacted_with', 'relations_count_with_popular', 'popularity', 'clusters']

    d = df[g_features]
    print(d.head())
    pd.set_option('display.max_columns', None)

    a = d.groupby(['popularity', 'clusters'])['user'].count()
    print(a)


# 0 - 0, 1 - 3, 2 - 4, 3 - 1, 4 - 2
    data = {}
    for p in range(0, 4):
        data[p] = []
        data[p].append(a[(p, 0)])
        # data[p].append(a[(p, 3)])
        data[p].append(a[(p, 4)])
        data[p].append(a[(p, 1)])
        data[p].append(a[(p, 2)])

    print(data)

    colours = {'Klaster 0': 'C0',
               'Klaster 1': 'C1',
               'Klaster 2': 'C2',
               'Klaster 3': 'C3',
               'Klaster 4': 'C4'}

    labels = 'Klaster 0', 'Klaster 2', 'Klaster 3', 'Klaster 4'
    sizes = np.array(data[3])
    # explode = (0, 0.1, 0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')

    def absolute_value(val):
        a = int(val / 100 * sizes.sum())
        return a

    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, labels=labels, colors=[colours[key] for key in labels], autopct=absolute_value, startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    plt.savefig('charts/popularity_3.png')
    plt.show()



def check_outliers_inf():
    df = pd.read_csv("files/result_influence_inf.csv")
    g_features = ['user', 'slot', 'post_written', 'comments_noreply_written', 'comments_reply_written', 'interacted_with',
                  'user_interaction_received',
                  'relations_count_with_influential', 'influence', 'clusters']

    d = df[g_features]
    print(d.head())

    a = d.groupby(['influence', 'clusters'])['user'].count()
    print(a)

    # 0 - 0, 1 - 3, 2 - 4, 3 - 1, 4 - 2
    data = {}
    for p in range(0, 4):
        data[p] = []
        # data[p].append(a[(p, 0)])
        data[p].append(a[(p, 1)])
        data[p].append(a[(p, 2)])
        data[p].append(a[(p, 3)])
        # data[p].append(a[(p, 4)])

    print(data)

    colours = {'Klaster 0': 'C0',
               'Klaster 1': 'C1',
               'Klaster 2': 'C2',
               'Klaster 3': 'C3',
               'Klaster 4': 'C4'}

    labels = 'Klaster 1', 'Klaster 2', 'Klaster 3'
    sizes = np.array(data[3])

    # explode = (0, 0.1, 0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')

    def absolute_value(val):
        a = int(val / 100 * sizes.sum())
        return a

    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, labels=labels, colors=[colours[key] for key in labels], autopct=absolute_value, startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    plt.savefig('charts/influence_3.png')
    plt.show()


def load_dataset(path):
    df = pd.read_csv(path)
    return df


def check_outliers_rel():
    df = pd.read_csv("files/result_relation.csv")
    g_features = ['user', 'slot', 'interacted_with', 'user_interaction_received', 'interacted_with_last_slot_percentage', 'user_interaction_received_last_slot_percentage', 'relations_count_with_popular', 'relations_count_with_influential', 'relation', 'clusters']

    d = df[g_features]
    print(d.head())
    pd.set_option('display.max_columns', None)

    a = d.groupby(['relation', 'clusters'])['user'].count()
    print(a)


# 0 - 0, 1 - 3, 2 - 4, 3 - 1, 4 - 2
    data = {}
    for p in range(0, 4):
        data[p] = []
        data[p].append(a[(p, 0)])
        # data[p].append(a[(p, 3)])
        data[p].append(a[(p, 1)])
        data[p].append(a[(p, 2)])
        data[p].append(a[(p, 3)])

    print(data)

    colours = {'Klaster 0': 'C0',
               'Klaster 1': 'C1',
               'Klaster 2': 'C2',
               'Klaster 3': 'C3',
               'Klaster 4': 'C4'}

    labels = 'Klaster 0', 'Klaster 1', 'Klaster 2', 'Klaster 3'
    sizes = np.array(data[3])
    # explode = (0, 0.1, 0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')

    def absolute_value(val):
        a = int(val / 100 * sizes.sum())
        return a

    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, labels=labels, colors=[colours[key] for key in labels], autopct=absolute_value, startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    plt.savefig('charts/relation_3.png')
    plt.show()

# check_outliers_rel()


# cluster_again(4)

def calculate_cluster_relation_levels_by_month(data, l, n_clusters):

    d = {}
    for i in range(0, 65):
        for j in range(0, n_clusters):
            d[(i, j)] = {}
            d[(i, j)]['weak'] = data.loc[(data['slot'] == i) & (data['clusters'] == j)]['weak_relations_count'].sum()
            d[(i, j)]['avg'] = data.loc[(data['slot'] == i) & (data['clusters'] == j)]['avg_relations_count'].sum()
            d[(i, j)]['strong'] = data.loc[(data['slot'] == i) & (data['clusters'] == j)]['strong_relations_count'].sum()
    # print_dict(d)
    d2 = {}
    for k in range(0, n_clusters):
        d2[k] = {}
        for p in ['weak', 'avg', 'strong']:
            d2[k][p] = []
            for i in range(0, 65):
                d2[k][p].append(d[(i, k)][p])
    for row in d2:
        plot_clusters_relation_levels(row, d2[row], l)


def plot_clusters_relation_levels(label, data, l):
    dates = []
    for i in range(2008, 2014):
        for j in range(1, 13):
            if i == 2013 and j == 6:
                break
            date = "0{}-{}".format(j, i)
            dates.append(date)

    fig, ax = plt.subplots(figsize=(10, 6))
    # plt.ylim(ymin=0)

    # plt.title('Liczność klastra {}'.format(cluster))
    plt.ylabel('Liczba relacji')
    for i in data:
        plt.plot(dates, data[i])

    plt.legend(['słaba relacja', 'relacja o średniej sile', 'silna relacja'], loc='upper right')

    # plt.yscale('log', nonposy='clip')
    plt.xticks(rotation=45)


    every_nth = 6
    for n, labell in enumerate(ax.xaxis.get_ticklabels()):
        if n % every_nth != 0:
            labell.set_visible(False)

    plt.savefig('charts/{}_cluster_{}_level_by_month.png'.format(label, l))
    plt.show()


def cluster_all_data(n_clusters, goal):
    if goal == 'popularity':
        goal_labels = ['popularity']
        g_features = ['post_written', 'comments_noreply_written', 'comments_reply_written', 'post_interaction_count', 'comment_interaction_count', 'interacted_with', 'relations_count_with_popular']
    elif goal == 'influence':
        goal_labels = ['influence']
        # popularity_stability, influence_stability
        # g_features = ['post_written', 'comments_noreply_written', 'comments_reply_written', 'interacted_with', 'user_interaction_received',
        #             'relations_count_with_influential']
        g_features = ['post_written', 'comments_noreply_written', 'comments_reply_written', 'interacted_with', 'user_interaction_received',
                      'relations_count_with_influential']
    elif goal == 'relation':
        # goal_labels = ['weak_relations_count', 'avg_relations_count', 'strong_relations_count']
        # g_features = ['interacted_with', 'user_interaction_received', 'interacted_with_last_slot_percentage', 'user_interaction_received_last_slot_percentage', 'relations_count_with_popular', 'relations_count_with_influential']
        goal_labels = ['relation']
        g_features = ['interacted_with', 'user_interaction_received', 'interacted_with_last_slot_percentage', 'user_interaction_received_last_slot_percentage', 'relations_count_with_popular', 'relations_count_with_influential']

    # df = pd.read_csv("files/clustering_all_active_fix_p.csv")
    df = pd.read_csv("files/clustering_all_active_relation.csv")
    df = df[df['slot'] > 0]
    df1 = df.iloc[:, 3:]
    features = list(df1.columns)

    data = df1[features]

    mms = MinMaxScaler()

    data[data.columns] = mms.fit_transform(data[data.columns])
    print(data.head())

    data1 = data[g_features]
    # elbow(data1)
    clustering_kmeans = KMeans(n_clusters=n_clusters, precompute_distances="auto", n_jobs=-1).fit(data1)
    c = clustering_kmeans.cluster_centers_
    print(c)

    data1['clusters'] = clustering_kmeans.predict(data1)

    data1['u'] = df.iloc[:, 1]
    data1['slot'] = df.iloc[:, 2]
    data1['weak_relations_count'] = data.iloc[:, -6]
    data1['avg_relations_count'] = data.iloc[:, -5]
    data1['strong_relations_count'] = data.iloc[:, -4]
    data1['popularity'] = data.iloc[:, -3]
    data1['influence'] = data.iloc[:, -2]
    data1['relation'] = data.iloc[:, -1]
    df['clusters'] = data1['clusters']

    groupped = data1.groupby('clusters')

    print_table_with_features(groupped, g_features)
    print_table_with_features(groupped, goal_labels)

    groupped = df.groupby('clusters')

    print()
    print("Not normalized")
    print()

    print_table_with_features(groupped, g_features)
    print_table_with_features(groupped, goal_labels)


    pd.DataFrame(df).to_csv("files/result_{}.csv".format(goal))

    calculate_cluster_size_by_month(data1, goal, n_clusters)
    get_stats_for_each_cluster(df, n_clusters, g_features, goal_labels)
    calculate_cluster_stats_by_month(df, g_features, goal,n_clusters)
    calculate_cluster_stats_by_month(df, goal_labels, goal, n_clusters)
    #
    # # calculate_cluster_stats_by_month_max(df, features_k)
    # # calculate_cluster_stats_by_month_max(df, labels)
    # calculate_cluster_goal_levels_by_month(df, goal, n_clusters)
    calculate_cluster_goal_levels_by_month(df, goal, n_clusters)


class Clustering(object):
    def __init__(self):
        with open('config.json', 'r') as f:
            config = json.load(f)
        self.goal = config['goal']
        self.data = load_dataset(config['data_path'])
        self.n_clusters = config['n_clusters']


    def get_features(self):
        if self.goal == 'popularity':
            goal_labels = ['popularity']
            features = ['post_written', 'comments_noreply_written', 'comments_reply_written',
                          'post_interaction_count', 'comment_interaction_count', 'interacted_with',
                          'relations_count_with_popular']
        elif self.goal == 'influence':
            goal_labels = ['influence']
            features = ['post_written', 'comments_noreply_written', 'comments_reply_written', 'interacted_with',
                          'user_interaction_received',
                          'relations_count_with_influential']
        elif self.goal == 'relation':
            goal_labels = ['relation']
            features = ['interacted_with', 'user_interaction_received', 'interacted_with_last_slot_percentage',
                          'user_interaction_received_last_slot_percentage', 'relations_count_with_popular',
                          'relations_count_with_influential']

        return goal_labels, features

    def cluster(self):
        goal_labels, features = self.get_features()
        df = self.data
        df = df[df['slot'] > 0]
        df1 = df.iloc[:, 3:]
        features = list(df1.columns)

        data = df1[features]

        mms = MinMaxScaler()

        data[data.columns] = mms.fit_transform(data[data.columns])
        print(data.head())

        data1 = data[features]
        clustering_kmeans = KMeans(n_clusters=self.n_clusters, precompute_distances="auto", n_jobs=-1).fit(data1)
        c = clustering_kmeans.cluster_centers_
        print(c)

        data1['clusters'] = clustering_kmeans.predict(data1)

        data1['u'] = df.iloc[:, 1]
        data1['slot'] = df.iloc[:, 2]
        data1['weak_relations_count'] = data.iloc[:, -6]
        data1['avg_relations_count'] = data.iloc[:, -5]
        data1['strong_relations_count'] = data.iloc[:, -4]
        data1['popularity'] = data.iloc[:, -3]
        data1['influence'] = data.iloc[:, -2]
        data1['relation'] = data.iloc[:, -1]
        df['clusters'] = data1['clusters']

        groupped = data1.groupby('clusters')

        print_table_with_features(groupped, features)
        print_table_with_features(groupped, goal_labels)

        groupped = df.groupby('clusters')

        print()
        print("Not normalized")
        print()

        print_table_with_features(groupped, features)
        print_table_with_features(groupped, goal_labels)

        pd.DataFrame(df).to_csv("files/result_{}.csv".format(self.goal))

        calculate_cluster_size_by_month(data1, self.goal, self.n_clusters)
        get_stats_for_each_cluster(df, self.n_clusters, features, goal_labels)
        calculate_cluster_stats_by_month(df, features, self.goal, self.n_clusters)
        calculate_cluster_stats_by_month(df, goal_labels, self.goal, self.n_clusters)
        calculate_cluster_goal_levels_by_month(df, self.goal, self.n_clusters)


if __name__ == '__main__':
    c = Clustering()
    c.cluster()

# cluster_all_data(4, 'relation')



