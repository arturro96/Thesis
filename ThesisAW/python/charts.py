from data_preparation import save_obj, load_obj, print_dict
import matplotlib.pyplot as plt

def plot_popularuty_histogram():
    user_popularity = load_obj("user_popularity_without_comment_replies")
    num_bins = 100
    vals = user_popularity.values()
    v = [float(x) for x in vals if float(x) >= 0.01]
    plt.title("Popularność użytkowników - histogram")
    plt.xlabel("Popularity")
    plt.ylabel("Liczba użytkowników")
    plt.yscale('log', nonposy='clip')
    plt.hist(v, num_bins, facecolor='blue', alpha=0.5)
    plt.show()


def plot_influence_histogram():
    post_user_measures = load_obj("post_user_measures")
    num_bins = 100
    influence = []
    for user in post_user_measures:
        if post_user_measures[user]["influence"] is not None:
            influence.append(post_user_measures[user]["influence"])
    print(influence)
    plt.title("Wpływowość użytkowników - histogram")
    plt.xlabel("Influence")
    plt.ylabel("Liczba użytkowników")
    plt.yscale('log', nonposy='clip')
    plt.hist(influence, num_bins, facecolor='blue', alpha=0.5)
    plt.show()


def plot_users_by_month():
    dates = []
    for i in range(2008, 2014):
        for j in range(1, 13):
            if i == 2013 and j == 7:
                break
            date = "0{}-{}".format(j, i)
            dates.append(date)
    print(len(dates))
    users = load_obj("users_by_month")
    print(len(users))
    fig, ax = plt.subplots(figsize=(10, 6))
    # plt.title("Liczba aktywnych użytkowników portalu Salon24 na przestrzeni 5 lat")
    plt.ylabel("Liczba użytkowników")
    plt.xticks(rotation=45)
    plt.plot(dates, users)
    print(ax.xaxis.get_ticklabels())

    every_nth = 6
    for n, label in enumerate(ax.xaxis.get_ticklabels()):
        print(label)
        if n % every_nth != 0:
            label.set_visible(False)

    plt.savefig("charts/users_by_month.png")
    plt.show()


def plot_active_users_by_month():
    dates = []
    for i in range(2008, 2014):
        for j in range(1, 13):
            if i == 2013 and j == 7:
                break
            date = "0{}-{}".format(j, i)
            dates.append(date)
    users_count = load_obj("active_users_by_month")
    fig, ax = plt.subplots(figsize=(10, 6))
    # plt.title("Aktywni użytkownicy w danym miesiącu")
    plt.ylabel("Liczba użytkowników")
    plt.xticks(rotation=45)
    plt.plot(dates, users_count)
    print(ax.xaxis.get_ticklabels())

    every_nth = 6
    for n, label in enumerate(ax.xaxis.get_ticklabels()):
        print(label)
        if n % every_nth != 0:
            label.set_visible(False)

    plt.savefig("charts/active_users_by_month.png")
    plt.show()


def plot_active_popular_users(dict):
    dates = []
    for i in range(2008, 2014):
        for j in range(1, 13):
            if i == 2013 and j == 7:
                break
            date = "0{}-{}".format(j, i)
            dates.append(date)
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.ylabel("Popularność")
    plt.xticks(rotation=45)

    all_time_popular_ids = [657, 783, 11926, 675, 1991]
    popular_users = {}

    for i in all_time_popular_ids:
        popular_users[i] = []
    # print_dict(popular_users)
    for i in range(0, 66):
        print(i)
        active_popularity = load_obj(dict + "_{}".format(i))
        for i in all_time_popular_ids:
            if i in active_popularity:
                popular_users[i].append(active_popularity[i]['popularity'])
            else:
                popular_users[i].append(0)

    for i in popular_users:
        plt.plot(dates, popular_users[i], linewidth=0.7)

    plt.legend(['FREE YOUR MIND', 'RENATA RUDECKA-KALINOWSKA', 'KRZYSZTOF LESKI', 'CEZARY KRYSZTOPA', 'SOWINIEC'],
               loc='upper right')

    every_nth = 6
    for n, label in enumerate(ax.xaxis.get_ticklabels()):
        print(label)
        if n % every_nth != 0:
            label.set_visible(False)

    plt.savefig("charts/active_popular_users.png")
    plt.show()


def plot_active_popular_users_weights(dict):
    dates = []
    for i in range(2008, 2014):
        for j in range(1, 13):
            if i == 2013 and j == 7:
                break
            date = "0{}-{}".format(j, i)
            dates.append(date)
    fig, ax = plt.subplots(figsize=(10, 6))

    plt.ylabel("Popularność")
    plt.xticks(rotation=45)

    all_time_popular_ids = [657, 783, 11926, 675, 1991]
    popular_users = {}
    for i in all_time_popular_ids:
        popular_users[i] = []
    # print_dict(popular_users)
    dict = load_obj(dict)
    for i in range(0, 66):
        print(i)
        active_popularity = dict[i]
        # print_dict(active_popularity)
        for j in all_time_popular_ids:
            if j in active_popularity:
                # print("{} - {} - {}".format())
                popular_users[j].append(active_popularity[j]['popularity_weight'])
            else:
                popular_users[j].append(0)


    for i in popular_users:
        plt.plot(dates, popular_users[i], linewidth=0.7)

    plt.legend(['FREE YOUR MIND', 'RENATA RUDECKA-KALINOWSKA', 'KRZYSZTOF LESKI', 'CEZARY KRYSZTOPA', 'SOWINIEC'],
               loc='upper right')


    every_nth = 6
    for n, label in enumerate(ax.xaxis.get_ticklabels()):
        # print(label)
        if n % every_nth != 0:
            label.set_visible(False)

    plt.savefig("charts/{}_weighted_fix.png".format("all"))
    plt.show()


def plot_post_count_by_month():
    dates = []
    for i in range(2008, 2014):
        for j in range(1, 13):
            if i == 2013 and j == 7:
                break
            date = "0{}-{}".format(j, i)
            dates.append(date)
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.title("Liczba postów zamieszczonych w ciągu miesiąca przez danego użytkownika")
    plt.ylabel("Liczba postów")
    plt.xticks(rotation=45)

    all_time_popular_ids = [657, 11926, 675, 440, 783] #, 496, 797, 11, 440, 66]
    popular_users_post_count = {}
    for i in all_time_popular_ids:
        popular_users_post_count[i] = []

    for i in range(0, 66):
        print(i)
        post_count = load_obj("post_count_by_month_{}".format(i))
        for j in all_time_popular_ids:
            if j in post_count:
                popular_users_post_count[j].append(post_count[j])
            else:
                popular_users_post_count[j].append(0)

    for i in popular_users_post_count:
        plt.plot(dates, popular_users_post_count[i], linewidth=0.7)

    plt.legend(['FREE YOUR MIND', 'KRZYSZTOF LESKI', 'CEZARY KRYSZTOPA', 'MAREK MIGALSKI', 'RENATA RUDECKA-KALINOWSKA'],
               loc='upper right')

    every_nth = 6
    for n, label in enumerate(ax.xaxis.get_ticklabels()):
        if n % every_nth != 0:
            label.set_visible(False)

    plt.savefig("charts/popular_users_post_by_month.png")
    plt.show()


def plot_popularity_histograms(dict):

    d = load_obj(dict)

    popularity_dicts = {2009: d[12],
                        2010: d[24],
                        2011: d[36],
                        2012: d[48]}

    num_bins = 100

    for year in popularity_dicts:
        plt.title("Histogram - popularność - styczeń {}".format(year))
        plt.xlabel("Popularity")
        plt.ylabel("Liczba użytkowników")
        plt.yscale('log', nonposy='clip')
        popularity = []
        for user in popularity_dicts[year]:
            popularity.append(float(popularity_dicts[year][user]['popularity_weight']))
        plt.hist(popularity, num_bins, facecolor='blue', alpha=0.5)
        plt.savefig('charts/{}_{}'.format(dict, year))
        plt.show()
        j01 = [i for i in popularity if i >= 0.1]
        j02 = [i for i in popularity if i >= 0.2]



def plot_popularity_max_and_avg(dict):
    max_vals = []
    avg_vals = []
    d2 = load_obj(dict)
    for i in range(0, 66):
        d = d2[i]
        m_val = max(user['popularity_weight'] for user in d.values())
        avg_val = sum(user['popularity_weight'] for user in d.values()) / len(d)
        max_vals.append(m_val)
        avg_vals.append(avg_val)

    dates = []
    for i in range(2008, 2014):
        for j in range(1, 13):
            if i == 2013 and j == 7:
                break
            date = "0{}-{}".format(j, i)
            dates.append(date)

    fig, ax = plt.subplots(figsize=(10, 6))
    plt.title('Średnia i maksymalna wartość miary popularności w zależności od miesiąca')
    plt.ylabel('Popularność')
    plt.plot(dates, max_vals)
    plt.plot(dates, avg_vals)
    plt.legend(['max', 'średnia'], loc='upper right')
    plt.yscale('log', nonposy='clip')
    plt.xticks(rotation=45)

    every_nth = 6
    for n, label in enumerate(ax.xaxis.get_ticklabels()):
        if n % every_nth != 0:
            label.set_visible(False)

    plt.savefig('charts/{}_max_avg'.format(dict))
    plt.show()


def plot_popularity_count(dict):
    weak_p = []
    avg_p = []
    big_p = []
    d2 = load_obj(dict)
    for i in range(0, 66):
        c_w = 0
        c_a = 0
        c_b = 0
        d = d2[i]
        for user in d:
            if d[user]['popularity_weight'] >= 0.3:
                c_b += 1
            elif 0.12 <= d[user]['popularity_weight'] < 0.3:
                c_a += 1
            elif 0.05 <= d[user]['popularity_weight'] < 0.12:
                c_w += 1
        weak_p.append(c_w)
        avg_p.append(c_a)
        big_p.append(c_b)

    dates = []
    for i in range(2008, 2014):
        for j in range(1, 13):
            if i == 2013 and j == 7:
                break
            date = "0{}-{}".format(j, i)
            dates.append(date)

    fig, ax = plt.subplots(figsize=(10, 6))
    # plt.title('Liczba użytkowników wg progów popularności w zależności od miesiąca')
    plt.ylabel('Liczba użytkowników')
    plt.plot(dates, weak_p)
    plt.plot(dates, avg_p)
    plt.plot(dates, big_p)
    plt.legend(['słaba popularność', 'średnia popularność', 'duża popularność'], loc='upper left')
    # plt.yscale('log', nonposy='clip')
    plt.xticks(rotation=45)

    print(sum(weak_p) / len(weak_p))
    print(sum(avg_p) / len(avg_p))
    print(sum(big_p) / len(big_p))

    every_nth = 6
    for n, label in enumerate(ax.xaxis.get_ticklabels()):
        if n % every_nth != 0:
            label.set_visible(False)

    plt.savefig('charts/{}_user_count'.format(dict))
    plt.show()


def plot_popular_users_posts_avg(dict):
    big = load_obj('avg_posts_count_active_popularity')
    avg = load_obj('avg_posts_count_average_active_popularity')

    dates = []
    for i in range(2008, 2014):
        for j in range(1, 13):
            if i == 2013 and j == 7:
                break
            date = "0{}-{}".format(j, i)
            dates.append(date)

    fig, ax = plt.subplots(figsize=(10, 6))
    plt.title('Częstotliwość zameszczania postów')
    plt.ylabel('Liczba postów')
    plt.plot(dates, big)
    plt.plot(dates, avg)
    plt.legend(['duża popularność', 'średnia popularność'], loc='upper right')
    # plt.yscale('log', nonposy='clip')
    plt.xticks(rotation=45)

    every_nth = 6
    for n, label in enumerate(ax.xaxis.get_ticklabels()):
        if n % every_nth != 0:
            label.set_visible(False)

    plt.savefig('charts/{}_user_post_frequency'.format(dict))
    plt.show()


def plot_comments_count():
    all = load_obj('comments_count_by_month')
    reply = load_obj('comments_reply_count_by_month')
    noreply = load_obj('comments_noreply_count_by_month')

    dates = []
    for i in range(2008, 2014):
        for j in range(1, 13):
            if i == 2013 and j == 7:
                break
            date = "0{}-{}".format(j, i)
            dates.append(date)

    fig, ax = plt.subplots(figsize=(10, 6))
    # plt.title('Liczba zamieszczanych komentarzy')
    plt.ylabel('Liczba komentarzy')
    plt.plot(dates, all)
    plt.plot(dates, reply)
    plt.plot(dates, noreply)

    plt.legend(['wszystkie komentarze', 'odpowiedzi na komentarze', 'bezpośrednie komentarze do posta'], loc='upper left')
    # plt.yscale('log', nonposy='clip')
    plt.xticks(rotation=45)

    every_nth = 6
    for n, label in enumerate(ax.xaxis.get_ticklabels()):
        if n % every_nth != 0:
            label.set_visible(False)

    plt.savefig('charts/comment_frequency')
    plt.show()


def plot_user_relation_hist(id):
    a = load_obj("authors")
    d = load_obj('user_post_100')
    l = []
    for u in d[id]:
        l.append(d[id][u]['percentage_of_commented_posts'])

    num_bins = 20
    plt.title("Histogram - % skomentowanych postów - {}".format(a[id]))
    plt.xlabel("% skomentowanych postów")
    plt.ylabel("Liczba użytkowników")
    # plt.yscale('log', nonposy='clip')
    plt.hist(l, num_bins, facecolor='blue', alpha=0.5)
    plt.savefig('charts/user_relation_{}'.format(a[id]))
    plt.show()


def sowiniec_chart():
    s = load_obj("sowiniec_godziemba")
    g = load_obj("godziemba_sowiniec")
    sp = load_obj("sowiniec_posts")
    gp = load_obj("godziemba_posts")

    s_ratio = []
    g_ratio = []

    for i in range(len(s)):
        if sp[i] == 0:
            s_ratio.append(0)
        else:
            s_ratio.append(s[i]/sp[i])
        if gp[i] == 0:
            g_ratio.append(0)
        else:
            g_ratio.append(g[i]/gp[i])
        if s_ratio[i] > 1:
            s_ratio[i] = 1
        if g_ratio[i] > 1:
            g_ratio[i] = 1

    dates = []
    for i in range(2008, 2014):
        for j in range(1, 13):
            if i == 2013 and j == 7:
                break
            date = "0{}-{}".format(j, i)
            dates.append(date)

    fig, ax = plt.subplots(figsize=(10, 6))
    # plt.title('Relacja użytkowników Sowiniec - Godziemba')
    plt.ylabel('Stosunek liczby komentarzy do liczby postów użytkownika')

    plt.plot(dates, s_ratio)
    plt.plot(dates, g_ratio)

    plt.legend(['komentarze otrzymane przez Sowińca od Godziemby', 'komentarze otrzymane przez Godziembę od Sownińca'],
               loc='lower right')
    # plt.yscale('log', nonposy='clip')
    plt.xticks(rotation=45)

    every_nth = 6
    for n, label in enumerate(ax.xaxis.get_ticklabels()):
        if n % every_nth != 0:
            label.set_visible(False)

    plt.savefig('charts/sowiniec_godziemba.png')
    plt.show()


def plot_relation_count():
    d = load_obj("comment_relation_count")
    plt.yscale('log', nonposy='clip')
    bars = plt.bar(list(d.keys()), d.values(), color='g')
    plt.gca().set_xticks(list(d.keys()))
    plt.title("Liczba relacji między użytkownikami wg podziału 0-10")
    plt.xlabel("Poziom relacji")
    plt.ylabel("Liczba relacji")
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + 0.4, yval + 0.005, yval, ha='center')
    plt.savefig("charts/comment_relation_count")
    plt.show()


# plot_relation_count()

def plot_user_interactions_histogram():
    d = load_obj("all_users_posts_comment_count")
    l = []
    for u in d:
        l.append(d[u])

    # num_bins = 20
    plt.title("Histogram - liczba wpisów użytkowników")
    plt.xlabel("Liczba wpisów")
    plt.ylabel("Liczba użytkowników")
    plt.yscale('log', nonposy='clip')
    plt.hist(l, 200, facecolor='blue', alpha=0.5)
    plt.savefig('charts/user_interactions')
    plt.show()


def plot_influence_hist():
    d = load_obj("influence_static")
    l = []
    for u in d:
        l.append(d[u])

    # num_bins = 20
    plt.title("Histogram - wpływowość użytkowników")
    plt.xlabel("Miara wpływowości")
    plt.ylabel("Liczba użytkowników")
    plt.yscale('log', nonposy='clip')
    plt.hist(l, 200, facecolor='blue', alpha=0.5)
    plt.savefig('charts/influence_static')
    plt.show()

import pandas as pd

def plot_relation_count():
    df = pd.read_csv("files/clustering_all_active_all.csv")
    d = df.groupby(['slot', 'relation'])['relation'].count()

    print(d)

    weak_p = []
    avg_p = []
    big_p = []
    for i in range(1, 65):
        weak_p.append(d[(i, 1)])
        avg_p.append(d[(i, 2)])
        if 3 not in d[i]:
            big_p.append(0)
        else:
            big_p.append(d[(i, 3)])
    print(big_p)

    dates = []
    for i in range(2008, 2014):
        for j in range(1, 13):
            if i == 2013 and j == 5:
                break
            date = "0{}-{}".format(j, i)
            dates.append(date)

    fig, ax = plt.subplots(figsize=(10, 6))
    # plt.title('Liczba użytkowników wg progów wpływowości w zależności od miesiąca')
    plt.ylabel('Liczba użytkowników')
    plt.plot(dates, weak_p)
    plt.plot(dates, avg_p)
    plt.plot(dates, big_p)
    plt.legend(['słaba relacja', 'średnia siła relacji', 'silna relacja'], loc='upper left')
    # plt.yscale('log', nonposy='clip')
    plt.xticks(rotation=45)

    every_nth = 6
    for n, label in enumerate(ax.xaxis.get_ticklabels()):
        if n % every_nth != 0:
            label.set_visible(False)

    plt.savefig('charts/relation_count.png')
    plt.show()

    print(sum(weak_p) / len(weak_p))
    print(sum(avg_p) / len(avg_p))
    print(sum(big_p) / len(big_p))

# plot_relation_count()


def plot_influence_count():
    data = load_obj("influence_rmi_by_month")
    weak_p = []
    avg_p = []
    big_p = []
    for i in range(0, 66):
        c_w = 0
        c_a = 0
        c_b = 0
        d = data[i]
        for user in d:
            if d[user]['infuence_w'] >= 15:
                c_b += 1
            elif 5 <= d[user]['infuence_w'] < 15:
                c_a += 1
            elif 2 <= d[user]['infuence_w'] < 5:
                c_w += 1
        weak_p.append(c_w)
        avg_p.append(c_a)
        big_p.append(c_b)

    dates = []
    for i in range(2008, 2014):
        for j in range(1, 13):
            if i == 2013 and j == 7:
                break
            date = "0{}-{}".format(j, i)
            dates.append(date)

    fig, ax = plt.subplots(figsize=(10, 6))
    # plt.title('Liczba użytkowników wg progów wpływowości w zależności od miesiąca')
    plt.ylabel('Liczba użytkowników')
    plt.plot(dates, weak_p)
    plt.plot(dates, avg_p)
    plt.plot(dates, big_p)
    plt.legend(['słaba wpływowość', 'średnia wpływowość', 'duża wpływowość'], loc='upper left')
    # plt.yscale('log', nonposy='clip')
    plt.xticks(rotation=45)

    every_nth = 6
    for n, label in enumerate(ax.xaxis.get_ticklabels()):
        if n % every_nth != 0:
            label.set_visible(False)

    plt.savefig('charts/influence_count.png')
    plt.show()

    print(sum(weak_p) / len(weak_p))
    print(sum(avg_p) / len(avg_p))
    print(sum(big_p) / len(big_p))






# plot_stat_results_by_year()



# plot_influence_hist()
# plot_influence_count()
# plot_active_popular_users_weights("active_popularity_reply_by_month")

# print(math.ceil(2.5))
# plot_user_interactions_histogram()
# def plot_users_stats_chart():



# plot_influence_histogram()
# plot_users_by_month()
# plot_active_users_by_month()
# d = load_obj("active_popularity_65")
# print_dict(d)
# plot_active_popular_users("popularity/active_popularity")
# # plot_post_count_by_month()
# plot_active_popular_users_weights("active_popularity_by_month")
# plot_popularity_histograms("active_popularity_reply")
# plot_popularity_max_and_avg("active_popularity_noreply_by_month")
# plot_popular_users_posts_avg("active_popularity_noreply")
# plot_comments_count()
# plot_user_relation_hist(1991)
# sowiniec_chart()
plot_popularity_count("active_popularity_by_month")

# plot_popularity_histograms("active_popularity_reply_by_month")