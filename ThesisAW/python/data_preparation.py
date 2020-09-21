import pickle
import pprint
from collections import OrderedDict
import datetime
from dateutil.relativedelta import relativedelta
import numpy as np
import pandas as pd


def save_obj(obj, name):
    with open('obj/' + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)


def print_dict(dict):
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(dict)


def delete_same_user_interaction():
    user_relation = load_obj("user_relation")
    for comment_author in list(user_relation):
        for post_author in list(user_relation[comment_author]):
            if comment_author == post_author:
                user_relation[comment_author].pop(post_author, None)


def create_post_author_comment_author_interaction_dict():
    user_relation = load_obj("user_relation")
    user_post = {}
    for comment_author in list(user_relation):
        for post_author in list(user_relation[comment_author]):
            if post_author not in user_post:
                user_post[post_author] = {}
            if comment_author not in user_post[post_author]:
                user_post[post_author][comment_author] = {}
            user_post[post_author][comment_author]["all_interactions"] = user_relation[comment_author][post_author]["all_interactions"]
            user_post[post_author][comment_author]["commented_posts"] = user_relation[comment_author][post_author]["commented_posts"]
    save_obj(user_post, "user_post")


def create_post_user_stats_dict():
    user_post = load_obj("user_post")
    post_count = load_obj("post_count")
    post_user_dict = {}

    for post_author in user_post:
        post_user_dict[post_author] = {}
        counter_10 = 0
        counter_5 = 0
        counter_20 = 0
        unique_users_counter = 0
        all_inter_sum = 0
        for comment_author in user_post[post_author]:
            unique_users_counter += 1
            all_inter_sum += user_post[post_author][comment_author]['all_interactions']
            if user_post[post_author][comment_author]['commented_posts'] >= 10:
                counter_10 += 1
            if user_post[post_author][comment_author]['commented_posts'] >= 20:
                counter_20 += 1
            if user_post[post_author][comment_author]['commented_posts'] >= 5:
                counter_5 += 1
        post_user_dict[post_author]["followers_5"] = counter_5
        post_user_dict[post_author]["followers_10"] = counter_10
        post_user_dict[post_author]["followers_20"] = counter_20
        post_user_dict[post_author]["unique_users_counter"] = unique_users_counter
        post_user_dict[post_author]["all_inter_sum"] = all_inter_sum
        for usr in post_count:
            if usr == post_author:
                post_user_dict[post_author]["post_count"] = post_count[usr]
        save_obj(post_user_dict, "post_user_dict")


def create_user_dict():
    user_post = load_obj("user_post")
    user_relation = load_obj("user_relation")
    user = {}
    for post_author in user_post:
        user[post_author] = {}
        followers = []
        followers_count = 0
        for comment_author in user_post[post_author]:
            followers.append(comment_author)
            followers_count += 1
        following = []
        following_count = 0
        if post_author in user_relation:
            for post_user in user_relation[post_author]:
                following.append(post_user)
                following_count += 1
        user[post_author]["followers"] = followers
        user[post_author]["followers_count"] = followers_count
        user[post_author]["following"] = following
        user[post_author]["following_count"] = following_count
    for comment_author in user_relation:
        if comment_author not in user:
            user[comment_author] = {}
            user[comment_author]["followers"] = []
            user[comment_author]["followers_count"] = 0
            following_l = []
            following_count_2 = 0
            for post_user in user_relation[comment_author]:
                following_l.append(post_user)
                following_count_2 += 1
            user[comment_author]["following"] = following_l
            user[comment_author]["following_count"] = following_count_2
    save_obj(user, "user")


def add_tweets_to_user():
    user_comment_count = load_obj("user_comment_count")
    post_count_dict = load_obj("post_count")
    user = load_obj("user")
    for u in user:
        comment_count = 0
        post_count = 0
        if u in user_comment_count:
            comment_count = user_comment_count[u]
        if u in post_count_dict:
            post_count = post_count_dict[u]
        user[u]["tweets"] = comment_count + post_count
    save_obj(user, "user")


def get_authors_ids(authors_list):
    authors = load_obj("authors")
    authors_ids = []
    for author in authors_list:
        authors_ids.append(list(authors.keys())[list(authors.values()).index(author)])
    return authors_ids


def add_weights_to_popular_dicts(dict):
    weights = [1.0, 0.84, 0.68, 0.52, 0.36, 0.2]
    popularity_by_user = {}
    for i in range(0, 66):
        d = load_obj(dict + "_{}".format(i))
        for user in d:
            d[user]['popularity_w'] = float(d[user]['popularity'])
            if user not in popularity_by_user:
                popularity_by_user[user] = []       #Wszyscy użytkownicy w zbiorze
        save_obj(d, dict + "_{}".format(i))

    for i in range(0, 66):
        d = load_obj(dict + "_{}".format(i))
        for user in popularity_by_user:
            if user not in d:
                popularity_by_user[user].append(0)
            else:
                popularity_by_user[user].append(d[user]['popularity_w'])

    c = 1
    for i in range(1, 66):
        d = load_obj(dict + "_{}".format(i))
        for user in d:
            sum = 0
            for k in range(c):
                sum += popularity_by_user[user][i - k] * weights[k]
            d[user]['popularity_w'] = sum
        c += 1
        if c >= 6:
            c = 6
        if i == 65:
            print_dict(d)
        save_obj(d, dict + "_{}".format(i))


def create_comment_relation_user_dict():
    comm_rel = load_obj("comment_relation")
    d = {}
    for c in comm_rel:
        parent = c[1]
        child = c[0]
        if parent not in d:
            d[parent] = {}
            d[parent]["replies"] = 0
        if child not in d[parent]:
            d[parent][child] = 1
        else:
            d[parent][child] += 1
        d[parent]["replies"] += 1
    print(len(d))
    for par in list(d):
        if d[par]["replies"] < 100:
            d.pop(par, None)
    for p in d:
        for c in d[p]:
            if c != "replies":
                d[p][c] = round(d[p][c] * 100 / d[p]["replies"], 3)
    save_obj(d, "comment_relation_100comments")
    # print_dict(d[1991])


def calculate_relation(p):
    rel = 0
    if p < 1:
        rel = 0
    elif 1 <= p < 10:
        rel = 1
    elif 11 <= p < 20:
        rel = 2
    elif 21 <= p < 30:
        rel = 3
    elif 31 <= p < 40:
        rel = 4
    elif 41 <= p < 50:
        rel = 5
    elif 51 <= p < 60:
        rel = 6
    elif 61 <= p < 70:
        rel = 7
    elif 71 <= p < 80:
        rel = 8
    elif 81 <= p < 90:
        rel = 9
    elif 91 <= p <= 100:
        rel = 10
    return rel


def calculate_idk():
    d = load_obj("user_post")
    a100 = load_obj("authors100")
    rel_count = {}
    for i in range(0, 11):
        rel_count[i] = 0
    for post_author in d:
        if post_author in a100:
            for comment_author in d[post_author]:
                if comment_author in a100:
                    rel = d[post_author][comment_author]['percentage_of_commented_posts']
                    rel = calculate_relation(rel)
                    if comment_author in d:
                        if post_author in d[comment_author]:
                            rel2 = d[comment_author][post_author]['percentage_of_commented_posts']
                            rel2 = calculate_relation(rel2)
                        else:
                            rel2 = 0
                    else:
                        rel2 = 0
                    relation = (rel + rel2) // 2
                    rel_count[relation] += 1
    for r in rel_count:
        rel_count[r] = rel_count[r]//2
    save_obj(rel_count, "relation_count")
    # print_dict(rel_count)
    # plt.bar(list(dictionary.keys()), dictionary.values(), color='g')
    # plt.show()

def calculate_idk2():
    d = load_obj("comment_relation_100comments")
    rel_count = {}
    for i in range(0, 11):
        rel_count[i] = 0
    for post_author in d:
        for comment_author in d[post_author]:
            if comment_author != "replies":
                rel = d[post_author][comment_author]
                rel = calculate_relation(rel)
                if comment_author in d:
                    if post_author in d[comment_author]:
                        rel2 = d[comment_author][post_author]
                        rel2 = calculate_relation(rel2)
                    else:
                        rel2 = 0
                else:
                    rel2 = 0
                relation = (rel + rel2) // 2
                rel_count[relation] += 1
    for r in rel_count:
        rel_count[r] = rel_count[r]//2
    print(rel_count)
    save_obj(rel_count, "comment_relation_count")


def create_active_users_dict():
    c = load_obj("comment_count")
    p = load_obj("post_count")
    d = {}
    for ca in c:
        d[ca] = c[ca]
    for pa in p:
        if pa not in d:
            d[pa] = p[pa]
        else:
            d[pa] += p[pa]
    save_obj(d, "all_users_posts_comment_count")
    print(len(d))
    for i in list(d):
        if d[i] < 66:
            d.pop(i, None)
    print(len(d))
    save_obj(d, "active_users_posts_comment_count")


def create_clustering_datasets_by_month():
    users = load_obj("active_users_posts_comment_count")
    u = load_obj("huge_dict")
    # ubm = load_obj("users_by_month")
    datee = datetime.date(2008, 1, 1)
    start_date = datee
    d = {}
    for j in range(0, 66):
        user_interaction = load_obj("user_interaction_{}".format(j))
        popularity = load_obj("popularity/active_popularity_noreply_{}".format(j))
        # users_count = ubm[j]
        d[j] = []
        # d = load_obj(dict + "_{}".format(i))
        addMonths = relativedelta(months=j + 1)
        end_date = datee + addMonths
        print(j)
        users_array = []
        for user in users:
        # if user is not None:
            # print(user)
            # print_dict(u[user])
            # user_list = []  # post count, interaction_count
            # user_list.append(user)
            post_ids = []
            for i, date in enumerate(u[user]["posts"]["dates"]):
                # print(date.date())
                if start_date <= date.date() < end_date:
                    post_ids.append(i)

            post_count = len(post_ids)

            # post_interaction_count = 0
            # for i in post_ids:
            #     post_interaction_count += u[user]["posts"]["interactions"][i]

            if user in user_interaction:
                post_interaction_count = len(user_interaction[user])
                user_interaction_received = len(set(user_interaction[user]))
            else:
                post_interaction_count = 0
                user_interaction_received = 0

            comment_ids = []
            interacted_with = []
            for i, date in enumerate(u[user]["commnets"]["dates"]):
                if start_date <= date.date() < end_date:
                    comment_ids.append(i)
                    interacted_with.append(u[user]["commnets"]["interactedWith"][i])
                    # print(date)

            interaction_count = len(set(interacted_with))
            comment_count = len(comment_ids)

            if user in popularity:
                popul = popularity[user]["popularity_w"]
            else:
                popul = 0

            popular_count = 0
            for id in set(interacted_with):
                if id in popularity:
                    if float(popularity[id]["popularity_w"]) > 0.2:
                        popular_count += 1



            # comm_interaction_count = 0
            # for i in comment_ids:
            #     comm_interaction_count += u[user]["commnets"]["interactions"][i]

            # popularity = (interaction_count + comm_interaction_count) / users_count
            user_list = [user, post_count, comment_count, popular_count, post_interaction_count, interaction_count, user_interaction_received, popul]
            # print(user_list)
            users_array.append(user_list)
        d[j] = np.array(users_array)
        # print(d[i])
        start_date = end_date
    # print_dict(d)
    save_obj(d, "data_by_month_for_clustering")
    # for i in d:
    #     kmeans = KMeans(n_clusters=5, random_state=0).fit(d[i])
    #     print(kmeans.labels_)
    #     print(kmeans.cluster_centers_)


def create_influence_dataset():
    users = load_obj("active_users_posts_comment_count")
    replies = load_obj("replies_count_by_user_by_month")
    comment_count = load_obj("comment_written_by_user_by_month")
    d = {}
    for j in range(0, 66):
        d[j] = {}
        user_interaction = load_obj("user_interaction_{}".format(j))
        post_count = load_obj("posts_by_month/post_count_by_month_{}".format(j))
        reply = replies[j]
        for user in users:
            d[j][user] = {}
            if user in user_interaction:
                post_interaction_count = len(user_interaction[user])
            else:
                post_interaction_count = 0
            if user in reply:
                comment_interaction_count = reply[user]
            else:
                comment_interaction_count = 0
            if user in comment_count[j]:
                comments_written = comment_count[j][user]
            else:
                comments_written = 0
            if user in post_count:
                post_written = post_count[user]
            else:
                post_written = 0

            posts_comments_written = post_written + comments_written
            if posts_comments_written == 0:
                rmi = 0
            else:
                rmi = round((post_interaction_count + comment_interaction_count) / (post_written + comments_written), 3)

            d[j][user]["influence"] = float(rmi)
            d[j][user]["influence_w"] = 0
    save_obj(d, "influence_rmi_by_month")



def create_user_pairs_id_dataset():
    comment_relation = load_obj("user_comment_relation_by_month")
    comment_post_relation = load_obj("user_comment_post_relation_by_month")

    d = {}
    id = 0

    for slot in comment_post_relation:
        for u1 in comment_post_relation[slot]:
            for u2 in comment_post_relation[slot][u1]:
                if u1 != u2:
                    if (u1, u2) not in d and (u2, u1) not in d:
                        d[(u1, u2)] = id
                        d[(u2, u1)] = id
                        id += 1
                    elif (u1, u2) in d and (u2, u1) not in d:
                        d[(u2, u1)] = d[(u1, u2)]
                    elif (u1, u2) not in d and (u2, u1) in d:
                        d[(u1, u2)] = d[(u2, u1)]

    for slot in comment_relation:
        for u1 in comment_relation[slot]:
            for u2 in comment_relation[slot][u1]:
                if u1 != u2:
                    if (u1, u2) not in d and (u2, u1) not in d:
                        d[(u1, u2)] = id
                        d[(u2, u1)] = id
                        id += 1
                    elif (u1, u2) in d and (u2, u1) not in d:
                        d[(u2, u1)] = d[(u1, u2)]
                    elif (u1, u2) not in d and (u2, u1) in d:
                        d[(u1, u2)] = d[(u2, u1)]

    # print_dict(d)
    save_obj(d, "user_pairs_id")
    print(d[(657, 1991)])
    print(d[(1991, 657)])


def refactor_relation_dicts():
    # comment_relation = load_obj("user_comment_relation_by_month")
    comment_post_relation = load_obj("user_comment_post_relation_by_month_fixed")

    comment_post_relation_ref = {}
    comment_relation_ref = {}

    for slot in comment_post_relation:
        comment_post_relation_ref[slot] = {}
        for u1 in comment_post_relation[slot]:      # u1 - post author
            comment_post_relation_ref[slot][u1] = {}
            for u2 in comment_post_relation[slot][u1]:  # u2 - comment author
                # print("{} - {} - {}".format(slot, u1, u2))
                if u2 not in comment_post_relation_ref[slot][u1]:
                    comment_post_relation_ref[slot][u1][u2] = 1
                else:
                    comment_post_relation_ref[slot][u1][u2] += 1

    save_obj(comment_post_relation_ref, "user_comment_post_relation_by_month_structured_fixed")



def count_relations_by_level_by_user_by_month():
    d = load_obj("data_by_month_user_relation_for_clustering_with_relation_level_fixed")
    active = load_obj("active_users_posts_comment_count")

    r = {}

    for u1 in active:
        if u1 not in r:
            r[u1] = {}
            r[u1]['strong'] = []
            r[u1]['avg'] = []
            r[u1]['weak'] = []
            r[u1]['strong_c'] = 0
            r[u1]['avg_c'] = 0
            r[u1]['weak_c'] = 0
    last_slot = 0

    for i in d:
        u1 = i[0]
        slot = i[3]
        level = i[8]

        if slot == last_slot:
            if u1 in r:
                if level >= 5:
                    r[u1]['strong_c'] += 1
                elif 3 <= level < 5:
                    r[u1]['avg_c'] += 1
                elif 1 <= level < 3:
                    r[u1]['weak_c'] += 1
        else:
            for u in r:
                r[u]['strong'].append(r[u]['strong_c'])
                r[u]['avg'].append(r[u]['avg_c'])
                r[u]['weak'].append(r[u]['weak_c'])
                r[u]['strong_c'] = 0
                r[u]['avg_c'] = 0
                r[u]['weak_c'] = 0

        last_slot = slot

    print_dict(r)


def relations_by_level_by_user_by_month():
    d = load_obj("data_by_month_user_relation_for_clustering_with_relation_level_fixed")

    r = {}

    for i in d:
        u1 = i[0]
        u2 = i[1]
        # if u1 in active and u2 in active:
        if u1 not in r:
            r[u1] = {}
        if u2 not in r[u1]:
            r[u1][u2] = []

    last_slot = 0

    for i in d:
        u1 = i[0]
        u2 = i[1]
        slot = i[3]
        level = i[8]

        if slot == last_slot:
            if level >= 5:
                r[u1][u2].append(3)
            elif 3 <= level < 5:
                r[u1][u2].append(2)
            elif 1 <= level < 3:
                r[u1][u2].append(1)
        else:
            for us_1 in r:
                for us_2 in r[us_1]:
                    if len(r[us_1][us_2]) != slot:
                        r[us_1][us_2].append(0)
        last_slot = slot

    save_obj(r, "relation_level_by_user1_user2_by_month")

    # print_dict(r[1991][1518])


def calculate_aa_score():
    d = {}
    user = load_obj("user_post")
    c = load_obj("comment_count")
    popularity = load_obj("user_popularity_without_comment_replies")
    for u in user:
        aa = 0
        for u2 in user[u]:
            replies_of_i = int(c[u2])
            replies_of_i_to_j = int(user[u][u2]['all_interactions'])
            if u2 in popularity:
                a = float(popularity[u2])
            else:
                a = 0
            aa += a * replies_of_i_to_j / replies_of_i
        d[u] = round(aa, 3)
    save_obj(d, "popularity_aas")


def find_relation_level(data, slot, pair_id, u1, u2, post_count):

    last_p = -1
    counter_p = 0
    last_avg_p = -1
    for i in range(slot-5, slot+1):
        if i >= 0:
            # post_count = load_obj("posts_by_month/post_count_by_month_{}".format(i))

            if pair_id not in data[i]:
                relation_p = 0
            else:
                if u1 not in post_count[i]:
                    u1_u2_p = 0
                else:
                    if u1 == data[i][pair_id]["u1"] and u2 == data[i][pair_id]["u2"]:
                        if data[i][pair_id]["u1_u2_p"] > post_count[i][u1]:
                            print("{} - {} - {} - {} - {}".format(data[i][pair_id]["u1_u2_p"], post_count[i][u1], u1, u2, i))
                        u1_u2_p = data[i][pair_id]["u1_u2_p"] / post_count[i][u1] # u2 - komentujący, u1 - dodający post
                    else:
                        u1_u2_p = 0
                if u1_u2_p > 1:
                    print("U1: {} - {} - {} - {} - {} - {}".format(u1_u2_p, data[i][pair_id]["u1_u2_p"], post_count[i][u1], u1, u2, i))
                if u2 not in post_count[i]:
                    u2_u1_p = 0
                else:
                    if u1 == data[i][pair_id]["u1"] and u2 == data[i][pair_id]["u2"]:
                        u2_u1_p = data[i][pair_id]["u2_u1_p"] / post_count[i][u2]
                    else:
                        u2_u1_p = 0
                if u2_u1_p > 1:
                    print("U2: {} - {} - {} - {} - {} - {}".format(u2_u1_p, data[i][pair_id]["u2_u1_p"], post_count[i][u1], u1, u2, i))

                relation_p = (u1_u2_p + u2_u1_p) / 2

            if last_p != -1:
                if counter_p == 1:
                    avg_p = (relation_p + last_p) / 2
                else:
                    avg_p = (relation_p + last_avg_p) / 2
                last_avg_p = avg_p
            last_p = relation_p
            if counter_p == 0:
                avg_p = relation_p
            counter_p += 1

    if avg_p > 1:
        print("avg: {} - {} - {} - {}".format(avg_p, u1, u2, i))


    return 10*round(avg_p, 3)


def weighted_influence():
    data = load_obj("influence_rmi_by_month")
    for i in range(0, 66):
        for user in data[i]:
            inf = weighted_influence_for_user(data, i, user)
            data[i][user]['infuence_w'] = inf
    save_obj(data, "influence_rmi_by_month")


def weighted_influence_for_user(data, slot, user):
    last_p = -1
    counter_p = 0
    last_avg_p = -1
    for i in range(slot-5, slot+1):
        if i >= 0:
            if user not in data[i]:
                measure = 0
            else:
                measure = data[i][user]['influence']

            if last_p != -1:
                if counter_p == 1:
                    avg_p = (measure + last_p) / 2
                else:
                    avg_p = (measure + last_avg_p) / 2
                last_avg_p = avg_p
            last_p = measure
            if counter_p == 0:
                avg_p = measure
            counter_p += 1

    return round(avg_p, 3)


def weighted_popularity():
    data = load_obj("active_popularity_by_month")
    for i in range(0, 66):
        for user in data[i]:
            pop = weighted_popularity_for_user(data, i, user)
            data[i][user]['popularity_weight'] = pop
    save_obj(data, "active_popularity_by_month")
    print_dict(data[56][1991])


def weighted_popularity_for_user(data, slot, user):
    last_p = -1
    counter_p = 0
    last_avg_p = -1
    for i in range(slot-5, slot+1):
        if i >= 0:
            if user not in data[i]:
                measure = 0
            else:
                measure = float(data[i][user]['popularity'])

            if last_p != -1:
                if counter_p == 1:
                    avg_p = (measure + last_p) / 2
                else:
                    avg_p = (measure + last_avg_p) / 2
                last_avg_p = avg_p
            last_p = measure
            if counter_p == 0:
                avg_p = measure
            counter_p += 1

    return round(avg_p, 3)


def influance_static():
    users = load_obj("active_users_posts_comment_count")
    com_wr = load_obj("comment_written_by_user")
    com_received = load_obj("comments_received_to_posts")
    replies_received = load_obj("replies_count_by_user")
    post_written = load_obj("post_count")

    d = {}
    for u in users:
        if u not in com_received:
            cr = 0
        else:
            cr = com_received[u]
        if u not in replies_received:
            rr = 0
        else:
            rr = replies_received[u]
        if u not in post_written:
            pw = 0
        else:
            pw = post_written[u]
        if u not in com_wr:
            cw = 0
        else:
            cw = com_wr[u]
        d[u] = float(round(((cr + rr) / (pw + cw)), 2))

        if u == 2168:
            print("{} {} {} {}".format(pw, cw, cr, rr))

    save_obj(d, "influence_static")
    ordered = OrderedDict(sorted(d.items(), key=lambda i: i[1]))
    print_dict(ordered)


def create_active_user_measures_levels_by_month():
    popularity = load_obj("active_popularity_by_month")
    influence = load_obj("influence_rmi_by_month")
    relation = load_obj("relation_level_by_user1_user2_by_month")
    active = load_obj("active_users_posts_comment_count")

    d = {}

    for u in active:
        d[u] = {}
        d[u]['popularity'] = []
        d[u]['influence'] = []
        d[u]['relation'] = {}

    for i in range(0, 66):
        for u in d:
            p_level = 0
            if u in popularity[i]:
                if popularity[i][u]['popularity_weight'] >= 0.3:
                    p_level = 3
                elif 0.12 <= popularity[i][u]['popularity_weight'] < 0.3:
                    p_level = 2
                elif 0.05 <= popularity[i][u]['popularity_weight'] < 0.12:
                    p_level = 1
                else:
                    p_level = 0
            d[u]['popularity'].append(p_level)

            i_level = 0
            if u in influence[i]:
                if influence[i][u]['infuence_w'] >= 15:
                    i_level = 3
                elif 5 <= influence[i][u]['infuence_w'] < 15:
                    i_level = 2
                elif 2 <= influence[i][u]['infuence_w'] < 5:
                    i_level = 1
                else:
                    i_level = 0
            d[u]['influence'].append(i_level)

    for u1 in relation:
        for u2 in relation[u1]:
            d[u1]['relation'][u2] = relation[u1][u2]

    save_obj(d, "active_user_measures_levels_by_month")
    print_dict(d[657])


def calc_stability(arr, lvl):
    subsequence_length = 0
    max_longest = 0
    b = True
    for i in arr:
        if i == lvl:
            subsequence_length += 1
        else:
            b = False
            longest = subsequence_length
            subsequence_length = 0
            if longest > max_longest:
                max_longest = longest
    if b:
        max_longest = subsequence_length
    return max_longest


def count_stability_by_year():
    d = load_obj("active_user_measures_levels_by_month")
    for u in d:
        d[u]['popularity_stability'] = []
        d[u]['influence_stability'] = []
        for i in range(5):
            start = 12 * i
            end = start + 12

            year_p = d[u]['popularity'][start:end]
            length = calc_stability(year_p, d[u]['popularity_goals'][i])
            if u == 657:
                print(i)
                print(length)

            if length >= 10:
                stability_p = 3
            elif 6 <= length < 10:
                stability_p = 2
            else:
                stability_p = 1

            if d[u]['popularity_goals'][i] == 0:
                stability_p = 0

            year_p = d[u]['influence'][start:end]
            length = calc_stability(year_p, d[u]['influence_goals'][i])

            if length >= 10:
                stability_i = 3
            elif 6 <= length < 10:
                stability_i = 2
            else:
                stability_i = 1

            if d[u]['influence_goals'][i] == 0:
                stability_i = 0

            d[u]['popularity_stability'].append(stability_p)
            d[u]['influence_stability'].append(stability_i)

    save_obj(d, "active_user_measures_levels_by_month")



def create_full_dataset_by_year():
    users = load_obj("active_users_posts_comment_count")
    replies = load_obj("replies_count_by_user_by_month")
    # comment_count = load_obj("comment_written_by_user_by_month")
    reply_written = load_obj("reply_comment_written_by_month_by_user")
    noreply_written = load_obj("noreply_comment_written_by_month_by_user")
    u = load_obj("huge_dict")
    d = {}
    u_i = {}
    p_c = {}
    for i in range(0, 66):
        u_i[i] = load_obj("user_interaction_{}".format(i))
        p_c[i] = load_obj("posts_by_month/post_count_by_month_{}".format(i))

    for i, user in enumerate(users):
        print(i)
        d[user] = {}
        d[user]['post_written'] = []
        # d[user]['comments_written'] = []
        d[user]['comments_noreply_written'] = []
        d[user]['comments_reply_written'] = []
        d[user]['post_interaction_count'] = []
        d[user]['comment_interaction_count'] = []
        d[user]['interacted_with'] = []
        d[user]['user_interaction_received'] = []
        datee = datetime.date(2008, 1, 1)
        start_date = datee
        for j in range(0, 66):
            addMonths = relativedelta(months=j + 1)
            end_date = datee + addMonths
            # user_interaction = load_obj("user_interaction_{}".format(j))
            # post_count = load_obj("posts_by_month/post_count_by_month_{}".format(j))
            user_interaction = u_i[j]
            post_count = p_c[j]
            reply = replies[j]

            if user in user_interaction:
                post_interaction_count = len(user_interaction[user])
                user_interaction_received = len(set(user_interaction[user]))
            else:
                post_interaction_count = 0
                user_interaction_received = 0
            if user in reply:
                comment_interaction_count = int(reply[user])
            else:
                comment_interaction_count = 0
            if user in noreply_written[j]:
                comments_noreply_written = noreply_written[j][user]
                comment_interaction_count = comment_interaction_count / comments_noreply_written
            else:
                comments_noreply_written = 0
            if user in reply_written[j]:
                comments_reply_written = reply_written[j][user]
            else:
                comments_reply_written = 0
            if user in post_count:
                post_written = post_count[user]
                post_interaction_count = post_interaction_count / post_written
            else:
                post_written = 0
                post_interaction_count = 0

            comment_ids = []
            interacted_with = []
            for i, date in enumerate(u[user]["commnets"]["dates"]):
                if start_date <= date.date() < end_date:
                    comment_ids.append(i)
                    interacted_with.append(u[user]["commnets"]["interactedWith"][i])

            interaction_count = len(set(interacted_with))

            start_date = end_date

            d[user]['post_written'].append(post_written)
            # d[user]['comments_written'].append(comments_written)
            d[user]['comments_noreply_written'].append(comments_noreply_written)
            d[user]['comments_reply_written'].append(comments_reply_written)
            d[user]['post_interaction_count'].append(post_interaction_count)
            d[user]['comment_interaction_count'].append(comment_interaction_count)
            d[user]['interacted_with'].append(interaction_count)
            d[user]['user_interaction_received'].append(user_interaction_received)
    save_obj(d, "user_stats_by_month_fix")


def create_dataset_for_clustering_by_month():
    users = load_obj("active_users_posts_comment_count")
    replies = load_obj("replies_count_by_user_by_month")
    # comment_count = load_obj("comment_written_by_user_by_month")
    reply_written = load_obj("reply_comment_written_by_month_by_user")
    noreply_written = load_obj("noreply_comment_written_by_month_by_user")
    u = load_obj("huge_dict")
    goals = load_obj("active_user_measures_levels_by_month")
    d = {}
    u_i = {}
    p_c = {}
    for i in range(0, 66):
        u_i[i] = load_obj("user_interaction_{}".format(i))
        p_c[i] = load_obj("posts_by_month/post_count_by_month_{}".format(i))

    for i, user in enumerate(users):
        print(i)
        d[user] = {}
        d[user]['post_written'] = []
        # d[user]['comments_written'] = []
        d[user]['comments_noreply_written'] = []
        d[user]['comments_reply_written'] = []
        d[user]['post_interaction_count'] = []
        d[user]['comment_interaction_count'] = []
        d[user]['interacted_with'] = []
        d[user]['user_interaction_received'] = []
        d[user]['user_interaction_received_last_slot_percentage'] = []
        d[user]['interacted_with_last_slot_percentage'] = []
        d[user]['relations_count_with_popular'] = []
        d[user]['relations_count_with_influential'] = []
        d[user]['weak_relations_count'] = []
        d[user]['avg_relations_count'] = []
        d[user]['strong_relations_count'] = []
        d[user]['last_popularity'] = []
        d[user]['last_influence'] = []

        datee = datetime.date(2008, 1, 1)
        start_date = datee
        last_interacted_with = []
        for j in range(0, 65):
            addMonths = relativedelta(months=j + 1)
            end_date = datee + addMonths
            # user_interaction = load_obj("user_interaction_{}".format(j))
            # post_count = load_obj("posts_by_month/post_count_by_month_{}".format(j))
            user_interaction = u_i[j]
            post_count = p_c[j]
            reply = replies[j]

            if user in user_interaction:
                post_interaction_count = len(user_interaction[user])
                user_interaction_received = len(set(user_interaction[user]))
                if j == 0:
                    user_interaction_received_last_slot_percentage = 0
                else:
                    if user not in u_i[j - 1]:
                        user_interaction_received_last_slot_percentage = 0
                    else:
                        user_interaction_last = u_i[j - 1][user]
                        ur_last_count = len(set(u_i[j - 1][user]))
                        ur_remain_count = len(set(user_interaction[user]) & set(user_interaction_last))
                        if ur_last_count == 0:
                            user_interaction_received_last_slot_percentage = 0
                        else:
                            user_interaction_received_last_slot_percentage = round(ur_remain_count / ur_last_count, 2) * 100
            else:
                post_interaction_count = 0
                user_interaction_received = 0
                user_interaction_received_last_slot_percentage = 0
            if user in reply:
                comment_interaction_count = int(reply[user])
            else:
                comment_interaction_count = 0
            if user in noreply_written[j]:
                comments_noreply_written = noreply_written[j][user]
                comment_interaction_count = comment_interaction_count / comments_noreply_written
            else:
                comments_noreply_written = 0
            if user in reply_written[j]:
                comments_reply_written = reply_written[j][user]
            else:
                comments_reply_written = 0
            if user in post_count:
                post_written = post_count[user]
                post_interaction_count = post_interaction_count / post_written
            else:
                post_written = 0
                post_interaction_count = 0

            comment_ids = []
            interacted_with = []

            for i, date in enumerate(u[user]["commnets"]["dates"]):
                if start_date <= date.date() < end_date:
                    comment_ids.append(i)
                    interacted_with.append(u[user]["commnets"]["interactedWith"][i])

            interaction_count = len(set(interacted_with))

            if j == 0:
                interacted_with_last_slot_percentage = 0
            else:
                interacted_with_remain = len(set(interacted_with) & set(last_interacted_with))
                l_count = len(set(last_interacted_with))
                if l_count == 0:
                    interacted_with_last_slot_percentage = 0
                else:
                    interacted_with_last_slot_percentage = round(interacted_with_remain / l_count, 2) * 100

            last_interacted_with = interacted_with

            start_date = end_date

            popular_count = 0
            for u2 in goals[user]['relation']:
                # print('{} - {} - {} - {}'.format(user, u2, j, len(goals[user]['relation'][u2])))
                if goals[user]['relation'][u2][j] != 0:
                    if goals[u2]['popularity'][j] != 0:
                        popular_count += 1

            influential_count = 0
            for u2 in goals[user]['relation']:
                if goals[user]['relation'][u2][j] != 0:
                    if goals[u2]['influence'][j] != 0:
                        influential_count += 1


            weak_count = 0
            avg_count = 0
            strong_count = 0
            for u2 in goals[user]['relation']:
                if goals[user]['relation'][u2][j] == 1:
                    weak_count += 1
                elif goals[user]['relation'][u2][j] == 2:
                    avg_count += 1
                elif goals[user]['relation'][u2][j] == 3:
                    strong_count += 1

            if j == 0:
                last_popularity = 1
                last_influence = 1
                # na tym samym poziomie w kolejnym slocie - 1, zmiana - 0
            else:
                if goals[user]['popularity'][j-1] == goals[user]['popularity'][j]:
                    last_popularity = 1
                else:
                    last_popularity = 0

                if goals[user]['influence'][j-1] == goals[user]['influence'][j]:
                    last_influence = 1
                else:
                    last_influence = 0



            d[user]['post_written'].append(post_written)
            d[user]['comments_noreply_written'].append(comments_noreply_written)
            d[user]['comments_reply_written'].append(comments_reply_written)
            d[user]['post_interaction_count'].append(post_interaction_count)
            d[user]['comment_interaction_count'].append(comment_interaction_count)
            d[user]['interacted_with'].append(interaction_count)
            d[user]['user_interaction_received'].append(user_interaction_received)
            d[user]['interacted_with_last_slot_percentage'].append(interacted_with_last_slot_percentage)
            d[user]['user_interaction_received_last_slot_percentage'].append(user_interaction_received_last_slot_percentage)
            d[user]['relations_count_with_popular'].append(popular_count)
            d[user]['relations_count_with_influential'].append(influential_count)

            d[user]['last_popularity'].append(last_popularity)
            d[user]['last_influence'].append(last_influence)

            d[user]['weak_relations_count'].append(weak_count)
            d[user]['avg_relations_count'].append(avg_count)
            d[user]['strong_relations_count'].append(strong_count)

    c = []
    for i in range(0, 65):
        print(i)
        for user in d:
            y_post_written = d[user]['post_written'][i]
            y_comments_noreply_written = d[user]['comments_noreply_written'][i]
            y_comments_reply_written = d[user]['comments_reply_written'][i]
            y_post_interaction_count = d[user]['post_interaction_count'][i]
            y_comment_interaction_count = d[user]['comment_interaction_count'][i]
            y_interacted_with = d[user]['interacted_with'][i]
            y_user_interaction_received = d[user]['user_interaction_received'][i]

            y_interacted_with_last_slot_percentage = d[user]['interacted_with_last_slot_percentage'][i]
            y_user_interaction_received_last_slot_percentage = d[user]['user_interaction_received_last_slot_percentage'][i]

            relations_count_with_popular = d[user]['relations_count_with_popular'][i]
            relations_count_with_influential = d[user]['relations_count_with_influential'][i]

            weak_relations_count = d[user]['weak_relations_count'][i]
            avg_relations_count = d[user]['avg_relations_count'][i]
            strong_relations_count = d[user]['strong_relations_count'][i]

            popularity = goals[user]['popularity'][i]
            influence = goals[user]['influence'][i]

            relation = 0
            if weak_relations_count > 0:
                relation = 1
            if avg_relations_count > 0:
                relation = 2
            if strong_relations_count > 0:
                relation = 3

            c.append(
                [user, i, y_post_written, y_comments_noreply_written, y_comments_reply_written, y_post_interaction_count,
                 y_comment_interaction_count, y_interacted_with, y_user_interaction_received, y_interacted_with_last_slot_percentage, y_user_interaction_received_last_slot_percentage, relations_count_with_popular,
                 relations_count_with_influential, weak_relations_count, avg_relations_count, strong_relations_count, popularity, influence, relation])
    all = np.array(c)
    pd.DataFrame(all).to_csv("files/clustering_all_active_all.csv")
