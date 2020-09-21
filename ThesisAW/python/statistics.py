from data_preparation import load_obj, get_authors_ids, save_obj, print_dict
import datetime
import calendar
from dateutil.relativedelta import relativedelta


popular_authors_list1 = ["FREE YOUR MIND", "RENATA RUDECKA-KALINOWSKA", "KRZYSZTOF LESKI", "CEZARY KRYSZTOPA",
                         "SOWINIEC", "STARY", "GRZEGORZ WSZOŁEK - GW1990", "UFKA", "MAREK MIGALSKI", "1MAUD",
                         "PANNA WODZIANNA", "IGOR JANKE", "RYBITZKY", "CORYLLUS", "SEAMAN"]


def calculate_influence():
    post_user_dict = load_obj("post_user_dict")
    users = []
    snp_list = []
    for user in post_user_dict:
        if post_user_dict[user]['followers_20'] != 0:
            ir = post_user_dict[user]['unique_users_counter'] / post_user_dict[user]['followers_20']
        else:
            ir = 0
        rmr = post_user_dict[user]['all_inter_sum'] / post_user_dict[user]['post_count']
        snp = round(((ir + rmr) / 2), 4)
        # print(str(user) + " - " + str(snp))
        users.append(user)
        snp_list.append(snp)
    list1, list2 = zip(*sorted(zip(snp_list, users), reverse=True))
    return list1, list2


def calculate_user_influence(user_id):
    post_user_dict = load_obj("post_user_dict")
    authors = load_obj("authors")
    if user_id in post_user_dict:
        if post_user_dict[user_id]['followers_10'] != 0:
            ir = post_user_dict[user_id]['unique_users_counter'] / post_user_dict[user_id]['followers_10']
        else:
            ir = 0
        rmr = post_user_dict[user_id]['all_inter_sum'] / post_user_dict[user_id]['post_count']
        snp = round(((ir + rmr) / 2), 4)
        print(str(authors[user_id]) + " & " + str(snp) + " &")
        return snp


def calculate_popular_users_influence():
    _, influence_list = calculate_influence()
    authors_ids = get_authors_ids(popular_authors_list1)
    for id in authors_ids:
        calculate_user_influence(id)
        place = get_influence_place(influence_list, id)
        print(str(place))# + "/" + str(len(influence_list)))


def get_influence_place(influence_ids, user):
    for i, id in enumerate(influence_ids):
        if id == user:
            return i


def create_post_user_measures_dict():
    post_user_measures = {}
    authors = load_obj("authors")
    popularity = load_obj("user_popularity_all")
    for id in authors:
        if id in popularity:
            post_user_measures[id] = {}
            post_user_measures[id]["influence"] = calculate_user_influence(id)
            post_user_measures[id]["popularity"] = popularity[id]
    save_obj(post_user_measures, "post_user_measures")


def add_percentage_of_commented_posts_to_dict():
    post_user_dict = load_obj("post_user_dict")
    user_relation = load_obj("user_relation")
    for comment_author in user_relation:
        for post_user in user_relation[comment_author]:
            post_count = post_user_dict[post_user]["post_count"]
            commented_posts = user_relation[comment_author][post_user]["commented_posts"]
            user_relation[comment_author][post_user]["percentage_of_commented_posts"] = round((commented_posts / post_count) * 100, 2)
    save_obj(user_relation, "user_relation")
    # print_dict(user_relation)


def create_user_relation_more_than_100_posts_dict():
    user_relation = load_obj("user_relation")
    post_user_dict = load_obj("post_user_dict")
    user_relation_100_posts = {}
    for comment_author in list(user_relation):
        user_relation_100_posts[comment_author] = {}
        for post_author in list(user_relation[comment_author]):
            if post_user_dict[post_author]["post_count"] >= 100:
                user_relation_100_posts[comment_author][post_author] = {}
                user_relation_100_posts[comment_author][post_author]["percentage_of_commented_posts"] = \
                    user_relation[comment_author][post_author]["percentage_of_commented_posts"]
    save_obj(user_relation_100_posts, "user_relation_100_posts")
    print_dict(user_relation_100_posts)


def add_percentage_of_commented_posts_to_post_user_dict():
    post_user_dict = load_obj("post_user_dict")
    user_post = load_obj("user_post")
    for post_author in user_post:
        post_count = post_user_dict[post_author]["post_count"]
        for comment_author in user_post[post_author]:
            commented_posts = user_post[post_author][comment_author]["commented_posts"]
            user_post[post_author][comment_author]["percentage_of_commented_posts"] = round((commented_posts / post_count) * 100, 2)
    save_obj(user_post, "user_post")
    # print_dict(user_post)


def return_sorted_relation_user_comment_author(user_id):
    user_relation = load_obj("user_relation_100_posts")
    post_users_dict = user_relation[user_id]
    res = sorted(post_users_dict.items(), key=lambda x: x[1]['percentage_of_commented_posts'], reverse=True)
    print_dict(res)


def return_sorted_relation_user_post_author(user_id):
    user_relation = load_obj("user_post")
    post_users_dict = user_relation[user_id]
    res = sorted(post_users_dict.items(), key=lambda x: x[1]['percentage_of_commented_posts'], reverse=True)
    print_dict(res)


def user_first_interaction():
    users = {}
    comment = load_obj("first_comment_date_by_user")
    post = load_obj("first_post_date_by_user")
    for user in comment:
        users[user] = comment[user]
    for user in post:
        if user not in users:
            users[user] = post[user]
        elif post[user] < users[user]:
            users[user] = post[user]
    save_obj(users, "user_first_interaction_date")


def count_users_by_month(i):
    date1 = datetime.date(2008, 1, 1)
    addMonths = relativedelta(months=i)
    date = date1 + addMonths
    count = 0
    for user in u:
        if u[user] < date:
            count += 1
    return count


def users_by_months():
    user_count_by_month = []
    for i in range(1, 67):
        user_count_by_month.append(count_users_by_month(i))
    save_obj(user_count_by_month, "users_by_month")


def find_max_val_and_key(d):
    max_val = 0
    max_val_key = 0
    for user in d:
        if d[user]['all_interactions'] > max_val:
            max_val = d[user]['all_interactions']
            max_val_key = user
    return max_val_key, max_val

def find_max_interaction_of_post_user():
    post_author = load_obj("user_post")
    a = load_obj("authors")
    d = {}

    for user in post_author:
        maxval = find_max_val_and_key(post_author[user])
        d[user] = maxval

    s = list(sorted(d.keys(), key=lambda x: d[x][1], reverse=True))

    # print_dict(s)
    for i in s:
        print(str(a[i]) + " & " + str(a[d[i][0]]) + " & " + str(d[i][1]) + "\\\\")






