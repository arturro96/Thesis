import psycopg2

from queries import POPULARITY_ALL, POPULARITY_WITHOUT_COMMENTS_REPLIES, POPULARITY_ONLY_COMMENTS_REPLIES, \
    POPULARITY_UNIQUE_USER_COMMENTS, POPULARITY_USER_UNIQUE_COMMENTS_REPLIES, COMMENT_POST_AUTHOR_RELATION, POST_COUNT, \
    AUTHORS, COMMENT_COUNT, FIRST_COMMENT_DATE, FIRST_POST_DATE, AUTHOR_COMMENT_DATE, AUTHOR_POST_DATE, COMMENT_USER_RELATION, COMMENT_AUTHORS_RELATION, AUTHOR_POST_COUNT_100, COMMENT_WORD_COUNT, POST_WORD_COUNT

from data_preparation import save_obj, load_obj, print_dict


AUTHORS_COUNT = 31750


class PsqlManager:
    def __init__(self):
        self.query = POST_WORD_COUNT

    def run(self):
        try:
            connection = psycopg2.connect(user="sna_user",
                                          password="sna_password",
                                          host="127.0.0.1",
                                          port="5432",
                                          database="salon24")

            cursor = connection.cursor()
            cursor.execute(self.query)

            self.records = cursor.fetchall()
            d = {}
            for row in self.records:
                d[row[0]] = row[1]
            save_obj(d, "{}".format(self.query))


        except (Exception, psycopg2.Error) as error:
            print("Error while connecting to PostgreSQL", error)
        finally:
            # closing database connection.
            if (connection):
                cursor.close()
                connection.close()
                # print("PostgreSQL connection is closed")

    def process_user_comm_relation(self):
        comm_relation = []
        for row in self.records:
            comment_author_id = row[0]
            parent_comment_author_id = row[1]
            comm_relation.append((comment_author_id, parent_comment_author_id))
        save_obj(comm_relation, "comment_relation")


    def process_user_relation_query(self):
        d = {}
        prev_post_author = self.records[0][3]
        prev_comment_author = self.records[0][0]
        counter = 0
        comments_counter = 0
        for row in self.records:
            # print(str(row[0]) + " - " + str(row[1]) + " - " + str(row[2]) + " - " + str(row[3]) + " - " + str(row[4]))
            comment_author = row[0]
            post_author = row[3]
            comments_under_post = row[4]
            if comment_author == prev_comment_author and post_author == prev_post_author:
                counter += 1
                comments_counter += comments_under_post
            elif (comment_author == prev_comment_author and post_author != prev_post_author) or comment_author != prev_comment_author:
                counter = 1
                comments_counter = comments_under_post
            if comment_author not in d:
                d[comment_author] = {}
            if post_author not in d[comment_author]:
                d[comment_author][post_author] = {}
            d[comment_author][post_author]["commented_posts"] = counter
            d[comment_author][post_author]["all_interactions"] = comments_counter
            prev_post_author = post_author
            prev_comment_author = comment_author
        save_obj(d, "user_relation")

    def process_popularity_query(self):
        popularity_dict = {}
        for row in self.records:
            popularity_dict[row[0]] = str(row[2])
        save_obj(popularity_dict, "user_popularity_without_comment_replies")

    def process_post_count(self):
        post_count = {}
        for row in self.records:
            author_id = row[0]
            count = row[1]
            post_count[author_id] = count
        save_obj(post_count, "post_count")

    def process_authors_query(self):
        authors = {}
        for row in self.records:
            author_id = row[0]
            name = row[1]
            authors[author_id] = name
        save_obj(authors, "authors")

    def process_comment_count_query(self):
        user_comment_count = {}
        for row in self.records:
            author_id = row[0]
            comment_count = row[1]
            user_comment_count[author_id] = comment_count
        save_obj(user_comment_count, "user_comment_count")

    def process_post_frequency_records(self):
        d = {}
        for row in self.records:
            author_id = row[0]
            d[author_id] = {}
            d[author_id]["min"] = row[1]
            d[author_id]["max"] = row[2]
            d[author_id]["quantity"] = row[3]
            d[author_id]["days"] = row[4]
            d[author_id]["years"] = row[5]
            d[author_id]["months"] = row[6]
            if row[4] != 0:
                d[author_id]["days_freq"] = row[3] / row[4]
            if row[6] != 0:
                d[author_id]["months_freq"] = row[3] / row[6]
            if row[5] != 0:
                d[author_id]["years_freq"] = row[3] / row[5]
        save_obj(d, "user_post_frequency")

    def process_first_comment_date_query(self):
        d = {}
        for row in self.records:
            author_id = row[0]
            date = row[1]
            d[author_id] = date
        save_obj(d, "first_comment_date_by_user")

    def process_first_post_date_query(self):
        d = {}
        for row in self.records:
            author_id = row[0]
            date = row[1]
            d[author_id] = date
        save_obj(d, "first_post_date_by_user")

    def process_author_comment_date_query(self):
        l = []
        for row in self.records:
            author_id = row[0]
            comment_id = row[1]
            date = row[2]
            l.append([author_id, comment_id, date])
        save_obj(l, "author_comment_date_list")

    def process_author_post_date_query(self):
        l = []
        for row in self.records:
            author_id = row[0]
            post_id = row[1]
            date = row[2]
            l.append([author_id, post_id, date])
        save_obj(l, "author_post_date_list")

    def process_post_count_by_month(self, i):
        d = {}
        for row in self.records:
            author_id = row[0]
            count = row[1]
            d[author_id] = count
        save_obj(d, "post_count_by_month_{}".format(i))

    def process_pop_query(self, i):
        d = {}
        count = 0
        for row in self.records:
            author_id = row[0]
            name = row[1]
            popularity = row[2]
            if count <= 10:
                print(name + " & " + str(popularity) + "\\\\")
            count += 1
            d[author_id] = {}
            d[author_id]["name"] = name
            d[author_id]["popularity"] = popularity
        save_obj(d, "active_popularity_reply_{}".format(i))

    def process_comment_user_relation_query(self):
        d = {}
        for row in self.records:
            id = row[0]
            date = row[1]
            author_id = row[2]
            parentcomment_id = row[3]
            d[id] = {}
            d[id]["date"] = date
            d[id]["author_id"] = author_id
            d[id]["parentcomment_id"] = parentcomment_id
        save_obj(d, "comment_user_relation")

    def create_huge_dictionary(self):
        d = {}
        for row in self.records:
            author_id = row[0]
            date = row[1]
            interactions = row[2]
            if author_id not in d:
                d[author_id] = {}
                d[author_id]["posts"] = {}
                d[author_id]["posts"]["dates"] = []
                d[author_id]["posts"]["interactions"] = []
                d[author_id]["commnets"] = {}
                d[author_id]["commnets"]["dates"] = []
                d[author_id]["commnets"]["interactedWith"] = []
                d[author_id]["commnets"]["interactions"] = []
            d[author_id]["posts"]["dates"].append(date)
            d[author_id]["posts"]["interactions"].append(interactions)
        save_obj(d, "huge_dict")

    def create_huge_dictionary2(self):
        d = load_obj("huge_dict")
        for row in self.records:
            author_id = row[0]
            date = row[1]
            interactedWith = row[2]
            interactions = row[3]
            if author_id not in d:
                d[author_id] = {}
                d[author_id]["posts"] = {}
                d[author_id]["posts"]["dates"] = []
                d[author_id]["posts"]["interactions"] = []
                d[author_id]["commnets"] = {}
                d[author_id]["commnets"]["dates"] = []
                d[author_id]["commnets"]["interactedWith"] = []
                d[author_id]["commnets"]["interactions"] = []
            d[author_id]["commnets"]["dates"].append(date)
            d[author_id]["commnets"]["interactedWith"].append(interactedWith)
            d[author_id]["commnets"]["interactions"].append(interactions)
        save_obj(d, "huge_dict")

    def create_huge_dictionary3(self):
        d = load_obj("huge_dict")
        for row in self.records:
            author_id = row[0]
            date = row[1]
            interactedWith = row[2]
            if author_id not in d:
                d[author_id] = {}
                d[author_id]["posts"] = {}
                d[author_id]["posts"]["dates"] = []
                d[author_id]["posts"]["interactions"] = []
                d[author_id]["commnets"] = {}
                d[author_id]["commnets"]["dates"] = []
                d[author_id]["commnets"]["interactedWith"] = []
                d[author_id]["commnets"]["interactions"] = []
            d[author_id]["commnets"]["dates"].append(date)
            d[author_id]["commnets"]["interactedWith"].append(interactedWith)
            d[author_id]["commnets"]["interactions"].append(0)
        save_obj(d, "huge_dict")


if __name__ == '__main__':
    psqlManager = PsqlManager()
    psqlManager.run()



