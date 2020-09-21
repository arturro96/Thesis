
POPULARITY_ALL = "select posts.author_id, authors.name, " \
           "round(cast(count(comments.post_id) as decimal) / (select count(*) from authors), 2) as popularity " \
           "from comments inner join posts on comments.post_id = posts.id " \
           "inner join authors on posts.author_id = authors.id " \
           "group by posts.author_id, authors.name order by count(comments.post_id) desc"

POPULARITY_WITHOUT_COMMENTS_REPLIES = "select posts.author_id, authors.name, round(cast(count(comments.post_id) " \
                                      "as decimal) / (select count(*) from authors), 2) " \
                                      "as popularity from comments inner join posts on comments.post_id = posts.id" \
                                      " inner join authors on posts.author_id = authors.id " \
                                      "group by posts.author_id, authors.name, comments.parentcomment_id " \
                                      "having comments.parentcomment_id is NULL  order by count(comments.post_id) desc"

POPULARITY_ONLY_COMMENTS_REPLIES = "with firstgroup as (select count(com.parentcomment_id) as count, parent.author_id " \
                                   "as par from comments as com left join comments as parent on " \
                                   "parent.id = com.parentcomment_id group by com.parentcomment_id, parent.author_id) " \
                                   "select authors.id, authors.name, round(cast(sum(count) as decimal) / " \
                                   "(select count(*) from authors), 2) as popularity from firstgroup " \
                                   "inner join authors on par = authors.id group by authors.id, authors.name " \
                                   "order by sum(count) desc"

POPULARITY_UNIQUE_USER_COMMENTS = "select posts.author_id, authors.name, round(cast(count(distinct comments.author_id) " \
                                  "as decimal) / (select count(*) from authors), 2) as popularity from comments " \
                                  "inner join posts on comments.post_id = posts.id " \
                                  "inner join authors on posts.author_id = authors.id " \
                                  "group by posts.author_id, authors.name order by count(distinct comments.author_id) desc"

POPULARITY_USER_UNIQUE_COMMENTS_REPLIES = "with firstgroup as (select count(distinct com.author_id) as count, " \
                                          "parent.author_id as par from comments as com left join comments as parent " \
                                          "on parent.id = com.parentcomment_id group by com.parentcomment_id, parent.author_id) " \
                                          "select authors.id, authors.name, round(cast(sum(count) as decimal) / " \
                                          "(select count(*) from authors), 2) as popularity from firstgroup " \
                                          "inner join authors on par = authors.id group by authors.id, authors.name " \
                                          "order by sum(count) desc"

COMMENT_POST_AUTHOR_RELATION = "select comments.author_id, comments.post_id, authors.name, authors.id, count(authors.id) " \
                               "from comments inner join posts on comments.post_id = posts.id " \
                               "inner join authors on authors.id = posts.author_id " \
                               "group by comments.author_id, comments.post_id, authors.name, authors.id " \
                               "order by comments.author_id, authors.name"

POST_COUNT = "select author_id, count(id) from posts group by author_id "

AUTHORS = "select id, name from authors"

FREQUENCY = "select author_id, min(date), max(date), count(date) as quantity, " \
                              "DATE_PART('day', max(date) - min(date)) as days, " \
                              "(DATE_PART('year', max(date)) - DATE_PART('year', min(date))) as years, " \
                              "round(DATE_PART('day', max(date) - min(date))/30) as months " \
                              "from posts group by author_id order by quantity desc;"

COMMENT_COUNT = "select author_id, count(author_id) from comments group by author_id"

FIRST_COMMENT_DATE = "select author_id, date(min(date)) from comments group by author_id order by author_id"

FIRST_POST_DATE = "select author_id, date(min(date)) from posts group by author_id order by author_id"

AUTHOR_POST_DATE = "select author_id, id, date(date) from posts order by date"

AUTHOR_COMMENT_DATE = "select author_id, id, date(date) from comments order by date"


POPULARITY_TIME_SLOT = "with posts_count(a_id, a_post_id, count_a) as (select posts.author_id, post_id, count(post_id) " \
                       "from comments inner join posts on comments.post_id = posts.id group by post_id, posts.author_id, " \
                       "comments.date having comments.date >= '2008-01-01' and comments.date <='2008-01-31' " \
                       "order by posts.author_id) " \
                       "select a_id, authors.name, round(sum(count_a) / (select count(*) from authors), 5) as popularity " \
                       "from posts_count inner join authors on a_id = authors.id " \
                       "group by a_id, authors.name order by sum(count_a) / (select count(*) from authors) desc"

COMMENT_USER_RELATION = "select id, date, author_id, parentcomment_id from comments"

ACTIVE_USERS_BY_MONTH = "SELECT COUNT(*) AS total FROM " \
                        "(select distinct author_id from posts where date >= '2008-01-01' and date <='2008-01-31' " \
                        "union " \
                        "select distinct author_id from comments where date >= '2008-01-01' and date <='2008-01-31') as a"

POPULARITY_ACTIVE_BY_MONTH = "with posts_count(a_id, a_post_id, count_a) as (select posts.author_id, post_id, count(post_id) from comments inner join posts on comments.post_id = posts.id group by post_id, posts.author_id, comments.date having comments.date >= '2010-04-01'and comments.date < '2010-05-01' order by posts.author_id) select a_id, authors.name, round(sum(count_a) / 4200, 3) as popularity from posts_count inner join authors on a_id = authors.id group by a_id, authors.name order by sum(count_a) / (select count(*) from authors) desc"

POPULARITY_NO_REPLIES_BY_MONTH = "with posts_count(a_id, a_post_id, count_a) as (select posts.author_id, post_id, count(post_id) from comments inner join posts on comments.post_id = posts.id where comments.parentcomment_id is NULL group by post_id, posts.author_id, comments.date having comments.date >= '2010-04-01'and comments.date < '2010-06-01' order by posts.author_id) select a_id, authors.name, round(sum(count_a) / 5200, 3) as popularity from posts_count inner join authors on a_id = authors.id group by a_id, authors.name order by sum(count_a) / (select count(*) from authors) desc"

POPULARITY_ONLY_REPLIES = "with firstgroup as (select count(com.parentcomment_id) as count, parent.author_id as par from comments as com left join comments as parent on parent.id = com.parentcomment_id where com.date >= '2010-04-01' and com.date <'2010-06-01' group by com.parentcomment_id, parent.author_id) select authors.id, authors.name, round(cast(sum(count) as decimal) / 5200, 2) as popularity from firstgroup inner join authors on par = authors.id group by authors.id, authors.name order by sum(count) desc"

POST_COUNT_BY_MONTH = "select author_id, count(id) from posts where date >= '2008-01-01' and date < '2008-02-01' group by author_id "

COMMENT_AUTHORS_RELATION = "SELECT users.author_id as autor_odp_na_kom, parent.author_id as autor_kom FROM comments AS users JOIN comments AS parent ON parent.id = users.parentcomment_id where users.parentcomment_id is not null"

AUTHOR_POST_COUNT_100 = "select author_id from posts group by author_id having count(id) >= 100"

POST_WORD_COUNT = "select author_id, sum(array_length(regexp_split_to_array(content, '\s'),1)) from posts group by author_id"

COMMENT_WORD_COUNT = "select author_id, sum(array_length(regexp_split_to_array(content, '\s'),1)) from comments group by author_id"

COMMENT_AUTHORS_RELATION_BY_MONTH = "SELECT users.author_id as autor_odp_na_kom, parent.author_id as autor_kom FROM comments AS users JOIN comments AS parent ON parent.id = users.parentcomment_id where users.parentcomment_id is not null and users.date >= '2010-04-01' and users.date <'2010-06-01'"