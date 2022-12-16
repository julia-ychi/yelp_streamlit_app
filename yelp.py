# Your Name
# si649f20 Altair transforms 2

# imports we will use
import streamlit as st
import pandas as pd
import math
import pyterrier as pt
import fastrank
import os

if not pt.started():
    pt.init()

def get_index(index_path, docs):
    if not os.path.exists(index_path + "/data.properties"):
        indexer = pt.DFIndexer(index_path, overwrite=True)
        index_ref = indexer.index(docs["text"], docs["docno"])
    else:
        index_ref = pt.IndexRef.of(index_path + "/data.properties")
    index = pt.IndexFactory.of(index_ref)
    return index

def yelp_search(your_query):
    qrels = pd.read_pickle("qrels.pkl")
    review_docs = qrels[["docno", "text"]].drop_duplicates()
    review_docs.columns = ["docno", "text"]
    restaurant_docs = qrels[["docno", "name"]].drop_duplicates()
    restaurant_docs.columns = ["docno", "text"]
    category_docs = qrels[["docno", "categories"]].drop_duplicates()
    category_docs.columns = ["docno", "text"]
    review_index_path = "./review_index"
    review_index = get_index(review_index_path, review_docs)
    restaurant_index_path = "./restaurant_index"
    restaurant_index = get_index(restaurant_index_path, restaurant_docs)
    category_index_path = "./category_index"
    category_index = get_index(category_index_path, category_docs)
    train_topics = qrels[["qid", "query"]][qrels["train_valid"] == "train"].drop_duplicates()
    valid_topics = qrels[["qid", "query"]][qrels["train_valid"] == "valid"].drop_duplicates()
    features = qrels[["docno", "name", "categories", "stars_review", "useful", "funny", "cool", "date", "stars_restaurant", "review_count"]].drop_duplicates()
    features["year"] = pd.to_datetime(features["date"]).dt.year
    feature_dict = {}
    for i in range(len(features)):
        feature_dict[features.iloc[i]["docno"]] = {"stars_review": features.iloc[i]["stars_review"],
                                                "useful": features.iloc[i]["useful"],
                                                "funny": features.iloc[i]["funny"],
                                                "cool": features.iloc[i]["cool"],
                                                "year": features.iloc[i]["year"],
                                                "stars_restaurant": features.iloc[i]["stars_restaurant"],
                                                "review_count": features.iloc[i]["review_count"],
                                                #"score_restaurant": features.iloc[i]["score_restaurant"],
                                                #"score_category": features.iloc[i]["score_category"],
                                                }

    qrels["label"] = qrels["label"].astype(int)
    review_bm25 = pt.BatchRetrieve(review_index, wmodel="BM25", controls={"bm25.b" : 0.425}) >> pt.pipelines.PerQueryMaxMinScoreTransformer()
    restaurant_bm25 = pt.BatchRetrieve(restaurant_index, wmodel="BM25") >> pt.pipelines.PerQueryMaxMinScoreTransformer()
    category_bm25 = pt.BatchRetrieve(category_index, wmodel="BM25") >> pt.pipelines.PerQueryMaxMinScoreTransformer()

    bm25 = 0.8 * review_bm25 + 0.15 * restaurant_bm25 + 0.05 * category_bm25

    feats = bm25 >> (pt.transformer.IdentityTransformer() 
                    ** 
                    pt.apply.doc_score(lambda row: feature_dict[row["docno"]]["stars_review"])
                    ** 
                    pt.apply.doc_score(lambda row: feature_dict[row["docno"]]["stars_restaurant"])
                    ** 
                    pt.apply.doc_score(lambda row: feature_dict[row["docno"]]["year"])
                    **
                    pt.apply.doc_score(lambda row: feature_dict[row["docno"]]["useful"] + feature_dict[row["docno"]]["funny"] + feature_dict[row["docno"]]["cool"])
                    )

    fnames = ["BM25", "Stars review", "Stars restaurant", "Review year", "Review likes"]
    train_request = fastrank.TrainRequest.coordinate_ascent()

    params = train_request.params
    params.init_random = True
    params.normalize = True
    params.seed = 1234567
    #params.num_restarts = 10
    #params.num_max_iterations = 30

    ca_pipe = feats >> pt.ltr.apply_learned_model(train_request, form='fastrank')

    ca_pipe.fit(train_topics, qrels)

    res = ca_pipe(your_query).dropna()
    res = pd.merge(res, qrels, on = ['docno'])
    res = res[res['train_valid'] == 'test']
    return res

def look_for_features(res,features):
    keep = []
    for i in range(len(res)):
        attribute = eval(res.iloc[i]['attributes'])
        ifkeep = True
        for feature in features:
            if feature in attribute and attribute[feature] == 'True':
                continue
            else:
                ifkeep = False
                break
        keep.append(ifkeep)
    return keep     

#Title
st.title("Yelp for Ann Arbor")


restaurant = pd.read_csv("restaurants_ann_arbor.csv")
nostar = '0star.png'
fullstar = 'fullstar.png'
halfstar='halfstar.png'

form = st.form(key='my_form')
title = form.text_input(label='Search Yelp:')
stars = form.select_slider(
    'Search for restaurants with at least a rating of:',
    options=['0', '1', '2', '3', '4', '5'])
features = form.multiselect(
    'Features',
    ['Takes Reservations', 'Offers Delivery', 'Offers Takeout', 'Good For Kids', 'Waiter Service', 'Accepts Credit Cards', 'Accepts Apple Pay'])

is_open = form.checkbox('is open')
submit_button = form.form_submit_button(label='Submit')
# title = st.text_input('Search')

# res = data[data['name'] == title]
# st.dataframe(res.iloc[0])
if submit_button:
    res = yelp_search(title)
    if is_open:
        res = res[res.is_open == 1]
    star_filter = int(stars)
    res = res[res.stars_restaurant >= star_filter]
    if features:
        feat_filter = look_for_features(res, features)
        res['keep'] = feat_filter
        res = res[res.keep == True]
    urls = pd.merge(res, restaurant, on=['business_id'])['url'].tolist()
    # st.write(urls)
    res['url'] = urls
    if len(res) == 0:
        st.subheader("Oops!... Nothing found.")
        st.write("**Suggestions for improving your results:**")
        st.markdown("- Drop some features in the filter")
        st.markdown("- Try a more general search")
        st.markdown("- Check the spelling or try alternate spellings")

    else:
        st.write(str(len(res)) + " results are found for you:")
        # st.dataframe(res)
        for i in range(len(res)):
            name = str(i+1) + ". " + res.iloc[i]['name']
            # st.write("[Explore more](%s)" % url)
            st.subheader(name)
            
            rating = res.iloc[i]['stars_restaurant']
            stars = [fullstar] * int(rating)
            if rating != int(rating):
                stars.append(halfstar)
                stars = stars + [nostar] * (4-int(rating))
            else:
                stars = stars + [nostar] * (5-int(rating))

            st.image(stars, width=25)
            st.write('**Overall rating:** ', rating, '         (', str(res.iloc[i]['review_count']), " reviews)")
            address = res.iloc[i]['address'] + ", " + res.iloc[i]['city'] + ", " + res.iloc[i]['state']
            st.write("**Address:** ", address)
            labels = res.iloc[i]['categories'].split(", ")
            html_str = ""
            for label in labels:
                html_str += f"<p><code>{label}</code></p>"
            # label_write = label_write[:-2]
            st.markdown('**Category:** ')
            st.markdown(html_str, unsafe_allow_html=True)
            
            feature_write = ""
            features = eval(res.iloc[i]['attributes'])
            feature_labels = []
            if features:
                for key in features:
                    if features[key] == 'True':
                        feature_labels.append(key)
                        feature_write += f"<p><code>{key}</code></p>"
                        if len(feature_labels) == 6:
                            break
                    else:
                        break
                # feature_write = feature_write[:-2]
                st.write('**Features:** ')
                st.markdown(feature_write, unsafe_allow_html=True)
            url = res.iloc[i]['url']
            st.write("[Explore more](%s)" % url)
            st.markdown("---")

            