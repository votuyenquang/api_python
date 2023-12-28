from flask import Flask, jsonify, request
from flask_cors import CORS
from wordcloud import WordCloud
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

app = Flask(__name__)
CORS(app)


@app.route('/api/hello', methods=['GET'])
def hello():
    return jsonify(message="Hello, World!")


# @app.route('/api/greet/<name>', methods=['GET'])
# def greet(name):
#     return jsonify(message=f"Hello, {name}!")


@app.route('/api/recommendation', methods=['POST'])
def sum_numbers():
    df = pd.read_csv("input/data_product_final.csv")

    # check the number of rows and columns
    df.shape
    # dataset_loc = df[df['user_id'].str.contains(id_can_loc, na=False) ]
    # print("Number datatset")
    # print(dataset_loc)

    # Check for missing values
    def check_missing_values(dataframe):
        return dataframe.isnull().sum()

    # print(check_missing_values(df))
    # df[df.rating_count.isnull()]

    # Remove rows with missing values in the rating_count column
    # df.dropna(subset=['rating_count'], inplace=True)
    # print(check_missing_values(df))

    # Check for duplicates
    def check_duplicates(dataframe):
        return dataframe.duplicated().sum()

    # print(check_duplicates(df))

    # Check data types
    def check_data_types(dataframe):
        return dataframe.dtypes

    # print(check_data_types(df))

    df['discounted_price'] = df['discounted_price'].astype(
        str).str.replace('₹', '').str.replace(',', '').astype(float)
    df['actual_price'] = df['actual_price'].astype(
        str).str.replace('₹', '').str.replace(',', '').astype(float)
    df['discount_percentage'] = df['discount_percentage'].astype(
        str).str.replace('%', '').astype(float)/100

    # The rating column has a value with an incorrect character, so we will exclude
    # the row to obtain a clean dataset.
    count = df['rating'].sum()
    # print(f"Total de linhas com '|' na coluna 'rating': {count}")
    df = df[df['rating'].apply(lambda x: '|' not in str(x))]
    count = df['rating'].sum()
    # print(f"Total de linhas com '|' na coluna 'rating': {count}")

    # df['rating'] = df['rating'].astype(float)
    # df['rating_count'] = df['rating_count'].astype(float)
    df['rating'] = df['rating'].astype(str).str.replace(',', '').astype(float)
    df['rating_count'] = df['rating_count'].astype(str).str.replace(',', '').astype(float)

    # Creating the column "rating_weighted"
    df['rating_weighted'] = df['rating'] * df['rating_count']

    # df['sub_category'] = df['category']
    # df['main_category'] = df['category_detail']
    df['sub_category'] = df['category'].astype(str).str.split('|').str[-1]
    df['main_category'] = df['category'].astype(str).str.split('|').str[0]

    # Analyzing distribution of products by main category
    # Select only the top 30 main categories.
    main_category_counts = df['main_category'].value_counts()[:30]
    plt.bar(range(len(main_category_counts)), main_category_counts.values)
    plt.ylabel('Number of Products')
    plt.title('Distribution of Products by Main Category (Top 30)')
    plt.xticks(range(len(main_category_counts)), '')  # hide X-axis labels
    # plt.show()

    # Top 30 main categories
    top_main_categories = pd.DataFrame(
        {'Main Category': main_category_counts.index, 'Number of Products': main_category_counts.values})
    # print('Top 30 main categories:')
    # print(top_main_categories.to_string(index=False))

    # Analyzing distribution of products by last category
    # Select only the top 30 last categories.
    sub_category_counts = df['sub_category'].value_counts()[:30]
    plt.bar(range(len(sub_category_counts)), sub_category_counts.values)
    plt.ylabel('Number of Products')
    plt.title('Distribution of Products by Sub Category (Top 30)')
    plt.xticks(range(len(sub_category_counts)), '')  # hide X-axis labels
    # plt.show()

    # Top 30 sub categories
    top_sub_categories = pd.DataFrame(
        {'Sub Category': sub_category_counts.index, 'Number of Products': sub_category_counts.values})
    # print('Top 30 sub categories:')
    # print(top_sub_categories.to_string(index=False))

    # 3.2 Analyze the distribution of customer ratings using a histogram.
    # Plot histogram
    plt.hist(df['rating'])
    plt.xlabel('Rating')
    plt.ylabel('Number of Reviews')
    plt.title('Distribution of Customer Ratings')
    # plt.show()

    # Create table with values per cluster
    # bins = [0, 1, 2, 3, 4, 5]  # Define bin edges
    # df['cluster'] = pd.cut(df['rating'], bins=bins, include_lowest=True, labels=[
    #                        '0-1', '1-2', '2-3', '3-4', '4-5'])
    # table = df['cluster'].value_counts().reset_index().sort_values('index').rename(
    #     columns={'index': 'Cluster', 'cluster': 'Number of Reviews'})
    # print(table)

    # Calculate the top main categories
    top = df.groupby(['main_category'])['rating'].mean(
    ).sort_values(ascending=False).head(10).reset_index()

    # Create a bar plot
    plt.bar(top['main_category'], top['rating'])

    # Add labels and title
    plt.xlabel('main_category')
    plt.ylabel('Rating')
    plt.title('Top main_category by Rating')

    # Rotate x-axis labels
    plt.xticks(rotation=90)

    # Show the plot
    # plt.show()
    ranking = df.groupby('main_category')['rating'].mean(
    ).sort_values(ascending=False).reset_index()
    # print(ranking)

    # Calculate the top sub categories
    top = df.groupby(['sub_category'])['rating'].mean().sort_values(
        ascending=False).head(10).reset_index()

    # Create a bar plot
    plt.bar(top['sub_category'], top['rating'])

    # Add labels and title
    plt.xlabel('sub_category')
    plt.ylabel('Rating')
    plt.title('Top sub_category by Rating')

    # Rotate x-axis labels
    plt.xticks(rotation=90)

    # Show the plot
    # plt.show()
    ranking = df.groupby('sub_category')['rating'].mean(
    ).sort_values(ascending=False).reset_index()
    # print(ranking)

    # sort the means in descending order
    mean_discount_by_category = df.groupby(
        'main_category')['discount_percentage'].mean()
    mean_discount_by_category = mean_discount_by_category.sort_values(
        ascending=True)

    # create the horizontal bar chart
    plt.barh(mean_discount_by_category.index, mean_discount_by_category.values)
    plt.title('Discount Percentage by Main Category')
    plt.xlabel('Discount Percentage')
    plt.ylabel('Main Category')
    # plt.show()

    table = pd.DataFrame({'Main Category': mean_discount_by_category.index,
                         'Mean Discount Percentage': mean_discount_by_category.values})

    # print(table)

    # sort the means in descending order
    mean_discount_by_sub_category = df.groupby(
        'sub_category')['discount_percentage'].mean().head(15)
    mean_discount_by_sub_category = mean_discount_by_sub_category.sort_values(
        ascending=True)

    # create the horizontal bar chart
    plt.barh(mean_discount_by_sub_category.index,
             mean_discount_by_sub_category.values)
    plt.title('Discount Percentage by Sub Category')
    plt.xlabel('Discount Percentage')
    plt.ylabel('Sub Category')
    # plt.show()

    table = pd.DataFrame({'Sub Category': mean_discount_by_sub_category.index,
                         'Mean Discount Percentage': mean_discount_by_sub_category.values})

    # print(table)

    # 3.3 Analyze the reviews by creating word clouds or frequency tables of the most common words used in the reviews.
    reviews_text = ' '.join(df['review_content'].dropna().values)
    wordcloud = WordCloud(width=800, height=800, background_color='white',
                          min_font_size=10).generate(reviews_text)
    plt.figure(figsize=(8, 8), facecolor=None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)
    # plt.show()

    # 3.4 Perform statistical analysis to identify any correlations between different features, such as the relationship between product price and customer rating.
    # Drop non-numeric columns
    numeric_cols = df.select_dtypes(include=['float64', 'int64'])
    correlation_matrix = numeric_cols.corr()

    # Print the correlation matrix
    # print(correlation_matrix)

    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    # plt.show()
    
    
    # label_encoders = {}

    # # Lặp qua các user_id duy nhất và tạo một LabelEncoder cho mỗi user_id
    # for user_id in df['user_id'].unique():
    #     le = LabelEncoder()
    #     df.loc[df['user_id'] == user_id, 'user_id_encoded'] = le.fit_transform(df.loc[df['user_id'] == user_id, 'user_id'])
    #     label_encoders[user_id] = le

    le = LabelEncoder()
    df['user_id_encoded'] = le.fit_transform(df['user_id'])

    
    # Create a new dataframe with the user_id frequency table
    freq_table = pd.DataFrame({'User ID': df['user_id_encoded'].value_counts(
    ).index, 'Frequency': df['user_id_encoded'].value_counts().values})
 

# # Đặt lại tên cột
#     value_counts.columns = ['user_id_encoded', 'Frequency']

# # Hiển thị DataFrame mới
#     print(value_counts) 
   
    # Chuyển đổi Series thành DataFrame
    # result_df = pd.DataFrame({'user_id': df['user_id'].unique(), 'user_id_encoded': le.transform(df['user_id'].unique())})


    # Hiển thị DataFrame mới
    # print(result_df)

    # Display the dataframe
    print(freq_table)
    id_example = freq_table.iloc[0, 0]

    def recommend_products(df, new_user_id):
        # Use TfidfVectorizer to transform the product descriptions into numerical feature vectors
        # le = LabelEncoder()
        # df['user_id_encoded'] = le.fit_transform(df['user_id'])
        print("++++++++++++++++++++++++++++++++++++++++++++++++", new_user_id)
        user_id_encoded = le.transform([new_user_id])[0]
        tfidf = TfidfVectorizer(stop_words='english')
        df['about_product'] = df['about_product'].fillna(
            '')  # fill NaN values with empty string
        tfidf_matrix = tfidf.fit_transform(df['about_product'])
        # Get the purchase history for the user
        user_history = df[df['user_id_encoded'] == user_id_encoded]

        # Use cosine_similarity to calculate the similarity between each pair of product descriptions
        # only for the products that the user has already purchased
        indices = user_history.index.tolist()
        if indices:
            # Create a new similarity matrix with only the rows and columns for the purchased products
            cosine_sim_user = cosine_similarity(
                tfidf_matrix[indices], tfidf_matrix)

            # Create a pandas Series with product indices as the index and product names as the values
            products = df.iloc[indices]['product_name']
            indices = pd.Series(products.index, index=products)

            # Get the indices and similarity scores of products similar to the ones the user has already purchased
            similarity_scores = list(enumerate(cosine_sim_user[-1]))
            similarity_scores = [(i, score) for (
                i, score) in similarity_scores if i not in indices]

            # Sort the similarity scores in descending order
            similarity_scores = sorted(
                similarity_scores, key=lambda x: x[1], reverse=True)

            # Get the indices of the top 5 most similar products
            top_products = [i[0] for i in similarity_scores[1:8]]

            # Get the names of the top 5 most similar products
            recommended_products = df.iloc[top_products]['product_name'].tolist(
            )
               # Get the names of the top 5 most similar products
            recommended_products_id = df.iloc[top_products]['product_id'].tolist(
            )

            # Get the reasons for the recommendation
            score = [similarity_scores[i][1] for i in range(7)]

            # Create a DataFrame with the results
            results_df = pd.DataFrame({'Id Encoded': [user_id_encoded] * 7,
                                       'recommended product': recommended_products,
                                       'id_product' : recommended_products_id,
                                       'score recommendation': score})
            print(results_df)
            results_list =  list(set(recommended_products_id))
            return results_list

        else:
            print("No purchase history found.")
            return None
    # dft = pd.read_csv("input/data_be.csv")
    print("=====================")
    # new_user_id = '227566c0-2d6c-11ec-9cf0-c9d95f18e810'
    req = request.get_json()
    new_user_id = req.get('idUser')
    # if new_user_id == "1": 
    #     new_user_id = '227566c0-2d6c-11ec-9cf0-c9d95f18e810'
    # print("=====================", new_user_id)
    # le = label_encoders[new_user_id]
    # new_user_id_encoded = le.transform([new_user_id])[0]
    results = recommend_products(df, new_user_id)
    print(results)
    return results


# Chạy ứng dụng trên cổng 5000
if __name__ == '__main__':
    app.debug = True
    app.run(host='127.0.0.1', debug=True)
